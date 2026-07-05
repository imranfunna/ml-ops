# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Deployment & Inference
# MAGIC
# MAGIC De twee `@champion`-modellen worden geladen uit de **UC Model Registry**
# MAGIC en toegepast op inkomende tickets — zowel **batch** als **streaming** —
# MAGIC met volledig logging per predictie voor monitoring.
# MAGIC
# MAGIC | Inference mode | Bron | Sink | Wanneer |
# MAGIC |----------------|------|------|---------|
# MAGIC | **Batch**      | `tickets_silver` | `tickets_predictions` | Backfill / dagelijkse batch |
# MAGIC | **Streaming**  | `incoming_tickets` (Delta) | `tickets_predictions` | Real-time triage |
# MAGIC | **REST endpoint** | Databricks Model Serving | HTTPS | Externe apps (webform, chat-widget) |

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

import time
import mlflow
import mlflow.pyfunc
from pyspark.sql import functions as F, types as T
from pyspark.sql.functions import current_timestamp
from pyspark.ml.functions import vector_to_array
import pandas as pd

# COMMAND ----------

log_pipeline_event(spark, "deploy_and_infer", "started")

# --- Champion-versies laden uit UC registry -----------------------------
# `@champion` = de door de quality-gate goedgekeurde, live-scorende versie.
# Modelnamen zijn 3-level UC namespaces (catalog.schema.name) — gehardcoded
# in _common.py zodat er geen naming-drift ontstaat.

edge_uri  = f"models:/{EDGE_MODEL_NAME}@{CHAMPION_ALIAS}"
cloud_uri = f"models:/{CLOUD_MODEL_NAME}@{CHAMPION_ALIAS}"

edge_model = mlflow.spark.load_model(edge_uri)
print(f"✅ edge  model geladen: {edge_uri}")

# Cloud model als Spark UDF → distributed inference
cloud_udf = mlflow.pyfunc.spark_udf(spark, cloud_uri, result_type="struct<suggested_response:string,matched_intent:string,confidence:double>")
print(f"✅ cloud model geladen: {cloud_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Label-index → naam mapper
# MAGIC De classifier geeft een `prediction` (double) terug — we mappen die
# MAGIC terug naar de originele category-string via de fitted StringIndexer.

# COMMAND ----------

# StringIndexer zit als voorlaatste stage in de pipeline; labels-array is
# in dezelfde volgorde als de prediction-indices.
label_names = edge_model.stages[-2].labelsArray[0]
print("labels:", list(label_names))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch-inference
# MAGIC Draait per default over alle silver-rijen zonder predictie. Latency
# MAGIC per row wordt gemeten en gelogd voor monitoring.

# COMMAND ----------

def run_batch_inference(source_df):
    """Applies edge + cloud model to a Spark DataFrame and writes predictions."""

    t0 = time.time()
    # Edge model
    scored = edge_model.transform(source_df)

    # index → naam via array-lookup
    labels_expr = F.array([F.lit(x) for x in label_names])
    scored = (scored
        .withColumn("predicted_category",
                    labels_expr.getItem(F.col("prediction").cast("int")))
        # confidence = max probability (VectorUDT → array → max)
        .withColumn("category_confidence",
                    F.array_max(vector_to_array(F.col("probability"))))
    )

    # Cloud model — voeg suggested response toe
    scored = scored.withColumn("cloud_out", cloud_udf(F.col("text_clean")))
    scored = (scored
        .withColumn("suggested_response", F.col("cloud_out.suggested_response"))
        .withColumn("matched_intent",     F.col("cloud_out.matched_intent"))
        .withColumn("response_confidence", F.col("cloud_out.confidence"))
        .drop("cloud_out", "features", "raw_features", "tokens",
              "tokens_clean", "rawPrediction", "probability", "prediction",
              "label")
    )

    # Priority-heuristiek op basis van category + confidence
    scored = scored.withColumn("priority", F.when(
        F.col("predicted_category").isin("REFUND", "PAYMENT", "CANCEL"), "high"
    ).when(F.col("category_confidence") < 0.5, "medium").otherwise("normal"))

    scored = (scored
        .withColumn("model_version_edge",  F.lit(edge_uri))
        .withColumn("model_version_cloud", F.lit(cloud_uri))
        .withColumn("scored_ts",           current_timestamp())
    )

    (scored.write.format("delta").mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(PREDICTIONS_TABLE))

    elapsed = time.time() - t0
    n       = scored.count()
    latency = 1000 * elapsed / max(n, 1)
    print(f"✅ batch: {n:,} rijen in {elapsed:.1f}s → ~{latency:.1f} ms/rij")
    return n, latency

# --- unlabeled tickets (bv. Twitter) ---
new_tickets = (spark.table(SILVER_TABLE)
    .filter(F.col("is_valid") & F.col("category_label").isNull())
    .select("ticket_id", "channel", "text_clean", "language"))
n_scored, avg_ms = run_batch_inference(new_tickets)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming-inference (Structured Streaming)
# MAGIC * Bron: `incoming_tickets` Delta-tabel — daar dropt Auto Loader in productie
# MAGIC   nieuwe tickets zodra ze binnenkomen.
# MAGIC * Sink: dezelfde `tickets_predictions` tabel (append) → downstream
# MAGIC   dashboards zien batch én streaming resultaten.
# MAGIC * `trigger(availableNow=True)` maakt de stream idempotent: draait
# MAGIC   éénmaal, verwerkt alle nieuwe files sinds vorige checkpoint, stopt.
# MAGIC   Voor continu draaien: verwissel met `trigger(processingTime='30 seconds')`.

# COMMAND ----------

# Idempotent aanmaken van de source-tabel zodat de stream altijd kan starten
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {INCOMING_STREAM_TABLE} (
    ticket_id STRING, channel STRING, text_clean STRING, language STRING
) USING DELTA
""")

def start_streaming_inference():
    stream = (spark.readStream.format("delta").table(INCOMING_STREAM_TABLE))
    scored = edge_model.transform(stream)
    labels_expr = F.array([F.lit(x) for x in label_names])
    scored = (scored
        .withColumn("predicted_category",
                    labels_expr.getItem(F.col("prediction").cast("int")))
        .withColumn("category_confidence",
                    F.array_max(vector_to_array(F.col("probability"))))
        .withColumn("cloud_out", cloud_udf(F.col("text_clean")))
        .withColumn("suggested_response", F.col("cloud_out.suggested_response"))
        .withColumn("matched_intent",     F.col("cloud_out.matched_intent"))
        .withColumn("response_confidence", F.col("cloud_out.confidence"))
        .drop("cloud_out", "features", "raw_features", "tokens",
              "tokens_clean", "rawPrediction", "probability", "prediction")
        .withColumn("priority", F.lit("normal"))
        .withColumn("model_version_edge",  F.lit(edge_uri))
        .withColumn("model_version_cloud", F.lit(cloud_uri))
        .withColumn("scored_ts",           current_timestamp())
    )
    return (scored.writeStream
        .format("delta")
        .option("checkpointLocation", f"{CHECKPOINT_PATH}/streaming_inference")
        .option("mergeSchema", "true")
        .outputMode("append")
        .trigger(availableNow=True)
        .toTable(PREDICTIONS_TABLE))

query = start_streaming_inference()
query.awaitTermination()
print("✅ streaming inference batch afgerond")

# COMMAND ----------

# MAGIC %md
# MAGIC ## REST-endpoint (Databricks Model Serving)
# MAGIC
# MAGIC De volgende cell **maakt/updatet** een serving-endpoint via de
# MAGIC MLflow Deployments API. Vereist dat je workspace Model Serving heeft
# MAGIC geactiveerd en dat de gebruiker `CAN_MANAGE` heeft op het model.
# MAGIC
# MAGIC De cel is **idempotent**: bestaat het endpoint al → update, anders → create.
# MAGIC Faalt de call (bv. geen serving in je workspace) → we loggen en gaan verder.

# COMMAND ----------

def upsert_serving_endpoint(name: str, model_name: str):
    from mlflow.deployments import get_deploy_client
    from mlflow.exceptions import MlflowException
    client = get_deploy_client("databricks")
    # UC: haal de version via alias (i.p.v. via stage-filter)
    mv = MlflowClient().get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    served_name = f"{name}-v{mv.version}"   # geen dots → geldige entity-name
    config = {
        "served_entities": [{
            "name":                     served_name,
            "entity_name":              model_name,        # 3-level UC name
            "entity_version":           mv.version,
            "workload_size":            "Small",
            "scale_to_zero_enabled":    True,
        }],
        "traffic_config": {
            "routes": [{"served_model_name": served_name, "traffic_percentage": 100}],
        },
    }
    try:
        client.get_endpoint(name)
        client.update_endpoint(endpoint=name, config=config)
        return "updated"
    except MlflowException:
        client.create_endpoint(name=name, config=config)
        return "created"

for endpoint, model in [
    ("flowsure-edge-classifier", EDGE_MODEL_NAME),
    ("flowsure-cloud-responder", CLOUD_MODEL_NAME),
]:
    try:
        status = upsert_serving_endpoint(endpoint, model)
        print(f"✅ endpoint {endpoint}: {status}")
    except Exception as e:
        print(f"⚠️  serving endpoint {endpoint} skipped: {type(e).__name__}: {e}")

# COMMAND ----------

log_pipeline_event(spark, "deploy_and_infer", "success",
                   f"batch_scored={n_scored} avg_ms={avg_ms:.1f}")
