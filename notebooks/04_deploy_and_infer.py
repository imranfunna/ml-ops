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

# MAGIC %pip install sentence-transformers

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

import time
import os
import mlflow
import mlflow.pyfunc
from pyspark.sql import functions as F, types as T
from pyspark.sql.functions import current_timestamp
import pandas as pd

os.environ["SPARKML_TEMP_DFS_PATH"] = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/checkpoints"
os.environ["MLFLOW_DFS_TMP"]       = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/checkpoints"

# COMMAND ----------

log_pipeline_event(spark, "deploy_and_infer", "started")

# --- Champion-versies laden uit UC registry -----------------------------
# `@champion` = de door de quality-gate goedgekeurde, live-scorende versie.
# Modelnamen zijn 3-level UC namespaces (catalog.schema.name) — gehardcoded
# in _common.py zodat er geen naming-drift ontstaat.

edge_uri  = f"models:/{EDGE_MODEL_NAME}@{CHAMPION_ALIAS}"
cloud_uri = f"models:/{CLOUD_MODEL_NAME}@{CHAMPION_ALIAS}"

# Edge model als Spark UDF → distributed inference (sklearn pyfunc)
edge_udf = mlflow.pyfunc.spark_udf(spark, edge_uri, result_type="struct<predicted_category:string,category_confidence:double>")
print(f"✅ edge  model geladen: {edge_uri}")

# Cloud model als Spark UDF → distributed inference
cloud_udf = mlflow.pyfunc.spark_udf(spark, cloud_uri, result_type="struct<suggested_response:string,matched_intent:string,confidence:double>")
print(f"✅ cloud model geladen: {cloud_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch-inference
# MAGIC Draait per default over alle silver-rijen zonder predictie. Latency
# MAGIC per row wordt gemeten en gelogd voor monitoring.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch-inference
# MAGIC Draait per default over alle silver-rijen zonder predictie. Latency
# MAGIC per row wordt gemeten en gelogd voor monitoring.

# COMMAND ----------

def run_batch_inference(source_df):
    """Applies edge + cloud model to a Spark DataFrame and writes predictions."""

    t0 = time.time()
    # Edge model (sklearn pyfunc UDF)
    scored = source_df.withColumn("edge_out", edge_udf(F.col("text_clean")))
    scored = (scored
        .withColumn("predicted_category",  F.col("edge_out.predicted_category"))
        .withColumn("category_confidence", F.col("edge_out.category_confidence"))
        .drop("edge_out")
    )

    # Cloud model — voeg suggested response toe
    scored = scored.withColumn("cloud_out", cloud_udf(F.col("text_clean")))
    scored = (scored
        .withColumn("suggested_response", F.col("cloud_out.suggested_response"))
        .withColumn("matched_intent",     F.col("cloud_out.matched_intent"))
        .withColumn("response_confidence", F.col("cloud_out.confidence"))
        .drop("cloud_out")
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
    scored = stream.withColumn("edge_out", edge_udf(F.col("text_clean")))
    scored = (scored
        .withColumn("predicted_category",  F.col("edge_out.predicted_category"))
        .withColumn("category_confidence", F.col("edge_out.category_confidence"))
        .drop("edge_out")
        .withColumn("cloud_out", cloud_udf(F.col("text_clean")))
        .withColumn("suggested_response", F.col("cloud_out.suggested_response"))
        .withColumn("matched_intent",     F.col("cloud_out.matched_intent"))
        .withColumn("response_confidence", F.col("cloud_out.confidence"))
        .drop("cloud_out")
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
# MAGIC Per endpoint doen we drie dingen — **fail-loud**:
# MAGIC 1. **Upsert** — idempotente create-or-update op basis van de `@champion` versie.
# MAGIC 2. **Wait-for-ready** — poll de endpoint-state tot `READY / NOT_UPDATING`.
# MAGIC 3. **Smoke-test** — echte `/invocations`-call met een sample payload.
# MAGIC
# MAGIC Als één van de drie stappen faalt voor edge of cloud → het notebook
# MAGIC raiset aan het eind. Dat zorgt dat de Databricks Workflow (en dus de CI/CD-run)
# MAGIC rood wordt bij een gefaalde deployment — geen stille regressies meer.

# COMMAND ----------

from mlflow.deployments import get_deploy_client
from mlflow.exceptions import MlflowException

DEPLOY_CLIENT = get_deploy_client("databricks")

def upsert_serving_endpoint(name: str, model_name: str) -> str:
    """Create-or-update een Model Serving endpoint voor een UC-model @champion."""
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
        DEPLOY_CLIENT.get_endpoint(name)
        DEPLOY_CLIENT.update_endpoint(endpoint=name, config=config)
        return "updated"
    except MlflowException:
        DEPLOY_CLIENT.create_endpoint(name=name, config=config)
        return "created"

def wait_until_ready(name: str, timeout_s: int = 900, poll_s: int = 20) -> dict:
    """Blokkeer tot het endpoint READY is; raise bij FAILED of timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        ep = DEPLOY_CLIENT.get_endpoint(name)
        state = (ep.get("state") or {}).get("config_update", "") + "/" + \
                (ep.get("state") or {}).get("ready", "")
        if "READY" in state and "NOT_UPDATING" in state:
            return ep
        if "FAILED" in state:
            raise RuntimeError(f"endpoint {name} update FAILED: {ep.get('state')}")
        print(f"⏳ {name}: {state} — wachten…")
        time.sleep(poll_s)
    raise TimeoutError(f"endpoint {name} niet READY binnen {timeout_s}s")

def smoke_test(name: str, payload: dict) -> dict:
    """Doet een echte /invocations call om te bewijzen dat het endpoint scoort."""
    resp = DEPLOY_CLIENT.predict(endpoint=name, inputs=payload)
    if not resp or "predictions" not in resp:
        raise RuntimeError(f"smoke-test {name} gaf geen predictions terug: {resp}")
    return resp

SAMPLE_TEXT = "how do i reset my password"
SMOKE_PAYLOADS = {
    "flowsure-edge-classifier": {"dataframe_split": {
        "columns": ["text_clean"], "data": [[SAMPLE_TEXT]]}},
    "flowsure-cloud-responder": {"dataframe_split": {
        "columns": ["text_clean"], "data": [[SAMPLE_TEXT]]}},
}

DEPLOY_SERVERLESS = False  # Set to True when running on a premium workspace with Model Serving enabled

deployment_errors = []
if DEPLOY_SERVERLESS:
    for endpoint, model in [
        ("flowsure-edge-classifier", EDGE_MODEL_NAME),
        ("flowsure-cloud-responder", CLOUD_MODEL_NAME),
    ]:
        try:
            status = upsert_serving_endpoint(endpoint, model)
            print(f"✅ endpoint {endpoint}: {status}")
            wait_until_ready(endpoint)
            print(f"✅ endpoint {endpoint}: READY")
            resp = smoke_test(endpoint, SMOKE_PAYLOADS[endpoint])
            print(f"✅ endpoint {endpoint}: smoke-test OK → {str(resp)[:120]}…")
            log_pipeline_event(spark, "serving_deploy", "success",
                               f"endpoint={endpoint} model={model}")
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"❌ endpoint {endpoint} FAILED: {msg}")
            log_pipeline_event(spark, "serving_deploy", "error",
                               f"endpoint={endpoint} err={msg}")
            deployment_errors.append((endpoint, msg))

    if deployment_errors:
        raise RuntimeError(f"serving deployment failed voor: {deployment_errors}")
else:
    print("⏭️ Skipping Serverless Model Serving deployment (DEPLOY_SERVERLESS=False)")
    log_pipeline_event(spark, "serving_deploy", "skipped", "Serverless endpoints disabled")

# COMMAND ----------

log_pipeline_event(spark, "deploy_and_infer", "success",
                   f"batch_scored={n_scored} avg_ms={avg_ms:.1f}")
