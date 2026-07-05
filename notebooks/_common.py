# Databricks notebook source
# MAGIC %md
# MAGIC # `_common` — gedeelde configuratie & helpers
# MAGIC
# MAGIC Elke pipeline-notebook draait dit met `%run ./_common` zodat catalogus-,
# MAGIC schema-, pad- en model-naam-conventies **op één plek** staan.
# MAGIC
# MAGIC Alle artefacten leven in **Unity Catalog**:
# MAGIC * Tabellen        → `flowsure.mlops.<table>` (3-level UC namespace)
# MAGIC * Bestanden       → `/Volumes/flowsure/mlops/artifacts/…` (UC Volume)
# MAGIC * Model registry  → UC-registry (`mlflow.set_registry_uri("databricks-uc")`)
# MAGIC * Model promotie  → **aliases** `@champion` / `@challenger`

# COMMAND ----------

import mlflow

# --- Unity Catalog namespace ------------------------------------------------
# Één plek voor alle constanten. Aanpassen voor prod/dev = één regel.
CATALOG = "flowsure"          # UC catalog
SCHEMA  = "mlops"             # UC schema
VOLUME  = "artifacts"         # UC managed volume voor bestanden

# Bronze / Silver / Gold (medallion) — allemaal 3-level UC names
BRONZE_TABLE  = f"{CATALOG}.{SCHEMA}.tickets_bronze"
SILVER_TABLE  = f"{CATALOG}.{SCHEMA}.tickets_silver"
GOLD_TABLE    = f"{CATALOG}.{SCHEMA}.tickets_gold"
GOLD_TRAIN    = f"{CATALOG}.{SCHEMA}.tickets_gold_train"
GOLD_VAL      = f"{CATALOG}.{SCHEMA}.tickets_gold_val"
GOLD_TEST     = f"{CATALOG}.{SCHEMA}.tickets_gold_test"

# Extra tabellen
KB_TABLE              = f"{CATALOG}.{SCHEMA}.knowledge_base"
DRIFT_BASELINE_TABLE  = f"{CATALOG}.{SCHEMA}.drift_baseline"
DRIFT_METRICS_TABLE   = f"{CATALOG}.{SCHEMA}.drift_metrics"
PREDICTIONS_TABLE     = f"{CATALOG}.{SCHEMA}.tickets_predictions"
MONITORING_LOG_TABLE  = f"{CATALOG}.{SCHEMA}.monitoring_log"
INCOMING_STREAM_TABLE = f"{CATALOG}.{SCHEMA}.incoming_tickets"
ALERTS_TABLE          = f"{CATALOG}.{SCHEMA}.alerts"

# Bestandsopslag → UC Volume (governance, lineage, per-object grants)
VOLUME_ROOT     = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
LANDING_PATH    = f"{VOLUME_ROOT}/landing"          # raw drop-zone
CHECKPOINT_PATH = f"{VOLUME_ROOT}/checkpoints"      # streaming checkpoints
ARTIFACT_PATH   = f"{VOLUME_ROOT}/models"           # extra artefacten
MOBILE_PATH     = f"{VOLUME_ROOT}/mobile"           # ONNX-exports voor iOS/Android

# MLflow — UC model registry i.p.v. workspace registry
mlflow.set_registry_uri("databricks-uc")

EXPERIMENT_PATH  = "/Shared/flowsure_experiment"
EDGE_MODEL_NAME  = f"{CATALOG}.{SCHEMA}.flowsure_edge_classifier"
CLOUD_MODEL_NAME = f"{CATALOG}.{SCHEMA}.flowsure_cloud_responder"

# UC-model promotion via aliases (i.p.v. legacy stages).
# `@champion` is de door de quality-gate goedgekeurde versie die live scoort.
# `@challenger` is een kandidaat-versie die nog beoordeeld / A/B-getest wordt.
CHAMPION_ALIAS   = "champion"
CHALLENGER_ALIAS = "challenger"

# Quality gates — een model dat hier onder scoort wordt NIET promoted
MIN_F1_FOR_PROMOTION = 0.60      # macro-F1 op holdout
PSI_ALERT_THRESHOLD  = 0.25      # >0.25 = significante drift
LATENCY_P95_ALERT_MS = 500       # p95 latency alert-drempel

# COMMAND ----------

def ensure_uc_objects(spark):
    """Idempotent: catalog + schema + volume aanmaken als ze nog niet bestaan."""
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
    spark.sql(f"CREATE SCHEMA  IF NOT EXISTS {CATALOG}.{SCHEMA}")
    spark.sql(f"CREATE VOLUME  IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")


def latest_model_version(model_name: str) -> str:
    """UC-safe: geef de hoogste version-string terug voor een geregistreerd model."""
    from mlflow.tracking import MlflowClient
    versions = MlflowClient().search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(f"Geen versies gevonden voor {model_name}")
    return str(max(int(v.version) for v in versions))


def log_pipeline_event(spark, stage: str, status: str, details: str = ""):
    """Simpele audit-trail — elke run schrijft één regel naar `monitoring_log`."""
    from pyspark.sql import Row
    from pyspark.sql.functions import current_timestamp
    row = Row(stage=stage, status=status, details=details)
    (
        spark.createDataFrame([row])
        .withColumn("event_ts", current_timestamp())
        .write.mode("append").format("delta")
        .saveAsTable(MONITORING_LOG_TABLE)
    )
