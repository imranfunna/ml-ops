# Databricks notebook source
# MAGIC %md
# MAGIC # `_common` — gedeelde configuratie & helpers
# MAGIC
# MAGIC Elke pipeline-notebook draait dit met `%run ./_common` zodat catalogus-, schema-,
# MAGIC pad- en model-naam-conventies **op één plek** staan (single source of truth).

# COMMAND ----------

# --- Naming conventies ----------------------------------------------------
# Één plek voor alle constanten. Aanpassen in prod = één regel.

CATALOG = "hive_metastore"          # in Unity Catalog swap voor bv. "main"
SCHEMA  = "flowsure"

# Bronze/Silver/Gold (medallion architecture)
BRONZE_TABLE  = f"{CATALOG}.{SCHEMA}.tickets_bronze"
SILVER_TABLE  = f"{CATALOG}.{SCHEMA}.tickets_silver"
GOLD_TABLE    = f"{CATALOG}.{SCHEMA}.tickets_gold"
GOLD_TRAIN    = f"{CATALOG}.{SCHEMA}.tickets_gold_train"
GOLD_VAL      = f"{CATALOG}.{SCHEMA}.tickets_gold_val"
GOLD_TEST     = f"{CATALOG}.{SCHEMA}.tickets_gold_test"

# Kennisbank + tabellen voor deployment/monitoring
KB_TABLE                 = f"{CATALOG}.{SCHEMA}.knowledge_base"
DRIFT_BASELINE_TABLE     = f"{CATALOG}.{SCHEMA}.drift_baseline"
DRIFT_METRICS_TABLE      = f"{CATALOG}.{SCHEMA}.drift_metrics"
PREDICTIONS_TABLE        = f"{CATALOG}.{SCHEMA}.tickets_predictions"
MONITORING_LOG_TABLE     = f"{CATALOG}.{SCHEMA}.monitoring_log"
INCOMING_STREAM_TABLE    = f"{CATALOG}.{SCHEMA}.incoming_tickets"
ALERTS_TABLE             = f"{CATALOG}.{SCHEMA}.alerts"

# Storage paden (DBFS root — swap voor UC Volume in productie)
DBFS_ROOT       = "dbfs:/FileStore/flowsure"
LANDING_PATH    = f"{DBFS_ROOT}/landing"           # raw drop-zone
CHECKPOINT_PATH = f"{DBFS_ROOT}/checkpoints"       # streaming checkpoints
ARTIFACT_PATH   = f"{DBFS_ROOT}/artifacts"         # extra artefacten

# MLflow
EXPERIMENT_PATH   = "/Shared/flowsure_experiment"
EDGE_MODEL_NAME   = "flowsure_edge_classifier"
CLOUD_MODEL_NAME  = "flowsure_cloud_responder"

# Quality gates — een model dat hier onder scoort wordt NIET gepromoveerd
MIN_F1_FOR_PROMOTION      = 0.60      # macro-F1 op holdout
PSI_ALERT_THRESHOLD       = 0.25      # >0.25 = significante drift
LATENCY_P95_ALERT_MS      = 500       # p95 latency alert-drempel

# COMMAND ----------

def ensure_schema(spark):
    """Idempotent: maakt catalog+schema aan als ze nog niet bestaan."""
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")


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
