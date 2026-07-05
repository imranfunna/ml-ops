# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Setup (eenmalig, Unity Catalog)
# MAGIC
# MAGIC * maakt UC **catalog** + **schema** + **volume** aan (idempotent)
# MAGIC * maakt landing-, checkpoint- en artefact-directories binnen het volume
# MAGIC * controleert of de sample-CSV's in de landing-volume staan
# MAGIC
# MAGIC ⚠️ **Vereisten**
# MAGIC * De workspace heeft **Unity Catalog** geactiveerd.
# MAGIC * De runnende identity heeft `CREATE CATALOG` (of de catalog bestaat al
# MAGIC   en de user heeft `USE CATALOG` + `CREATE SCHEMA` + `CREATE VOLUME`).
# MAGIC * Cluster gebruikt `data_security_mode = SINGLE_USER` **of** `USER_ISOLATION`
# MAGIC   (SHARED zonder UC-support werkt niet).

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

# --- UC namespace initialiseren -------------------------------------------
ensure_uc_objects(spark)

# Subdirs binnen het volume — dbutils.fs.mkdirs werkt op /Volumes/…-paden
dbutils.fs.mkdirs(LANDING_PATH)
dbutils.fs.mkdirs(CHECKPOINT_PATH)
dbutils.fs.mkdirs(ARTIFACT_PATH)
print(f"✅ UC gereed: {CATALOG}.{SCHEMA}   volume = {VOLUME_ROOT}")

# COMMAND ----------

# --- Sample-data check ---------------------------------------------------
# In productie landt data via Auto Loader vanuit S3/ADLS in het volume.
# Voor de demo verwacht de pipeline dat de twee CSV's in de landing-volume staan
# (upload via Catalog → Volume → Upload, of vanuit de CLI met
# `databricks fs cp <file> <volume-path>/` — zie instructies.md).

expected = ["bitext_sample.csv", "twitter_sample.csv"]
present  = {f.name for f in dbutils.fs.ls(LANDING_PATH)}
missing  = [f for f in expected if f not in present]
assert not missing, (
    f"❌ Bestanden ontbreken in {LANDING_PATH}: {missing}. "
    f"Upload ze via Catalog → Volume → Upload naar {LANDING_PATH} "
    "voor je verder gaat (zie instructies.md)."
)
print(f"✅ Alle {len(expected)} bronbestanden staan op {LANDING_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Klaar
# MAGIC Ga verder naar **`01_data_pipeline`**.
