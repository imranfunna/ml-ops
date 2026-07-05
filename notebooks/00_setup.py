# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Setup (eenmalig)
# MAGIC
# MAGIC * maakt catalog/schema aan
# MAGIC * maakt landing- en checkpoint-directories op DBFS
# MAGIC * kopieert de twee sample-CSV's naar de landing-zone zodat de pipeline er
# MAGIC   direct mee kan werken
# MAGIC
# MAGIC **Deze notebook hoef je maar één keer te draaien per workspace.**

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

# --- schema + DBFS-directories aanmaken ---------------------------------
ensure_schema(spark)
dbutils.fs.mkdirs(LANDING_PATH)
dbutils.fs.mkdirs(CHECKPOINT_PATH)
dbutils.fs.mkdirs(ARTIFACT_PATH)
print("✅ schema en DBFS-paden klaar")

# COMMAND ----------

# --- Sample-data uploaden ------------------------------------------------
# In een echte omgeving landt de data via Auto Loader vanuit S3/ADLS.
# Voor de demo verwacht deze notebook dat je de twee CSV's manueel naar
# `/FileStore/flowsure/landing` upload (zie instructies.md). Deze cel
# controleert alleen of ze aanwezig zijn.

expected = ["bitext_sample.csv", "twitter_sample.csv"]
present  = {f.name for f in dbutils.fs.ls(LANDING_PATH)}
missing  = [f for f in expected if f not in present]
assert not missing, (
    f"❌ Bestanden ontbreken in {LANDING_PATH}: {missing}. "
    "Upload ze via Data → Add data → DBFS voordat je verder gaat."
)
print(f"✅ Alle {len(expected)} bronbestanden staan op {LANDING_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Klaar
# MAGIC Ga verder naar **`01_data_pipeline`**.
