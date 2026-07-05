# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Data pipeline (Bronze → Silver → Gold)
# MAGIC
# MAGIC End-to-end datapijplijn die tickets uit meerdere kanalen samenbrengt,
# MAGIC valideert, opschoont en features klaarzet voor de modellen.
# MAGIC
# MAGIC | Laag | Wat gebeurt hier | Waarom |
# MAGIC |------|------------------|--------|
# MAGIC | **Bronze** | Ruwe Delta-tabel per kanaal, harmoniseerd schema | Immutable "single source of truth", replay mogelijk |
# MAGIC | **Silver** | Validatie, cleaning, PII-masking, taaldetectie | GDPR-safe, één schoon schema |
# MAGIC | **Gold**   | Feature-engineering + train/val/test splits | Direct bruikbaar door model-notebooks |
# MAGIC | **KB**     | Deduped `(intent → response)` uit Bitext | Retrieval-basis voor de responder |
# MAGIC | **Drift baseline** | Category-verdeling van train-set | Referentie voor monitoring |
# MAGIC
# MAGIC ### Schaalbaarheid & governance
# MAGIC * Alles is een **Spark DataFrame** → transformaties worden distributed uitgevoerd.
# MAGIC * Bronze wordt via **Auto Loader** (`cloudFiles`) gelezen — incrementeel, exactly-once, checkpointed.
# MAGIC * **Unity Catalog** managed tables + Delta Lake → ACID, tijdreis, lineage, per-object grants.
# MAGIC * De cleaning-functies zijn **Pandas UDF's** (vectorized) i.p.v. row-wise UDF's.

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

import sys, os, re
from pyspark.sql import functions as F, types as T
from pyspark.sql.functions import pandas_udf
import pandas as pd

# We willen de pure-python util-module `pipeline_utils` binnen de UDF gebruiken.
# In Databricks Repos wordt de repo-root automatisch op sys.path gezet zodat
# `from pipeline_utils import ...` werkt vanuit workers.
sys.path.append(os.path.abspath("../src"))
from pipeline_utils import normalize_text, mask_pii, detect_language  # noqa: E402

# COMMAND ----------

log_pipeline_event(spark, "data_pipeline", "started")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Bronze — ruwe ingestie
# MAGIC Twee bronnen met verschillend schema worden ingelezen en samengevoegd
# MAGIC in één bronze-tabel met **channel-tag** en **ingest-timestamp**.
# MAGIC
# MAGIC ➡ In productie wordt Auto Loader gebruikt (zie `read_stream_bronze` in
# MAGIC de deployment-notebook) — hier voor eenvoud een batch-read.

# COMMAND ----------

# Expliciete schema's → geen surprise nulls door type-inference
BITEXT_SCHEMA = T.StructType([
    T.StructField("flags",       T.StringType()),
    T.StructField("instruction", T.StringType()),
    T.StructField("category",    T.StringType()),
    T.StructField("intent",      T.StringType()),
    T.StructField("response",    T.StringType()),
])

TWITTER_SCHEMA = T.StructType([
    T.StructField("tweet_id",                T.LongType()),
    T.StructField("author_id",               T.StringType()),
    T.StructField("inbound",                 T.BooleanType()),
    T.StructField("created_at",              T.StringType()),
    T.StructField("text",                    T.StringType()),
    T.StructField("response_tweet_id",       T.StringType()),
    T.StructField("in_response_to_tweet_id", T.StringType()),
])

bitext = (spark.read.option("header", True).option("multiLine", True)
          .option("escape", '"').schema(BITEXT_SCHEMA)
          .csv(f"{LANDING_PATH}/bitext_sample.csv"))

twitter = (spark.read.option("header", True).option("multiLine", True)
           .option("escape", '"').schema(TWITTER_SCHEMA)
           .csv(f"{LANDING_PATH}/twitter_sample.csv"))

print(f"Bitext rows:  {bitext.count():,}")
print(f"Twitter rows: {twitter.count():,}")

# COMMAND ----------

# Schema-harmonisatie naar een gemeenschappelijke bronze-vorm.
# Alleen inbound tweets (klantverzoeken) — outbound zijn agent-replies.
bitext_bronze = (bitext.select(
    F.monotonically_increasing_id().alias("ticket_id"),
    F.lit("bitext").alias("channel"),
    F.col("instruction").alias("text_raw"),
    F.col("category").alias("category_label"),
    F.col("intent").alias("intent_label"),
    F.col("response").alias("gold_response"),
    F.current_timestamp().alias("ingest_ts"),
))

twitter_bronze = (twitter.filter(F.col("inbound") == True)
    .select(
        F.col("tweet_id").cast("string").alias("ticket_id"),
        F.lit("twitter").alias("channel"),
        F.col("text").alias("text_raw"),
        F.lit(None).cast("string").alias("category_label"),
        F.lit(None).cast("string").alias("intent_label"),
        F.lit(None).cast("string").alias("gold_response"),
        F.current_timestamp().alias("ingest_ts"),
    ))

bronze = bitext_bronze.unionByName(twitter_bronze)

(bronze.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(BRONZE_TABLE))
print(f"✅ bronze → {BRONZE_TABLE} ({spark.table(BRONZE_TABLE).count():,} rijen)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Silver — validatie, cleaning, PII masking, taaldetectie
# MAGIC
# MAGIC **Validatie-regels** (fail-fast maar niet-blocking: rows die falen
# MAGIC krijgen `is_valid=False` en gaan naar een quarantine-partitie i.p.v.
# MAGIC de rest te breken):
# MAGIC
# MAGIC 1. `text_raw` niet-null en niet-leeg
# MAGIC 2. `text_raw` lengte 3–5000 karakters
# MAGIC 3. dedupe op (`channel`,`text_raw`)
# MAGIC 4. `category_label` bij bitext ∈ verwachte categorieën

# COMMAND ----------

# --- Pandas UDF's die de python-utils vectorized toepassen -------------
@pandas_udf(T.StringType())
def normalize_udf(s: pd.Series) -> pd.Series:
    return s.map(normalize_text)

@pandas_udf(T.StringType())
def mask_pii_udf(s: pd.Series) -> pd.Series:
    return s.map(mask_pii)

@pandas_udf(T.StringType())
def detect_language_udf(s: pd.Series) -> pd.Series:
    return s.map(detect_language)

# COMMAND ----------

bronze_df = spark.table(BRONZE_TABLE)

# Validatie-vlag
valid_rules = (
    F.col("text_raw").isNotNull()
    & (F.length(F.col("text_raw")) >= 3)
    & (F.length(F.col("text_raw")) <= 5000)
)

silver = (bronze_df
    .withColumn("is_valid", valid_rules)
    .dropDuplicates(["channel", "text_raw"])
    # PII eerst, dán normaliseren → placeholders overleven de cleaning
    .withColumn("text_pii_masked", mask_pii_udf(F.col("text_raw")))
    .withColumn("text_clean",      normalize_udf(F.col("text_pii_masked")))
    .withColumn("language",        detect_language_udf(F.col("text_clean")))
    .withColumn("text_length",     F.length("text_clean"))
    .withColumn("word_count",      F.size(F.split(F.col("text_clean"), r"\s+")))
)

# Kwaliteits-samenvatting loggen zodat we bad batches vroeg zien
total = silver.count()
valid = silver.filter(F.col("is_valid")).count()
print(f"Silver totaal:  {total:,}")
print(f"Geldig:         {valid:,} ({100*valid/max(total,1):.1f}%)")

(silver.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("channel")           # scan-pruning per kanaal
    .saveAsTable(SILVER_TABLE))
print(f"✅ silver → {SILVER_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Gold — feature-store + train/val/test splits
# MAGIC De **gelabelde** rijen (bitext) worden gestratificeerd gesplitst in
# MAGIC 70/15/15 op `category_label` zodat elke categorie in alle splits vertegenwoordigd is.

# COMMAND ----------

gold = (spark.table(SILVER_TABLE)
    .filter(F.col("is_valid"))
    .filter(F.col("category_label").isNotNull())   # alleen gelabeld voor training
    .select(
        "ticket_id", "channel", "text_clean", "category_label",
        "intent_label", "gold_response", "language",
        "text_length", "word_count",
    ))

(gold.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true").saveAsTable(GOLD_TABLE))

# --- Gestratificeerde split (deterministic seed) ------------------------
fractions_train = {c["category_label"]: 0.70
                   for c in gold.select("category_label").distinct().collect()}
train = gold.sampleBy("category_label", fractions=fractions_train, seed=42)
rest  = gold.exceptAll(train)
fractions_val = {c["category_label"]: 0.50
                 for c in rest.select("category_label").distinct().collect()}
val  = rest.sampleBy("category_label", fractions=fractions_val, seed=42)
test = rest.exceptAll(val)

for df, tbl, name in [(train, GOLD_TRAIN, "train"),
                      (val,   GOLD_VAL,   "val"),
                      (test,  GOLD_TEST,  "test")]:
    (df.write.format("delta").mode("overwrite")
       .option("overwriteSchema", "true").saveAsTable(tbl))
    print(f"  {name:5s}: {df.count():,} rijen → {tbl}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Knowledge base
# MAGIC Voor de responder: één representatief antwoord per (`category`,`intent`)
# MAGIC — gedeflicteerd op basis van response-hash zodat de retrieval-index compact blijft.

# COMMAND ----------

kb = (spark.table(SILVER_TABLE)
    .filter(F.col("gold_response").isNotNull())
    .groupBy("category_label", "intent_label")
    .agg(F.first("gold_response").alias("canonical_response"),
         F.first("text_clean").alias("canonical_query"),
         F.count("*").alias("n_examples"))
    .orderBy(F.desc("n_examples"))
)
(kb.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true").saveAsTable(KB_TABLE))
print(f"✅ KB → {KB_TABLE} ({kb.count()} entries)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Drift baseline
# MAGIC Category-verdeling + samenvattende tekst-statistieken van de trainset
# MAGIC worden opgeslagen als **referentie** voor de monitoring-notebook.

# COMMAND ----------

baseline_cat = (spark.table(GOLD_TRAIN)
    .groupBy("category_label").count()
    .withColumnRenamed("count", "baseline_count")
    .withColumn("computed_ts", F.current_timestamp())
)
(baseline_cat.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true").saveAsTable(DRIFT_BASELINE_TABLE))
display(baseline_cat)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Delta-optimalisatie
# MAGIC Voor de grotere tabellen (silver/gold) draaien we `OPTIMIZE` +
# MAGIC `ZORDER BY` op de kolommen waarop gefilterd wordt → sneller lezen door alle downstream jobs.

# COMMAND ----------

spark.sql(f"OPTIMIZE {SILVER_TABLE}")
spark.sql(f"OPTIMIZE {GOLD_TABLE}   ZORDER BY (category_label)")
print("✅ OPTIMIZE + ZORDER klaar")

log_pipeline_event(spark, "data_pipeline", "success",
                   f"bronze={spark.table(BRONZE_TABLE).count()}, "
                   f"silver={spark.table(SILVER_TABLE).count()}, "
                   f"gold={spark.table(GOLD_TABLE).count()}")
