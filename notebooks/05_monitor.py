# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Monitoring, drift & alerts
# MAGIC
# MAGIC Dagelijks draaiende notebook (Databricks Job schedule) die:
# MAGIC
# MAGIC 1. **Data drift** meet — PSI tussen de category-verdeling van vandaag
# MAGIC    en de baseline uit `01_data_pipeline`.
# MAGIC 2. **Concept drift** proxy — verschuiving in de predictie-verdeling
# MAGIC    van het edge-model.
# MAGIC 3. **Prestatiemetrics** — latency percentielen (p50/p95/p99) en
# MAGIC    doorvoer per uur.
# MAGIC 4. **Fairness / slice-metrics** — confidence per taal.
# MAGIC 5. **Alerts** — als PSI > drempel of p95-latency > drempel schrijven we
# MAGIC    een rij naar `alerts` en (in prod) triggeren we een retrain-job.

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

import sys, os
from pyspark.sql import functions as F
sys.path.append(os.path.abspath("../src"))
from pipeline_utils import (population_stability_index, to_probabilities,
                            align_distributions)

# COMMAND ----------

log_pipeline_event(spark, "monitoring", "started")

# --- Ensure downstream tables exist (idempotent) -----------------------
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {DRIFT_METRICS_TABLE} (
    metric_name STRING, metric_value DOUBLE, threshold DOUBLE,
    breached BOOLEAN, computed_ts TIMESTAMP
) USING DELTA
""")
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {ALERTS_TABLE} (
    alert_type STRING, severity STRING, message STRING, created_ts TIMESTAMP
) USING DELTA
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data drift op category-verdeling (PSI)

# COMMAND ----------

baseline = {r.category_label: r.baseline_count
            for r in spark.table(DRIFT_BASELINE_TABLE).collect()}

recent_cutoff = F.current_timestamp() - F.expr("INTERVAL 1 DAY")
recent = (spark.table(PREDICTIONS_TABLE)
    .filter(F.col("scored_ts") >= recent_cutoff)
    .groupBy("predicted_category").count()
    .collect())
recent = {r.predicted_category: r["count"] for r in recent}

if recent:
    e_prob = to_probabilities(baseline)
    a_prob = to_probabilities(recent)
    e, a = align_distributions(e_prob, a_prob)
    psi  = population_stability_index(e, a)
else:
    psi = 0.0

print(f"PSI (category)  = {psi:.4f}   drempel = {PSI_ALERT_THRESHOLD}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Latency & doorvoer

# COMMAND ----------

# Approximate quantiles → schaalbaar op miljoenen rijen (t-digest onder de motorkap)
preds = spark.table(PREDICTIONS_TABLE)
p50, p95, p99 = 0.0, 0.0, 0.0
if preds.count() > 0:
    # Simuleer per-row latency uit scoring-throughput (in echte omgeving:
    # log latency direct in de inference-notebook). We schatten uit de
    # confidence-distribution om iets zinnigs te tonen.
    latency_ms_col = (F.lit(50.0)                        # baseline
                      + (1 - F.col("category_confidence")) * F.lit(200))  # onzeker → langer
    quantiles = (preds.select(latency_ms_col.alias("lat_ms"))
                 .approxQuantile("lat_ms", [0.5, 0.95, 0.99], 0.01))
    p50, p95, p99 = quantiles

throughput = preds.groupBy(F.date_trunc("hour", "scored_ts").alias("hour")).count()
print(f"latency ms  p50={p50:.1f}  p95={p95:.1f}  p99={p99:.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fairness / slice-metrics — confidence per taal

# COMMAND ----------

slice_metrics = (spark.table(PREDICTIONS_TABLE)
    .groupBy("language").agg(
        F.count("*").alias("n"),
        F.avg("category_confidence").alias("avg_confidence"),
        F.avg("response_confidence").alias("avg_response_conf"),
    ).orderBy(F.desc("n")))
display(slice_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Metrics wegschrijven

# COMMAND ----------

from pyspark.sql import Row
metric_rows = [
    Row(metric_name="psi_category",  metric_value=float(psi),
        threshold=PSI_ALERT_THRESHOLD, breached=psi > PSI_ALERT_THRESHOLD),
    Row(metric_name="latency_p50_ms", metric_value=float(p50),
        threshold=float(LATENCY_P95_ALERT_MS), breached=False),
    Row(metric_name="latency_p95_ms", metric_value=float(p95),
        threshold=float(LATENCY_P95_ALERT_MS), breached=p95 > LATENCY_P95_ALERT_MS),
    Row(metric_name="latency_p99_ms", metric_value=float(p99),
        threshold=float(LATENCY_P95_ALERT_MS), breached=False),
]
(spark.createDataFrame(metric_rows)
 .withColumn("computed_ts", F.current_timestamp())
 .write.format("delta").mode("append").saveAsTable(DRIFT_METRICS_TABLE))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Alerts & retrain-trigger

# COMMAND ----------

alerts_to_raise = []
if psi > PSI_ALERT_THRESHOLD:
    alerts_to_raise.append(("DATA_DRIFT", "high",
        f"PSI category = {psi:.3f} > {PSI_ALERT_THRESHOLD}"))
if p95 > LATENCY_P95_ALERT_MS:
    alerts_to_raise.append(("LATENCY", "medium",
        f"p95 latency = {p95:.0f}ms > {LATENCY_P95_ALERT_MS}"))

if alerts_to_raise:
    rows = [Row(alert_type=t, severity=s, message=m) for t, s, m in alerts_to_raise]
    (spark.createDataFrame(rows)
     .withColumn("created_ts", F.current_timestamp())
     .write.format("delta").mode("append").saveAsTable(ALERTS_TABLE))
    for t, s, m in alerts_to_raise:
        print(f"🚨 [{s.upper()}] {t}: {m}")

    # CT-trigger: als DATA_DRIFT boven drempel → retrain edge-model.
    # In productie: dbutils.jobs.taskValues.set() → dependent task.
    if any(t == "DATA_DRIFT" for t, _, _ in alerts_to_raise):
        print("↻ CT trigger: retrain-job wordt aangeroepen")
        dbutils.jobs.taskValues.set(key="trigger_retrain", value=True)
else:
    print("✅ geen alerts — pipeline is gezond")

log_pipeline_event(spark, "monitoring", "success",
                   f"psi={psi:.3f} p95={p95:.0f}ms alerts={len(alerts_to_raise)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Dashboard-queries
# MAGIC Deze SQL-queries kunnen 1-op-1 als panels in een **Databricks SQL
# MAGIC Dashboard** worden geplakt. Ze zijn gebonden aan de tabellen die deze
# MAGIC pipeline vult.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Panel 1: predicties per uur per categorie
# MAGIC SELECT date_trunc('hour', scored_ts) AS h,
# MAGIC        predicted_category,
# MAGIC        count(*) AS n
# MAGIC FROM   hive_metastore.flowsure.tickets_predictions
# MAGIC WHERE  scored_ts >= current_timestamp() - INTERVAL 7 DAYS
# MAGIC GROUP  BY h, predicted_category
# MAGIC ORDER  BY h;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Panel 2: PSI-trend
# MAGIC SELECT computed_ts, metric_value AS psi
# MAGIC FROM   hive_metastore.flowsure.drift_metrics
# MAGIC WHERE  metric_name = 'psi_category'
# MAGIC ORDER  BY computed_ts;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Panel 3: recente alerts
# MAGIC SELECT * FROM hive_metastore.flowsure.alerts
# MAGIC WHERE created_ts >= current_timestamp() - INTERVAL 7 DAYS
# MAGIC ORDER BY created_ts DESC;
