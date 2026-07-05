# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Orchestrator
# MAGIC
# MAGIC Draait de hele pipeline in de juiste volgorde. Handig voor:
# MAGIC * lokaal / handmatig een volledige run
# MAGIC * als **root-task** in een Databricks Workflow / Job
# MAGIC
# MAGIC Elke stap is een aparte notebook zodat je ze óók individueel als task
# MAGIC in een Databricks Job kunt hangen (zie `databricks.yml`).

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

STAGES = [
    ("01_data_pipeline",      3600),
    ("02_train_edge_model",   7200),
    ("03_train_cloud_model",  3600),
    ("04_deploy_and_infer",   3600),
    ("05_monitor",             900),
]

for nb, timeout in STAGES:
    print(f"\n▶ start {nb}")
    dbutils.notebook.run(nb, timeout)
    print(f"✔ {nb} klaar")

print("\n🎉 volledige pipeline afgerond")
