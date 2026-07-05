# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Cloud model (retrieval-augmented responder)
# MAGIC
# MAGIC **Doel**: gegeven een klantvraag → een conceptantwoord voorstellen dat de
# MAGIC agent kan reviewen en versturen.
# MAGIC
# MAGIC ### Aanpak
# MAGIC 1. Bouw een TF-IDF index over de **canonical queries** in de knowledge base.
# MAGIC 2. Bij een nieuwe vraag: vind de top-K meest vergelijkbare KB-entries via
# MAGIC    cosine similarity en geef het antwoord van de nr. 1 terug.
# MAGIC 3. Wrap dit als een **`mlflow.pyfunc.PythonModel`** zodat het model
# MAGIC    portable is — zelfde artefact draait als batch UDF, streaming en REST endpoint.
# MAGIC
# MAGIC ### Waarom retrieval en niet een LLM-call?
# MAGIC * Geen externe kosten / latency, deterministisch reproducable.
# MAGIC * De KB is klein (< 100 entries) → in-memory nearest-neighbour is O(ms).
# MAGIC * **Toggle** aanwezig (`USE_FOUNDATION_MODEL = True`) die in productie
# MAGIC   een Databricks Foundation Model endpoint aanroept via `mlflow.deployments`.

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

import json
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# COMMAND ----------

log_pipeline_event(spark, "train_cloud_model", "started")
mlflow.set_experiment(EXPERIMENT_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Knowledge base laden

# COMMAND ----------

kb_pdf = spark.table(KB_TABLE).toPandas()
print(f"KB entries: {len(kb_pdf)}")
kb_pdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## PyFunc-wrapper voor de responder
# MAGIC * `load_context`: rebuild de TF-IDF vectorizer + KB uit de artefacten.
# MAGIC * `predict`: krijg een DataFrame binnen met kolom `text_clean` en geef
# MAGIC   antwoord + confidence terug.

# COMMAND ----------

class RetrievalResponder(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        with open(context.artifacts["vectorizer"], "rb") as f:
            import pickle
            self.vectorizer = pickle.load(f)
        with open(context.artifacts["kb_vectors"], "rb") as f:
            self.kb_vectors = np.load(f, allow_pickle=False)
        self.kb = pd.read_parquet(context.artifacts["kb_df"])

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        queries = model_input["text_clean"].fillna("").astype(str).tolist()
        q_vec   = self.vectorizer.transform(queries)
        sims    = cosine_similarity(q_vec, self.kb_vectors)
        best    = sims.argmax(axis=1)
        confs   = sims.max(axis=1)
        return pd.DataFrame({
            "suggested_response": self.kb["canonical_response"].iloc[best].values,
            "matched_intent":     self.kb["intent_label"].iloc[best].values,
            "confidence":         confs,
        })

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trainen: vectorizer fitten + artefacten serializen

# COMMAND ----------

import pickle, os
os.makedirs("/tmp/responder", exist_ok=True)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                             max_features=20_000, sublinear_tf=True)
kb_vectors = vectorizer.fit_transform(kb_pdf["canonical_query"].fillna(""))

with open("/tmp/responder/vectorizer.pkl", "wb") as f: pickle.dump(vectorizer, f)
np.save("/tmp/responder/kb_vectors.npy", kb_vectors.toarray())
kb_pdf.to_parquet("/tmp/responder/kb.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluatie
# MAGIC We meten hit-rate op de holdout: voor elke test-query — matcht het
# MAGIC top-1 antwoord met het gouden antwoord op **intent-niveau**?

# COMMAND ----------

test_pdf = spark.table(GOLD_TEST).select("text_clean", "intent_label").toPandas()
q_vec    = vectorizer.transform(test_pdf["text_clean"].fillna(""))
sims     = cosine_similarity(q_vec, kb_vectors)
best_idx = sims.argmax(axis=1)
predicted_intents = kb_pdf["intent_label"].iloc[best_idx].values
top1_hit  = (predicted_intents == test_pdf["intent_label"].values).mean()

# top-3 hit rate
top3_idx = np.argsort(-sims, axis=1)[:, :3]
top3_hit = np.mean([
    test_pdf["intent_label"].iloc[i] in kb_pdf["intent_label"].iloc[top3_idx[i]].values
    for i in range(len(test_pdf))
])
mean_conf = sims.max(axis=1).mean()
print(f"top-1 intent hit: {top1_hit:.3f}")
print(f"top-3 intent hit: {top3_hit:.3f}")
print(f"mean confidence:  {mean_conf:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow: log + register

# COMMAND ----------

with mlflow.start_run(run_name="cloud_responder") as run:
    mlflow.log_metrics({
        "top1_intent_hit": float(top1_hit),
        "top3_intent_hit": float(top3_hit),
        "mean_confidence": float(mean_conf),
        "n_kb_entries":    len(kb_pdf),
    })
    mlflow.log_params({
        "retriever":  "tfidf",
        "ngram_max":  2,
        "max_features": 20_000,
    })
    mlflow.set_tags({"model_type": "cloud_responder",
                     "framework": "sklearn+pyfunc",
                     "dataset":   "bitext_kb"})

    input_example = pd.DataFrame({"text_clean": ["how do i reset my password?"]})
    signature = mlflow.models.infer_signature(
        input_example,
        pd.DataFrame({"suggested_response": ["..."],
                      "matched_intent":     ["..."],
                      "confidence":         [0.0]}),
    )
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=RetrievalResponder(),
        artifacts={
            "vectorizer": "/tmp/responder/vectorizer.pkl",
            "kb_vectors": "/tmp/responder/kb_vectors.npy",
            "kb_df":      "/tmp/responder/kb.parquet",
        },
        input_example=input_example,
        signature=signature,
        registered_model_name=CLOUD_MODEL_NAME,
        pip_requirements=["scikit-learn", "pandas", "numpy"],
    )
    run_id = run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## UC alias promotion
# MAGIC Voor de responder gebruiken we top-3 hit rate als quality-gate (`≥ 0.5`).
# MAGIC Onder de drempel → `@challenger` (agent-review nodig). Boven → `@champion`.

# COMMAND ----------

client  = MlflowClient()
new_ver = latest_model_version(CLOUD_MODEL_NAME)   # UC-safe helper uit _common

if top3_hit >= 0.5:
    client.set_registered_model_alias(CLOUD_MODEL_NAME, CHAMPION_ALIAS, new_ver)
    try:
        client.delete_registered_model_alias(CLOUD_MODEL_NAME, CHALLENGER_ALIAS)
    except Exception:
        pass
    alias = CHAMPION_ALIAS
else:
    client.set_registered_model_alias(CLOUD_MODEL_NAME, CHALLENGER_ALIAS, new_ver)
    alias = CHALLENGER_ALIAS

client.set_model_version_tag(CLOUD_MODEL_NAME, new_ver, "top3_intent_hit",
                             f"{top3_hit:.4f}")
client.set_model_version_tag(CLOUD_MODEL_NAME, new_ver, "run_id", run_id)
print(f"✅ {CLOUD_MODEL_NAME} v{new_ver} → @{alias}")

log_pipeline_event(spark, "train_cloud_model", "success",
                   f"version={new_ver} alias={alias} top3={top3_hit:.3f}")
