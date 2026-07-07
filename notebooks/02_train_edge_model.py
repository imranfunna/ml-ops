# Databricks notebook source
# MAGIC %pip install --quiet scikit-learn skl2onnx onnxruntime onnx

# COMMAND ----------

# MAGIC %md
# MAGIC # 02 — Edge model (categorie-classifier)
# MAGIC
# MAGIC **Doel**: elk ticket krijgt binnen milliseconden een `category_label` zodat
# MAGIC het naar de juiste squad gerouteerd kan worden.
# MAGIC
# MAGIC | Beslissing | Reden |
# MAGIC |------------|-------|
# MAGIC | scikit-learn Pipeline | Geen Spark Connect ML gRPC-limiet (256 MB), lichtgewicht, CPU-only |
# MAGIC | TF-IDF + LogisticRegression | Lichtgewicht > laag latency + interpretable weights per class |
# MAGIC | GridSearchCV met param-grid | Automatisch tunen van `C` (regularisatie) |
# MAGIC | MLflow pyfunc + expliciete logging | Reproduceerbaar, artefacten in **UC Model Registry** |
# MAGIC | Quality-gate voor promotion | Alleen modellen met macro-F1 >= `MIN_F1_FOR_PROMOTION` krijgen `@champion` alias |
# MAGIC
# MAGIC > **Waarom sklearn i.p.v. Spark ML?**  
# MAGIC > Databricks Serverless gebruikt Spark Connect, dat een harde 256 MB limiet
# MAGIC > heeft op model-serialisatie via gRPC. De Spark ML Pipeline (HashingTF + IDF +
# MAGIC > LogReg) overschrijdt die limiet (~300-600 MB). sklearn heeft dit probleem
# MAGIC > niet: het model wordt als compact MLflow pyfunc artefact opgeslagen en via
# MAGIC > `spark_udf()` als UDF gedistribueerd over het cluster.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as SkLogReg
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from mlflow.tracking import MlflowClient
import os, json

# COMMAND ----------

log_pipeline_event(spark, "train_edge_model", "started")

mlflow.set_experiment(EXPERIMENT_PATH)

# Gold data → pandas (support tickets = klein genoeg voor single node)
pdf_train = spark.table(GOLD_TRAIN).select("text_clean", "category_label").toPandas()
pdf_val   = spark.table(GOLD_VAL  ).select("text_clean", "category_label").toPandas()
pdf_test  = spark.table(GOLD_TEST ).select("text_clean", "category_label").toPandas()

print(f"train={len(pdf_train):,}  val={len(pdf_val):,}  test={len(pdf_test):,}")
labels = sorted(pdf_train["category_label"].unique().tolist())
print(f"n_classes={len(labels)} → {labels}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline definitie + Hyperparameter-tuning met GridSearchCV

# COMMAND ----------

sk_pipe = SkPipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                              min_df=2, max_features=500, sublinear_tf=True)),
    ("clf",   SkLogReg(max_iter=200, solver="liblinear", multi_class="auto")),
])

param_grid = {"clf__C": [1.0, 10.0, 100.0]}

gs = GridSearchCV(sk_pipe, param_grid, cv=3, scoring="f1_macro",
                  refit=True, n_jobs=-1)
gs.fit(pdf_train["text_clean"].astype(str), pdf_train["category_label"])
best_pipe = gs.best_estimator_
print(f"Best C = {gs.best_params_['clf__C']}, CV F1 = {gs.best_score_:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluatie op val + test

# COMMAND ----------

metrics = {}
for split_name, pdf in [("val", pdf_val), ("test", pdf_test)]:
    y_true = pdf["category_label"]
    y_pred = best_pipe.predict(pdf["text_clean"].astype(str))
    metrics[f"{split_name}_f1"]                = f1_score(y_true, y_pred, average="macro")
    metrics[f"{split_name}_accuracy"]          = accuracy_score(y_true, y_pred)
    metrics[f"{split_name}_weightedPrecision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics[f"{split_name}_weightedRecall"]    = recall_score(y_true, y_pred, average="weighted", zero_division=0)

for k, v in metrics.items():
    print(f"  {k}: {v:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow: log + register als pyfunc
# MAGIC
# MAGIC We wrappen de sklearn pipeline als een `mlflow.pyfunc.PythonModel` zodat
# MAGIC het model in `04_deploy_and_infer` als Spark UDF geladen kan worden via
# MAGIC `mlflow.pyfunc.spark_udf()`.

# COMMAND ----------

class EdgeClassifier(mlflow.pyfunc.PythonModel):
    """Sklearn-based edge classifier wrapped as MLflow pyfunc."""

    def load_context(self, context):
        import pickle, json as _json
        with open(context.artifacts["pipeline"], "rb") as f:
            self.pipeline = pickle.load(f)
        with open(context.artifacts["labels"], "r") as f:
            self.labels = _json.load(f)["labels"]

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        texts = model_input["text_clean"].fillna("").astype(str)
        preds = self.pipeline.predict(texts)
        probas = self.pipeline.predict_proba(texts)
        confs = probas.max(axis=1)
        return pd.DataFrame({
            "predicted_category": preds,
            "category_confidence": confs,
        })

# COMMAND ----------

# Serialize artifacts
import pickle

os.makedirs("/tmp/edge_model", exist_ok=True)
with open("/tmp/edge_model/pipeline.pkl", "wb") as f:
    pickle.dump(best_pipe, f)

labels_list = sorted(best_pipe.named_steps["clf"].classes_.tolist())
with open("/tmp/edge_model/labels.json", "w") as f:
    json.dump({"labels": labels_list}, f)

# COMMAND ----------

with mlflow.start_run(run_name="edge_classifier") as run:
    mlflow.log_metrics(metrics)

    mlflow.log_params({
        "max_features": 500,
        "best_C":       gs.best_params_["clf__C"],
        "cv_folds":     3,
        "n_train":      len(pdf_train),
        "n_classes":    len(labels),
    })
    mlflow.set_tags({"model_type": "edge_classifier",
                     "framework":  "sklearn+pyfunc",
                     "dataset":    "bitext"})

    # Confusion matrix als artefact
    from sklearn.metrics import confusion_matrix
    y_pred_test = best_pipe.predict(pdf_test["text_clean"].astype(str))
    cm = confusion_matrix(pdf_test["category_label"], y_pred_test, labels=labels_list)
    cm_df = pd.DataFrame(cm, index=labels_list, columns=labels_list)
    cm_path = "/tmp/flowsure_confusion.csv"
    cm_df.to_csv(cm_path)
    mlflow.log_artifact(cm_path, "evaluation")

    # Model signature
    input_example = pd.DataFrame({"text_clean": ["how do i reset my password?"]})
    signature = mlflow.models.infer_signature(
        input_example,
        pd.DataFrame({"predicted_category": ["ACCOUNT"],
                       "category_confidence": [0.95]}),
    )
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=EdgeClassifier(),
        artifacts={
            "pipeline": "/tmp/edge_model/pipeline.pkl",
            "labels":   "/tmp/edge_model/labels.json",
        },
        input_example=input_example,
        signature=signature,
        registered_model_name=EDGE_MODEL_NAME,
        pip_requirements=["scikit-learn", "pandas", "numpy"],
    )
    run_id = run.info.run_id

print(f"✅ run_id={run_id}")
print(f"   macro-F1 (test) = {metrics['test_f1']:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quality gate + UC alias promotion
# MAGIC UC kent geen "stages" meer — we gebruiken **aliases**. `@champion` = huidige
# MAGIC productie-versie; `@challenger` = wacht op review. Bij promotie schuiven
# MAGIC we simpelweg de alias-pointer; oude versies blijven bewaard.

# COMMAND ----------

client  = MlflowClient()
new_ver = latest_model_version(EDGE_MODEL_NAME)   # UC-safe helper uit _common

if metrics["test_f1"] >= MIN_F1_FOR_PROMOTION:
    client.set_registered_model_alias(EDGE_MODEL_NAME, CHAMPION_ALIAS, new_ver)
    # Ruim eventueel oude challenger-pointer op — champion is nu leidend.
    try:
        client.delete_registered_model_alias(EDGE_MODEL_NAME, CHALLENGER_ALIAS)
    except Exception:
        pass
    alias = CHAMPION_ALIAS
else:
    client.set_registered_model_alias(EDGE_MODEL_NAME, CHALLENGER_ALIAS, new_ver)
    alias = CHALLENGER_ALIAS

client.set_model_version_tag(EDGE_MODEL_NAME, new_ver, "test_f1",
                             f"{metrics['test_f1']:.4f}")
client.set_model_version_tag(EDGE_MODEL_NAME, new_ver, "run_id", run_id)
print(f"✅ {EDGE_MODEL_NAME} v{new_ver} → @{alias}")

log_pipeline_event(spark, "train_edge_model", "success",
                   f"version={new_ver} alias={alias} f1={metrics['test_f1']:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📱 On-device export — ONNX voor iOS/Android
# MAGIC
# MAGIC Eén artefact draait daarna op:
# MAGIC - **iOS** — ONNX Runtime (Swift/Obj-C pod) of via `onnx-coreml` naar Core ML
# MAGIC - **Android** — ONNX Runtime Mobile (AAR) met optionele NNAPI-acceleratie
# MAGIC - **Web** — ONNX Runtime Web (browser/PWA)
# MAGIC
# MAGIC Typische footprint: **< 5 MB**, **< 10 ms** inferentie per ticket, geen netwerk.


# COMMAND ----------

import shutil
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
import onnxruntime as ort

# --- Export naar ONNX ---
onnx_model = to_onnx(
    best_pipe,
    initial_types=[("text_clean", StringTensorType([None, 1]))],
    target_opset=15,
    options={id(best_pipe.named_steps["clf"]): {"zipmap": False}},
)

local_onnx = "/tmp/model.onnx"
with open(local_onnx, "wb") as f:
    f.write(onnx_model.SerializeToString())

# Labels-mapping meesturen — telefoon-app doet argmax → label
labels_json = "/tmp/labels.json"
with open(labels_json, "w") as f:
    json.dump({"labels": labels_list}, f)

# --- Sanity check: ONNX-runtime inferentie moet identiek zijn aan sklearn ---
sess = ort.InferenceSession(local_onnx, providers=["CPUExecutionProvider"])
sample = pdf_test["text_clean"].astype(str).head(50).to_numpy().reshape(-1, 1)
onnx_pred = sess.run(None, {"text_clean": sample})[0]
sk_pred   = best_pipe.predict(sample.ravel())
parity    = float((onnx_pred == sk_pred).mean())
print(f"ONNX ↔ sklearn parity op 50 samples = {parity:.2%}")
assert parity == 1.0, "ONNX-export wijkt af van sklearn — export niet uploaden."

# COMMAND ----------

# --- INT8 dynamic quantization → ~4× kleiner, ~2× sneller op ARM CPU's ---
from onnxruntime.quantization import quantize_dynamic, QuantType

local_onnx_int8 = "/tmp/model.int8.onnx"
quantize_dynamic(local_onnx, local_onnx_int8, weight_type=QuantType.QInt8)

# Parity-check ook op de gequantiseerde variant (mag 1 label afwijken op 50)
sess_q = ort.InferenceSession(local_onnx_int8, providers=["CPUExecutionProvider"])
q_pred = sess_q.run(None, {"text_clean": sample})[0]
parity_int8 = float((q_pred == sk_pred).mean())
print(f"INT8 ONNX ↔ sklearn parity op 50 samples = {parity_int8:.2%}")
assert parity_int8 >= 0.95, "INT8 quantisatie degradeert te veel — niet uploaden."

# COMMAND ----------

# --- Persist naar UC Volume + MLflow artifact + tag op de UC model-versie ---

shutil.copy(local_onnx,      f"{MOBILE_PATH}/model.onnx")
shutil.copy(local_onnx_int8, f"{MOBILE_PATH}/model.int8.onnx")
shutil.copy(labels_json,     f"{MOBILE_PATH}/labels.json")

# Manifest — telefoon-app leest dit om te weten welke versie geladen is
sk_test_f1 = metrics["test_f1"]
manifest = {
    "model_name": EDGE_MODEL_NAME,
    "model_version": new_ver,
    "run_id": run_id,
    "input_name": "text_clean",
    "input_shape": [None, 1],
    "input_dtype": "string",
    "output_name": "probabilities",
    "labels_file": "labels.json",
    "fp32_file": "model.onnx",
    "int8_file": "model.int8.onnx",
    "opset": 15,
    "sklearn_test_f1": sk_test_f1,
    "parity_fp32": parity,
    "parity_int8": parity_int8,
}
manifest_local = "/tmp/flowsure_edge_manifest.json"
with open(manifest_local, "w") as f:
    json.dump(manifest, f, indent=2)
shutil.copy(manifest_local, f"{MOBILE_PATH}/manifest.json")

size_fp32 = os.path.getsize(local_onnx)      / 1024
size_int8 = os.path.getsize(local_onnx_int8) / 1024

with mlflow.start_run(run_id=run_id):
    mlflow.log_artifact(local_onnx,      "mobile")
    mlflow.log_artifact(local_onnx_int8, "mobile")
    mlflow.log_artifact(labels_json,     "mobile")
    mlflow.log_artifact(manifest_local,  "mobile")
    mlflow.log_metric("onnx_sklearn_parity",       parity)
    mlflow.log_metric("onnx_int8_sklearn_parity",  parity_int8)
    mlflow.log_metric("onnx_fp32_size_kb", size_fp32)
    mlflow.log_metric("onnx_int8_size_kb", size_int8)

client.set_model_version_tag(EDGE_MODEL_NAME, new_ver, "onnx_fp32_size_kb", f"{size_fp32:.1f}")
client.set_model_version_tag(EDGE_MODEL_NAME, new_ver, "onnx_int8_size_kb", f"{size_int8:.1f}")
client.set_model_version_tag(EDGE_MODEL_NAME, new_ver, "onnx_path", f"{MOBILE_PATH}/")

print(f"✅ ONNX fp32 ({size_fp32:.1f} KB) → {MOBILE_PATH}/model.onnx")
print(f"✅ ONNX int8 ({size_int8:.1f} KB) → {MOBILE_PATH}/model.int8.onnx")
print(f"✅ Labels                        → {MOBILE_PATH}/labels.json")
print(f"✅ Manifest                      → {MOBILE_PATH}/manifest.json")
