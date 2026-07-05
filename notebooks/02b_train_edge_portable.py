# Databricks notebook source
# MAGIC %md
# MAGIC # 02b · Portable Edge Model (sklearn → ONNX)
# MAGIC
# MAGIC Traint een **draagbare** variant van de edge-classifier die identiek werkt in de cloud
# MAGIC (Docker + onnxruntime) én on-device (Android / iOS via ONNX Runtime Mobile).
# MAGIC
# MAGIC Waarom een aparte pipeline i.p.v. de Spark-versie converteren?
# MAGIC Spark ML kent geen native ONNX-export voor `HashingTF + IDF`. Sklearn + skl2onnx wél.
# MAGIC De gold-set is ~50k rijen — past ruim in driver-geheugen (`toPandas()`), dus lokaal trainen is prima.

# COMMAND ----------

# MAGIC %pip install --quiet scikit-learn==1.5.2 skl2onnx==1.17.0 onnxruntime==1.19.2 onnx==1.17.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

import json
import os
import tempfile
from pathlib import Path

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline


ensure_uc_objects(spark)
mlflow.set_registry_uri("databricks-uc")

MODEL_NAME_ONNX = f"{CATALOG}.{SCHEMA}.flowsure_edge_classifier_onnx"
ONNX_VOLUME_DIR = MOBILE_PATH   # /Volumes/flowsure/mlops/artifacts/mobile
dbutils.fs.mkdirs(ONNX_VOLUME_DIR)

# COMMAND ----------

# MAGIC %md ## 1 · Load gold data into pandas

# COMMAND ----------

train_pdf = spark.table(GOLD_TRAIN).select("text_clean", "category").toPandas()
test_pdf  = spark.table(GOLD_TEST ).select("text_clean", "category").toPandas()

train_pdf = train_pdf.dropna()
test_pdf  = test_pdf.dropna()

print(f"train={len(train_pdf):,} rows | test={len(test_pdf):,} rows")

# Deterministic label encoding — same order in Python, Docker and mobile
labels = sorted(train_pdf["category"].unique().tolist())
label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
y_train = train_pdf["category"].map(label_to_idx).to_numpy()
y_test  = test_pdf ["category"].map(label_to_idx).to_numpy()

# COMMAND ----------

# MAGIC %md ## 2 · Fit sklearn pipeline (TF-IDF + LogReg)
# MAGIC Kleine vocabulary (`max_features=5000`) houdt het ONNX-model < 3 MB — telefoon-vriendelijk.

# COMMAND ----------

mlflow.set_experiment(EXPERIMENT_PATH)

with mlflow.start_run(run_name="edge_onnx_portable") as run:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2),
            max_features=5000, min_df=2, sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
        )),
    ])
    pipe.fit(train_pdf["text_clean"].to_numpy(), y_train)

    y_pred = pipe.predict(test_pdf["text_clean"].to_numpy())
    metrics = {
        "test_accuracy":  accuracy_score(y_test, y_pred),
        "test_f1":        f1_score(y_test, y_pred, average="weighted"),
        "test_precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "test_recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    mlflow.log_params({
        "vectorizer": "tfidf", "max_features": 5000,
        "ngram_range": "1-2", "clf": "logreg", "C": 1.0,
    })
    mlflow.log_metrics(metrics)
    print(json.dumps(metrics, indent=2))

    # ------------------------------------------------------------------ #
    # 3 · Export to ONNX
    # ------------------------------------------------------------------ #
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import StringTensorType

    onnx_model = convert_sklearn(
        pipe,
        initial_types=[("input_text", StringTensorType([None, 1]))],
        target_opset=17,
        options={id(pipe.named_steps["clf"]): {"zipmap": False}},
    )

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path   = Path(tmp) / "model.onnx"
        labels_path = Path(tmp) / "labels.json"
        onnx_path.write_bytes(onnx_model.SerializeToString())
        labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2))

        # Copy to UC Volume for mobile/Docker consumption
        dbutils.fs.cp(f"file:{onnx_path}",   f"{ONNX_VOLUME_DIR}/model.onnx",   True)
        dbutils.fs.cp(f"file:{labels_path}", f"{ONNX_VOLUME_DIR}/labels.json",  True)

        mlflow.log_artifact(str(onnx_path),   artifact_path="onnx")
        mlflow.log_artifact(str(labels_path), artifact_path="onnx")

        size_mb = onnx_path.stat().st_size / 1024 / 1024
        mlflow.log_metric("onnx_size_mb", size_mb)
        print(f"ONNX model size: {size_mb:.2f} MB → {ONNX_VOLUME_DIR}/model.onnx")

    # ------------------------------------------------------------------ #
    # 4 · Register in Unity Catalog (as pyfunc wrapper around ONNX)
    # ------------------------------------------------------------------ #
    class OnnxEdgeWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            import json as _json
            import onnxruntime as ort
            self._sess = ort.InferenceSession(
                context.artifacts["onnx_model"], providers=["CPUExecutionProvider"]
            )
            self._labels = _json.loads(open(context.artifacts["labels"]).read())

        def predict(self, context, model_input):
            import numpy as _np
            texts = model_input["text"].astype(str).to_numpy().reshape(-1, 1)
            preds, probs = self._sess.run(None, {"input_text": texts})
            top_idx = preds.astype(int)
            return [
                {"category": self._labels[i],
                 "confidence": float(_np.max(probs[k]))}
                for k, i in enumerate(top_idx)
            ]

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path   = Path(tmp) / "model.onnx"
        labels_path = Path(tmp) / "labels.json"
        onnx_path.write_bytes(onnx_model.SerializeToString())
        labels_path.write_text(json.dumps(labels, ensure_ascii=False))

        mlflow.pyfunc.log_model(
            artifact_path="pyfunc",
            python_model=OnnxEdgeWrapper(),
            artifacts={"onnx_model": str(onnx_path), "labels": str(labels_path)},
            pip_requirements=["onnxruntime==1.19.2", "numpy", "pandas"],
            registered_model_name=MODEL_NAME_ONNX if metrics["test_f1"] >= MIN_F1_FOR_PROMOTION else None,
        )

# COMMAND ----------

# MAGIC %md ## 5 · Promote to @champion if it beats the quality gate

# COMMAND ----------

client = MlflowClient()
if metrics["test_f1"] >= MIN_F1_FOR_PROMOTION:
    versions = client.search_model_versions(f"name='{MODEL_NAME_ONNX}'")
    latest = max(int(v.version) for v in versions)
    client.set_registered_model_alias(MODEL_NAME_ONNX, CHAMPION_ALIAS, latest)
    print(f"✓ Promoted {MODEL_NAME_ONNX} v{latest} → @champion")
else:
    print(f"✗ test_f1={metrics['test_f1']:.3f} < {MIN_F1_FOR_PROMOTION} — not registered")

# COMMAND ----------

# MAGIC %md ## 6 · Sanity check — round-trip ONNX inference

# COMMAND ----------

import onnxruntime as ort
local = "/tmp/edge_check.onnx"
dbutils.fs.cp(f"{ONNX_VOLUME_DIR}/model.onnx", f"file:{local}", True)
sess = ort.InferenceSession(local, providers=["CPUExecutionProvider"])
sample = np.array([["my invoice is wrong please help"],
                   ["cannot login to the app"]], dtype=object)
pred, prob = sess.run(None, {"input_text": sample})
for text, p, pr in zip(sample[:, 0], pred, prob):
    print(f"{text!r:45s} → {labels[int(p)]:20s} ({float(np.max(pr)):.2f})")
