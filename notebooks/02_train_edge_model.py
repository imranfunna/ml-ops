# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Edge model (categorie-classifier)
# MAGIC
# MAGIC **Doel**: elk ticket krijgt binnen milliseconden een `category_label` zodat
# MAGIC het naar de juiste squad gerouteerd kan worden.
# MAGIC
# MAGIC | Beslissing | Reden |
# MAGIC |------------|-------|
# MAGIC | Spark ML `Pipeline` | Native distributed, artefact = één `PipelineModel` (features + estimator samen → geen train/serve skew) |
# MAGIC | TF-IDF + LogisticRegression | Lichtgewicht → laag latency + interpretable weights per class |
# MAGIC | `CrossValidator` met param-grid | Automatisch tunen van `regParam` en `numFeatures` |
# MAGIC | MLflow autolog + expliciete logging | Reproduceerbaar, artefacten in **UC Model Registry** |
# MAGIC | Quality-gate voor promotion | Alleen modellen met macro-F1 ≥ `MIN_F1_FOR_PROMOTION` krijgen `@champion` alias |

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

import mlflow
import mlflow.spark
from pyspark.ml import Pipeline
from pyspark.ml.feature import (RegexTokenizer, StopWordsRemover,
                                HashingTF, IDF, StringIndexer)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

# COMMAND ----------

log_pipeline_event(spark, "train_edge_model", "started")

mlflow.set_experiment(EXPERIMENT_PATH)
#mlflow.pyspark.ml.autolog(log_models=False)   # models loggen we handmatig i.v.m. signature

train = spark.table(GOLD_TRAIN)
val   = spark.table(GOLD_VAL)
test  = spark.table(GOLD_TEST)

print(f"train={train.count():,}  val={val.count():,}  test={test.count():,}")
labels = [r.category_label for r in train.select("category_label").distinct().collect()]
print(f"n_classes={len(labels)} → {labels}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline definitie
# MAGIC Elk transform-stadium zit in de opgeslagen `PipelineModel`. Bij inference
# MAGIC hoeven we alleen `model.transform(df)` te doen — nooit meer handmatig
# MAGIC tokenizen of vectorizen.

# COMMAND ----------

tokenizer  = RegexTokenizer(inputCol="text_clean", outputCol="tokens",
                            pattern=r"\W+", minTokenLength=2)
stopwords  = StopWordsRemover(inputCol="tokens", outputCol="tokens_clean")
hashing_tf = HashingTF(inputCol="tokens_clean", outputCol="raw_features")
idf        = IDF(inputCol="raw_features", outputCol="features")
label_idx  = StringIndexer(inputCol="category_label", outputCol="label",
                           handleInvalid="keep")
lr         = LogisticRegression(featuresCol="features", labelCol="label",
                                maxIter=50, family="multinomial")

pipeline = Pipeline(stages=[tokenizer, stopwords, hashing_tf, idf, label_idx, lr])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter-tuning met 3-fold CrossValidator

# COMMAND ----------

grid = (ParamGridBuilder()
        .addGrid(hashing_tf.numFeatures, [2**14, 2**16])
        .addGrid(lr.regParam,            [0.0, 0.01, 0.1])
        .build())

evaluator = MulticlassClassificationEvaluator(labelCol="label",
                                              predictionCol="prediction",
                                              metricName="f1")

cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=grid,
                    evaluator=evaluator,
                    numFolds=3,
                    parallelism=2,
                    seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trainen + evaluatie + logging

# COMMAND ----------

with mlflow.start_run(run_name="edge_classifier_cv") as run:
    cv_model     = cv.fit(train)
    best_model   = cv_model.bestModel
    best_lr      = best_model.stages[-1]
    best_hash_tf = best_model.stages[2]

    # --- evalueren op val + test ---
    val_pred, test_pred = best_model.transform(val), best_model.transform(test)
    metrics = {}
    for name in ["f1", "accuracy", "weightedPrecision", "weightedRecall"]:
        ev = MulticlassClassificationEvaluator(labelCol="label",
                                               predictionCol="prediction",
                                               metricName=name)
        metrics[f"val_{name}"]  = ev.evaluate(val_pred)
        metrics[f"test_{name}"] = ev.evaluate(test_pred)
    mlflow.log_metrics(metrics)

    # --- best hyperparams + dataset-fingerprints ---
    mlflow.log_params({
        "numFeatures": best_hash_tf.getNumFeatures(),
        "regParam":    best_lr.getRegParam(),
        "n_train":     train.count(),
        "n_classes":   len(labels),
    })
    mlflow.set_tags({"model_type": "edge_classifier",
                     "framework":  "sparkml",
                     "dataset":    "bitext"})

    # --- confusion matrix als artefact ---
    cm_df = (test_pred.groupBy("category_label", "prediction")
             .count().toPandas())
    cm_path = "/tmp/flowsure_confusion.csv"
    cm_df.to_csv(cm_path, index=False)
    mlflow.log_artifact(cm_path, "evaluation")

    # --- model met input/output signature loggen ---
    signature = mlflow.models.infer_signature(
        val.select("text_clean").limit(5).toPandas(),
        val_pred.select("prediction").limit(5).toPandas(),
    )
    mlflow.spark.log_model(best_model, "model",
                           signature=signature,
                           registered_model_name=EDGE_MODEL_NAME)
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
# MAGIC Het Spark-model draait niet op een telefoon (JVM + Spark nodig). Daarom
# MAGIC trainen we hetzelfde algoritme (TF-IDF + LogisticRegression) parallel in
# MAGIC **scikit-learn** op exact dezelfde gold-data, en exporteren dat naar
# MAGIC **ONNX**. Eén artefact draait daarna op:
# MAGIC - **iOS** — ONNX Runtime (Swift/Obj-C pod) of via `onnx-coreml` naar Core ML
# MAGIC - **Android** — ONNX Runtime Mobile (AAR) met optionele NNAPI-acceleratie
# MAGIC - **Web** — ONNX Runtime Web (browser/PWA)
# MAGIC
# MAGIC Typische footprint: **< 5 MB**, **< 10 ms** inferentie per ticket, geen netwerk.

# COMMAND ----------

# MAGIC %pip install --quiet scikit-learn skl2onnx onnxruntime onnx

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json, shutil, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import f1_score
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
import onnxruntime as ort

# Gold data → pandas (support tickets = klein genoeg voor single node)
pdf_train = spark.table(GOLD_TRAIN).select("text_clean", "category_label").toPandas()
pdf_test  = spark.table(GOLD_TEST ).select("text_clean", "category_label").toPandas()

sk_pipe = SkPipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                              min_df=2, max_features=best_hash_tf.getNumFeatures())),
    ("clf",   LogisticRegression(C=1.0 / max(best_lr.getRegParam(), 1e-6),
                                 max_iter=200, solver="liblinear",
                                 multi_class="auto")),
])
sk_pipe.fit(pdf_train["text_clean"].astype(str), pdf_train["category_label"])

sk_test_f1 = f1_score(pdf_test["category_label"],
                      sk_pipe.predict(pdf_test["text_clean"].astype(str)),
                      average="macro")
print(f"sklearn twin macro-F1 (test) = {sk_test_f1:.3f}")

# COMMAND ----------

# --- Export naar ONNX ---
onnx_model = to_onnx(
    sk_pipe,
    initial_types=[("text_clean", StringTensorType([None, 1]))],
    target_opset=15,
    options={id(sk_pipe.named_steps["clf"]): {"zipmap": False}},   # rauwe probs, mobile-vriendelijk
)

local_onnx = "/tmp/flowsure_edge_classifier.onnx"
with open(local_onnx, "wb") as f:
    f.write(onnx_model.SerializeToString())

# Labels-mapping meesturen — telefoon-app doet argmax → label
labels_json = "/tmp/flowsure_edge_labels.json"
with open(labels_json, "w") as f:
    json.dump({"labels": list(sk_pipe.named_steps["clf"].classes_)}, f)

# --- Sanity check: ONNX-runtime inferentie moet identiek zijn aan sklearn ---
sess = ort.InferenceSession(local_onnx, providers=["CPUExecutionProvider"])
sample = pdf_test["text_clean"].astype(str).head(50).to_numpy().reshape(-1, 1)
onnx_pred_idx = sess.run(None, {"text_clean": sample})[0].argmax(axis=1)
onnx_pred = np.array(sk_pipe.named_steps["clf"].classes_)[onnx_pred_idx]
sk_pred   = sk_pipe.predict(sample.ravel())
parity    = float((onnx_pred == sk_pred).mean())
print(f"ONNX ↔ sklearn parity op 50 samples = {parity:.2%}")
assert parity == 1.0, "ONNX-export wijkt af van sklearn — export niet uploaden."

# COMMAND ----------

# --- INT8 dynamic quantization → ~4× kleiner, ~2× sneller op ARM CPU's ---
from onnxruntime.quantization import quantize_dynamic, QuantType

local_onnx_int8 = "/tmp/flowsure_edge_classifier.int8.onnx"
quantize_dynamic(local_onnx, local_onnx_int8, weight_type=QuantType.QInt8)

# Parity-check ook op de gequantiseerde variant (mag 1 label afwijken op 50)
sess_q = ort.InferenceSession(local_onnx_int8, providers=["CPUExecutionProvider"])
q_pred_idx = sess_q.run(None, {"text_clean": sample})[0].argmax(axis=1)
q_pred = np.array(sk_pipe.named_steps["clf"].classes_)[q_pred_idx]
parity_int8 = float((q_pred == sk_pred).mean())
print(f"INT8 ONNX ↔ sklearn parity op 50 samples = {parity_int8:.2%}")
assert parity_int8 >= 0.95, "INT8 quantisatie degradeert te veel — niet uploaden."

# COMMAND ----------

# --- Persist naar UC Volume + MLflow artifact + tag op de UC model-versie ---
import os, json as _json

shutil.copy(local_onnx,      f"{MOBILE_PATH}/edge_classifier.onnx")
shutil.copy(local_onnx_int8, f"{MOBILE_PATH}/edge_classifier.int8.onnx")
shutil.copy(labels_json,     f"{MOBILE_PATH}/edge_labels.json")

# Manifest — telefoon-app leest dit om te weten welke versie geladen is
manifest = {
    "model_name": EDGE_MODEL_NAME,
    "model_version": new_ver,
    "run_id": run_id,
    "input_name": "text_clean",
    "input_shape": [None, 1],
    "input_dtype": "string",
    "output_name": "probabilities",
    "labels_file": "edge_labels.json",
    "fp32_file": "edge_classifier.onnx",
    "int8_file": "edge_classifier.int8.onnx",
    "opset": 15,
    "sklearn_twin_test_f1": sk_test_f1,
    "parity_fp32": parity,
    "parity_int8": parity_int8,
}
manifest_local = "/tmp/flowsure_edge_manifest.json"
with open(manifest_local, "w") as f:
    _json.dump(manifest, f, indent=2)
shutil.copy(manifest_local, f"{MOBILE_PATH}/manifest.json")

size_fp32 = os.path.getsize(local_onnx)      / 1024
size_int8 = os.path.getsize(local_onnx_int8) / 1024

with mlflow.start_run(run_id=run_id):
    mlflow.log_artifact(local_onnx,      "mobile")
    mlflow.log_artifact(local_onnx_int8, "mobile")
    mlflow.log_artifact(labels_json,     "mobile")
    mlflow.log_artifact(manifest_local,  "mobile")
    mlflow.log_metric("sklearn_twin_test_f1", sk_test_f1)
    mlflow.log_metric("onnx_sklearn_parity",       parity)
    mlflow.log_metric("onnx_int8_sklearn_parity",  parity_int8)
    mlflow.log_metric("onnx_fp32_size_kb", size_fp32)
    mlflow.log_metric("onnx_int8_size_kb", size_int8)

client.set_model_version_tag(EDGE_MODEL_NAME, new_ver, "onnx_fp32_size_kb", f"{size_fp32:.1f}")
client.set_model_version_tag(EDGE_MODEL_NAME, new_ver, "onnx_int8_size_kb", f"{size_int8:.1f}")
client.set_model_version_tag(EDGE_MODEL_NAME, new_ver, "onnx_path", f"{MOBILE_PATH}/")

print(f"✅ ONNX fp32 ({size_fp32:.1f} KB) → {MOBILE_PATH}/edge_classifier.onnx")
print(f"✅ ONNX int8 ({size_int8:.1f} KB) → {MOBILE_PATH}/edge_classifier.int8.onnx")
print(f"✅ Labels                        → {MOBILE_PATH}/edge_labels.json")
print(f"✅ Manifest                      → {MOBILE_PATH}/manifest.json")
