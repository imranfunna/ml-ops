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
mlflow.pyspark.ml.autolog(log_models=False)   # models loggen we handmatig i.v.m. signature

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
