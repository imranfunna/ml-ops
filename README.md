# FlowSure — AI-assisted Support MLOps Pipeline

End-to-end MLOps-oplossing voor **FlowSure**: ticket-triage, categorie-
classificatie en concept-antwoord-generatie op basis van historische
support-data. Gebouwd op **Databricks + PySpark + Delta Lake + Unity Catalog
+ MLflow** en volledig CI/CD/CT-geautomatiseerd.

> Zie [`architecture.mmd`](architecture.mmd) voor het architectuurdiagram en
> [`instructies.md`](instructies.md) voor de stap-voor-stap deploy-instructies.

---

## 🏗️ Architectuur in één oogopslag

```
                 ┌── e-mail / chat / webform / in-app / social  (multi-channel)
                 ▼
┌───────────────────────────────────────────────────────────────────────────┐
│               U N I T Y   C A T A L O G   (flowsure.mlops)                │
│  BRONZE   Delta  ⟵ Auto Loader / batch — schema-on-read, immutable         │
│  SILVER   Delta  ⟵ validatie + PII-mask + taaldetectie + cleaning          │
│  GOLD     Delta  ⟵ feature store + gestratificeerde train/val/test split   │
│  Volume   /Volumes/flowsure/mlops/artifacts  (landing · checkpoints)      │
└───────────────────────────────────────────────────────────────────────────┘
     │                                                          │
     ▼                                                          ▼
┌─────────────────────────┐                        ┌────────────────────────┐
│ EDGE model              │                        │ CLOUD model            │
│ Spark ML TF-IDF + LogReg│                        │ TF-IDF retrieval +     │
│ CrossValidator tuning   │                        │ pyfunc PythonModel     │
│ MLflow → UC Registry    │                        │ MLflow → UC Registry   │
│   @champion / @challenger│                        │  @champion / @challenger│
└──────────┬──────────────┘                        └──────────┬─────────────┘
           │                                                  │
           └──────────────┬──────────────┬────────────────────┘
                          ▼              ▼
              batch UDF   ·   Structured Streaming   ·   Model-Serving REST
                          │              │                       │
                          └──────────────┴────────────┬──────────┘
                                                     ▼
                                   ┌──────────────────────────────────┐
                                   │ tickets_predictions  (Delta)     │
                                   │ monitoring_log · drift_metrics   │
                                   │ alerts  ⟶  CT retrain trigger    │
                                   └──────────────────────────────────┘
```

---

## 📂 Repository-structuur

```
flowsure_mlops/
├── notebooks/                # Databricks source-format .py notebooks
│   ├── _common.py            #  gedeelde UC-config & helpers
│   ├── 00_setup.py           #  catalog + schema + volume aanmaken
│   ├── 01_data_pipeline.py   #  Bronze → Silver → Gold + KB + drift baseline
│   ├── 02_train_edge_model.py#  Spark-classifier + CrossValidator + UC MLflow
│   ├── 02b_train_edge_portable.py# sklearn twin → ONNX (Docker + mobile)
│   ├── 03_train_cloud_model.py# retrieval-responder + UC MLflow
│   ├── 04_deploy_and_infer.py#  UC alias-URIs → batch + streaming + REST endpoint
│   ├── 05_monitor.py         #  drift + latency + fairness + alerts
│   └── 06_orchestrator.py    #  runt alle stappen sequentieel
├── docker/                   # FastAPI + onnxruntime container voor edge-ONNX
├── mobile/                   # Android (Kotlin) + iOS (Swift) integratie-guides
├── src/pipeline_utils.py     # pure-python utils (getest zonder Spark)
├── tests/test_pipeline_utils.py  # 20 unit tests
├── .github/workflows/ci-cd.yml   # lint · test · deploy · CT-trigger
├── databricks.yml            # Databricks Asset Bundle (jobs + schedule)
├── requirements.txt          # runtime deps
├── requirements-dev.txt      # CI-only deps
├── pyproject.toml            # ruff + pytest config
├── architecture.mmd          # Mermaid architectuurdiagram
├── instructies.md            # stap-voor-stap deploy
└── README.md                 # dit bestand
```

---

## 🧠 Modellen

| Model | Type | Framework | Purpose | Waar draait het |
|-------|------|-----------|---------|-----------------|
| **Edge classifier** | Multinomial LogReg over TF-IDF | Spark ML | 11-klassen categorie-voorspelling + prioriteit | Batch / streaming Spark job + optionele Model-Serving endpoint |
| **Edge classifier (portable)** | TF-IDF + LogReg → **ONNX** (< 3 MB) | scikit-learn + skl2onnx | Zelfde taak — draaibaar in Docker én on-device | `docker/` container · Android (`onnxruntime-android`) · iOS (`onnxruntime-objc`) |
| **Cloud responder** | TF-IDF retrieval + pyfunc | scikit-learn | Suggested-response met matched-intent + confidence | Batch UDF · Structured Streaming · Model-Serving REST |

**Waarom retrieval i.p.v. een LLM?** De KB is klein (~30 intents), latency
onder de 50 ms, geen externe kosten, deterministisch en makkelijk te
governance-en. De architectuur staat vervanging door een **Databricks
Foundation-Model-endpoint** (bv. `databricks-mixtral-8x7b-instruct`) toe
zonder de rest van de pipeline te raken — alleen de `RetrievalResponder`
pyfunc-klasse hoeft de call te swappen.

---

## 🗂️ Unity Catalog namespace

Alles wat de pipeline produceert leeft binnen één UC namespace:

| Object | UC-naam |
|--------|---------|
| Catalog | `flowsure` |
| Schema  | `flowsure.mlops` |
| Volume  | `flowsure.mlops.artifacts` (landing, checkpoints, artefacten) |
| Bronze  | `flowsure.mlops.tickets_bronze` |
| Silver  | `flowsure.mlops.tickets_silver` |
| Gold    | `flowsure.mlops.tickets_gold[_train/_val/_test]` |
| KB      | `flowsure.mlops.knowledge_base` |
| Predicties | `flowsure.mlops.tickets_predictions` |
| Drift / Alerts | `flowsure.mlops.drift_metrics`, `flowsure.mlops.alerts` |
| Edge model | `flowsure.mlops.flowsure_edge_classifier` |
| Cloud model | `flowsure.mlops.flowsure_cloud_responder` |

**Model-versionering**: UC-Model-Registry gebruikt **aliases** i.p.v. stages —
`@champion` is de door de quality-gate goedgekeurde live-scorende versie,
`@challenger` een kandidaat-versie in review / A/B-test. Promoveren = de
alias-pointer verzetten; oude versies blijven bewaard voor audit & rollback.

---

## 🔁 CI/CD/CT

* **CI** (GitHub Actions): `ruff` lint + `pytest` unit tests op elke PR
* **CD dev**: bij merge naar `main` deployt de asset-bundle automatisch
  naar de dev-workspace en start een smoke-run van de hele pipeline
* **CD prod**: gedreven door **version-tags** (`v1.2.3`) mét manual approval
  (GitHub Environments)
* **CT**:
  * Elke retrain-run doorloopt de quality-gate → alleen bij `test_f1 ≥ 0.60`
    verhuist het `@champion`-alias naar de nieuwe versie
  * Drift-getriggerd: als `05_monitor` PSI > 0.25 detecteert → aparte
    `flowsure_retrain_on_drift` job wordt automatisch aangeroepen

---

## 📊 Monitoring

Dashboard-queries staan in `05_monitor` (Python-generated) en kunnen 1-op-1
in Databricks SQL worden geplakt (ze gebruiken al UC 3-level namen):

1. **Volume per uur per categorie** — trend & piek-detectie
2. **PSI-trend** — data drift
3. **Latency-percentielen** — p50 / p95 / p99
4. **Slice-metrics per taal** — fairness proxy
5. **Recente alerts** — geconsolideerde alert-lijst

Alle metrics worden gepersisteerd in `drift_metrics` en `alerts` Delta-tabellen
zodat historie query-baar blijft en lineage automatisch bijgehouden wordt
door UC.

---

## 🔒 Governance highlights

| Concern | Hoe geadresseerd |
|---------|------------------|
| **GDPR / PII** | Silver-tekst gaat door `mask_pii` (email/phone/order-ref → `<placeholder>`) vóór feature-engineering |
| **Toegangsbeheer** | UC `GRANT`s per rol (analysts read-only op gold; support-agents alleen op predictions/KB; MLOps-SP full access — zie `instructies.md`) |
| **Lineage & audit** | UC houdt lineage automatisch bij; extra `monitoring_log`-tabel logt elke stage-run |
| **Dataretentie** | Delta `VACUUM` + tabel-property `delta.deletedFileRetentionDuration` (configureerbaar per tabel) |
| **Versiebeheer** | Git + Databricks Repos; Asset Bundle voor infra-as-code; UC Model Registry met aliases voor modelversies |
| **Reproduceerbaarheid** | Deterministische seeds (`seed=42`); MLflow logt params/metrics/artifacts/environment |
