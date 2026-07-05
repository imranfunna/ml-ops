# Stap-voor-stap deploy-instructies (Unity Catalog)

Deze pipeline draait volledig op **Unity Catalog** — tabellen als 3-level
namespace, files in een UC managed **Volume**, en de MLflow-registry op
`databricks-uc`. Voor de allereerste setup zijn twee handmatige acties
nodig; daarna is alles CI/CD-geautomatiseerd.

---

## 📋 Vereisten workspace

* **Unity Catalog is geactiveerd** op de workspace (`Admin console → Metastore`).
* De gebruiker/service-principal die de eerste run doet heeft **`CREATE CATALOG`**
  (of een admin heeft de catalog `flowsure` al aangemaakt en de user heeft
  `USE CATALOG` + `CREATE SCHEMA` + `CREATE VOLUME`).
* Clusters/jobs draaien in **`SINGLE_USER`** of **`USER_ISOLATION`** mode — dit
  is al zo gezet in `databricks.yml`.
* MLflow client-versie **≥ 2.9** in het cluster-runtime (Databricks Runtime
  15.4 ML voldoet).

Wil je een andere catalog-naam? Pas één regel aan in `notebooks/_common.py`:

```python
CATALOG = "flowsure"    # bv. "flowsure_dev" of "flowsure_prod"
```

---

## 🚀 Eenmalige setup (~10 min)

### 1. Clone naar een Databricks Repo
1. Ga in de workspace naar **Repos → Add Repo**.
2. Plak de HTTPS-URL van deze Git-repository.
3. Kies de branch `main`.

### 2. Sample-data uploaden naar het UC Volume

De pipeline verwacht twee CSV's op
`/Volumes/flowsure/mlops/artifacts/landing/`.

Kies één van deze drie routes:

**a) Databricks UI**
1. Sidebar → **Catalog** → open `flowsure` → `mlops` → `artifacts`.
2. Klik **Upload to this volume** en zet ze in de map `landing/`.

**b) Databricks CLI (v0.205+)**
```bash
databricks fs cp bitext_sample.csv  \
    /Volumes/flowsure/mlops/artifacts/landing/bitext_sample.csv
databricks fs cp twitter_sample.csv \
    /Volumes/flowsure/mlops/artifacts/landing/twitter_sample.csv
```

**c) Externe cloud-sync (productie)**
Maak van `artifacts` een **external volume** met een storage-credential naar
S3/ADLS/GCS. Auto Loader in `01_data_pipeline` pikt nieuwe files dan
automatisch op. Zie Databricks docs → *External volumes*.

### 3. Draai `00_setup`
Open **`notebooks/00_setup.py`** en klik **Run All**. Deze notebook:

* maakt de catalog `flowsure`, schema `mlops` en volume `artifacts` aan
  (idempotent — bestaan ze al, dan gebeurt er niets),
* maakt de sub-directories `landing/`, `checkpoints/`, `models/` in het volume,
* controleert of de twee CSV's aanwezig zijn.

---

## ⚙️ Deployment via de Asset Bundle

Rest van de deployment gaat via de CLI — geen klikwerk meer.

```bash
# 1. Install (eenmalig)
brew install databricks/tap/databricks    # of pip install databricks-cli

# 2. Auth (eenmalig)
databricks configure --host https://<workspace>.cloud.databricks.com

# 3. Validate + deploy naar dev
databricks bundle validate -t dev
databricks bundle deploy   -t dev

# 4. Volledige pipeline handmatig draaien (of wachten op schedule 03:00 UTC)
databricks bundle run flowsure_full_pipeline -t dev
```

Wat de bundle heeft aangemaakt:

* Notebooks in `/Workspace/Users/<jij>/.bundle/flowsure/dev/notebooks/`
* Job **`flowsure-full-pipeline`** met 5 taken (data → edge → cloud → deploy → monitor)
* Job **`flowsure-streaming-inference`** (gepauzeerd — activeer bij prod-go-live)
* Job **`flowsure-retrain-on-drift`** (getriggerd door monitoring bij PSI-drift)

---

## 🔐 CI/CD secrets in GitHub

Zet in **Settings → Secrets and variables → Actions**:

| Secret | Waarde |
|--------|--------|
| `DATABRICKS_HOST_DEV`   | `https://<dev-workspace>.cloud.databricks.com` |
| `DATABRICKS_TOKEN_DEV`  | PAT of OAuth-M2M token met workspace-scope |
| `DATABRICKS_HOST_PROD`  | Prod-workspace URL |
| `DATABRICKS_TOKEN_PROD` | Prod PAT (of OAuth-M2M) |
| `SQL_WAREHOUSE_ID`      | ID van een klein SQL-warehouse voor de CT-check |

En maak een **Environment `production`** aan met **manual approval**
(Settings → Environments → New environment) — dan blijft prod-deployment gated.

---

## 🧪 Lokaal testen

De pure-python utils zijn los te testen zonder Spark of UC:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest tests/ -v
ruff check src/ tests/
```

---

## 🖥️ Dashboard aanmaken

1. Ga in Databricks naar **SQL → Dashboards → Create dashboard**.
2. Voeg 3 panels toe met de queries uit `notebooks/05_monitor.py`
   (sectie *Dashboard-queries* — ze gebruiken al de UC 3-level namen).
3. Zet auto-refresh op 5 minuten.

---

## 🚨 Alerting

1. **SQL → Alerts → Create alert**.
2. Query:
   ```sql
   SELECT count(*) AS n
   FROM   flowsure.mlops.alerts
   WHERE  severity='high'
   AND    created_ts >= current_timestamp() - INTERVAL 15 MINUTES
   ```
3. Trigger: `n > 0`.
4. Destination: PagerDuty of Slack-webhook.

---

## 🔑 UC governance grants

Productie-grants (via SQL of Catalog Explorer) — voer eenmalig uit als admin:

```sql
-- Data-analisten: alleen gold + monitoring
GRANT SELECT ON SCHEMA flowsure.mlops TO `analysts`;

-- Support-agents: predicties + KB
GRANT SELECT ON TABLE flowsure.mlops.tickets_predictions TO `support_agents`;
GRANT SELECT ON TABLE flowsure.mlops.knowledge_base       TO `support_agents`;

-- MLOps SP: alles kunnen schrijven
GRANT ALL PRIVILEGES ON SCHEMA flowsure.mlops TO `flowsure-mlops-sp`;
GRANT ALL PRIVILEGES ON VOLUME flowsure.mlops.artifacts TO `flowsure-mlops-sp`;
```

---

## 🔍 Troubleshooting

| Symptoom | Oorzaak | Fix |
|----------|---------|-----|
| `00_setup` faalt op `CREATE CATALOG` | Geen `CREATE CATALOG` privilege | Laat een admin de catalog aanmaken + grant je `USE CATALOG` + `CREATE SCHEMA` |
| `00_setup` faalt op assert missing files | CSV's staan niet in het volume | Herhaal stap 2 hierboven |
| Model wordt niet gepromoveerd naar `@champion` | `test_f1 < 0.60` (`MIN_F1_FOR_PROMOTION`) | Kijk in MLflow-run, tune het grid in `02_train_edge_model.py` |
| `ModelVersionAliasNotFound` bij inference | Nog nooit een model door de quality-gate gekomen | Draai `02_train_edge_model` en `03_train_cloud_model` opnieuw, controleer test-metrics |
| Serving-endpoint faalt met permission-denied | Model Serving niet enabled / `CAN_MANAGE` mist op UC-model | Admin: `GRANT ALL PRIVILEGES ON MODEL flowsure.mlops.flowsure_edge_classifier TO <user>` |
| Streaming job maakt geen predicties | `incoming_tickets` is leeg | Verwacht in dev — voeg via `INSERT INTO flowsure.mlops.incoming_tickets ...` een testrecord toe |
| `AccessDenied` op `/Volumes/…` | User heeft geen `READ VOLUME` grant | Admin: `GRANT READ VOLUME ON VOLUME flowsure.mlops.artifacts TO <user>` |

---

## 📱 Edge model on-device (Docker + Android + iOS)

De Spark-versie van de edge-classifier kan **niet** op een telefoon — daarvoor
draait notebook **`02b_train_edge_portable.py`** een sklearn-twin (TF-IDF +
LogReg) op de gold-data en exporteert die naar **ONNX**:

```
/Volumes/flowsure/mlops/artifacts/mobile/model.onnx              # < 3 MB
/Volumes/flowsure/mlops/artifacts/mobile/labels.json             # index → categorie
```

Het model wordt óók geregistreerd in UC als
`flowsure.mlops.flowsure_edge_classifier_onnx` (alias `@champion`) — zodat je
het via MLflow kunt versioneren en promoveren.

**Runbook:**

1. Draai in Databricks één keer `notebooks/02b_train_edge_portable.py`
   (of voeg 'm toe als extra task aan `flowsure_full_pipeline` in `databricks.yml`).
2. **Server-side (Docker)** — zie `docker/README.md` voor build + run.
3. **On-device (Android / iOS / RN / Flutter)** — zie `mobile/README.md`
   voor complete Kotlin- en Swift-snippets.

**Waarom dit werkt (en de Spark-versie niet):**

| | Spark ML PipelineModel | ONNX-export |
|---|---|---|
| Runtime nodig | JVM + Spark (> 200 MB) | ONNX Runtime Mobile (~ 5 MB) |
| Cold start | seconden | < 100 ms |
| Inference | seconden | < 10 ms |
| Werkt offline | nee | **ja** |
| Privacy | ticket-tekst gaat naar cloud | ticket-tekst verlaat telefoon niet |
