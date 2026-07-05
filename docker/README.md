# FlowSure Edge — Portable ONNX Container

Serveert `model.onnx` via HTTP op poort 8080. Zelfde artefact als dat on-device
op Android/iOS draait — één training, drie deployment-doelen.

## Waar staan de model-bestanden?

Notebook `02b_train_edge_portable` schrijft ze naar de UC Volume:

```
/Volumes/flowsure/mlops/artifacts/mobile/
├── model.onnx          (~1-3 MB)
└── labels.json
```

## Optie A — Pull-and-run (dev/test)

```bash
# 1. Haal artefacten op via Databricks CLI (v0.240+)
databricks fs cp -r /Volumes/flowsure/mlops/artifacts/mobile ./models

# 2. Build & run
docker build -t flowsure-edge:latest .
docker run --rm -p 8080:8080 -v "$(pwd)/models:/models:ro" flowsure-edge:latest
```

## Optie B — Fully-baked image (CI/CD, aanbevolen voor prod)

De CI job `docker_publish` (zie `.github/workflows/ci-cd.yml`) doet dit
automatisch bij elke tag-release:

1. Pullt `model.onnx` uit UC via `databricks fs cp`
2. Bakt ze in via `COPY models/ /models/` (zie `Dockerfile.baked`)
3. Pusht naar `ghcr.io/<org>/flowsure-edge:<git-tag>`

```bash
docker pull ghcr.io/<org>/flowsure-edge:v1.0.0
docker run --rm -p 8080:8080 ghcr.io/<org>/flowsure-edge:v1.0.0
```

## Testen

```bash
curl -sS localhost:8080/predict \
  -H 'content-type: application/json' \
  -d '{"texts":["cannot login","invoice is wrong"]}' | jq
```

```json
{
  "predictions": [
    {"category":"login_issue","confidence":0.87,"latency_ms":0.9},
    {"category":"billing",    "confidence":0.91,"latency_ms":0.9}
  ]
}
```

## Vanaf je telefoon testen (dev)

Draai de container op je laptop, zoek je LAN-IP (`ipconfig` / `ifconfig`):

```
http://192.168.x.y:8080/predict
```

Zorg dat laptop en telefoon op hetzelfde wifi zitten en dat je firewall
poort 8080 openzet. Voor productie draai je hem in een cloud runtime
(Azure Container Apps, GCP Cloud Run, AWS Fargate) achter TLS.
