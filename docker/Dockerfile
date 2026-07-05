# ---------- FlowSure Edge Model — portable ONNX inference container ----------
# Same model.onnx runs here (Docker) and on Android/iOS (ONNX Runtime Mobile).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/models

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Model artifacts are mounted at runtime from the UC Volume (see README).
# For a fully-baked image, `COPY models/ /models/` here instead.
VOLUME ["/models"]

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
