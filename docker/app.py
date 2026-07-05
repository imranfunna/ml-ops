"""FastAPI inference server for the portable FlowSure edge model.

Loads `${MODEL_DIR}/model.onnx` + `${MODEL_DIR}/labels.json` once at startup
and serves `/predict` on port 8080. Same ONNX artifact is used on-device.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models"))
ONNX_PATH = MODEL_DIR / "model.onnx"
LABELS_PATH = MODEL_DIR / "labels.json"

if not ONNX_PATH.exists() or not LABELS_PATH.exists():
    raise RuntimeError(
        f"Missing model artifacts in {MODEL_DIR}. "
        "Mount the UC volume path /Volumes/flowsure/mlops/artifacts/mobile there."
    )

_session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
_labels: List[str] = json.loads(LABELS_PATH.read_text())
_input_name = _session.get_inputs()[0].name


class PredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=256)


class Prediction(BaseModel):
    category: str
    confidence: float
    latency_ms: float


class PredictResponse(BaseModel):
    predictions: List[Prediction]


app = FastAPI(title="FlowSure Edge Classifier", version="1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "labels": len(_labels), "model": ONNX_PATH.name}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        arr = np.array([[t] for t in req.texts], dtype=object)
        t0 = time.perf_counter()
        preds, probs = _session.run(None, {_input_name: arr})
        dt = (time.perf_counter() - t0) * 1000.0 / len(req.texts)
        return PredictResponse(predictions=[
            Prediction(
                category=_labels[int(p)],
                confidence=float(np.max(probs[i])),
                latency_ms=dt,
            )
            for i, p in enumerate(preds)
        ])
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
