"""FastAPI inference server for the portable FlowSure edge model.

Loads `${MODEL_DIR}/model.onnx` + `${MODEL_DIR}/labels.json` once at startup
and serves `/predict` on port 8080. Same ONNX artifact is used on-device.
"""
from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
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
_labels_data = json.loads(LABELS_PATH.read_text())
_labels: List[str] = _labels_data.get("labels", []) if isinstance(_labels_data, dict) else _labels_data
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


# --- BESTAANDE ENDPOINTS (INTACT) ---

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
                category=str(p),
                confidence=float(np.max(probs[i])),
                latency_ms=dt,
            )
            for i, p in enumerate(preds)
        ])
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# --- NIEUWE UI ENDPOINTS (TABEL / CSV SUPPORT) ---

@app.get("/", response_class=HTMLResponse)
def home_ui() -> str:
    """Renders a simple, responsive HTML dashboard for mobile and desktop."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FlowSure Edge UI</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 20px; max-width: 600px; margin: 0 auto; background-color: #f4f6f8; }
            .card { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-top: 40px; }
            h2 { color: #111; margin-top: 0; }
            p { color: #555; font-size: 14px; line-height: 1.5; }
            label { display: block; margin-top: 15px; font-weight: bold; font-size: 14px; color: #333; }
            input[type="file"] { display: block; margin: 10px 0 20px 0; padding: 15px; border: 2px dashed #0070f3; width: 100%; border-radius: 8px; box-sizing: border-box; background: #f0f7ff; cursor: pointer; }
            input[type="text"] { display: block; margin: 10px 0 25px 0; padding: 12px; border: 1px solid #ccc; width: 100%; border-radius: 6px; box-sizing: border-box; font-size: 14px; }
            button { background-color: #0070f3; color: white; border: none; padding: 14px 20px; border-radius: 6px; font-size: 16px; font-weight: bold; cursor: pointer; width: 100%; transition: background 0.2s; }
            button:hover { background-color: #0051cb; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>FlowSure Edge Portal</h2>
            <p>Upload een CSV-bestand om bulk-voorspellingen uit te voeren via het lokale ONNX-model op je laptop.</p>
            <form action="/predict-csv" enctype="multipart/form-data" method="post">
                <label>Kies CSV-bestand:</label>
                <input name="file" type="file" accept=".csv" required>
                
                <label>Kolomnaam met tekst (optioneel):</label>
                <input name="column_name" type="text" placeholder="Laat leeg om eerste kolom te gebruiken">
                
                <button type="submit">Start Voorspelling</button>
            </form>
        </div>
    </body>
    </html>
    """


@app.post("/predict-csv", response_class=HTMLResponse)
async def predict_csv(file: UploadFile = File(...), column_name: str = Form(None)) -> str:
    """Parses the uploaded CSV, runs batch ONNX inference, and returns an HTML table."""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if df.empty:
            return "<h3>Fout: Het geüploade CSV-bestand bevat geen data.</h3><a href='/'>Terug</a>"
        
        # Bepaal de tekstkolom (gegeven naam -> fallback op 'text' -> fallback op eerste kolom)
        if column_name and column_name in df.columns:
            text_col = column_name
        elif "text" in df.columns:
            text_col = "text"
        else:
            text_col = df.columns[0]
            
        texts = df[text_col].astype(str).tolist()
        
        # Voer de ONNX-batch-inference uit
        arr = np.array([[t] for t in texts], dtype=object)
        preds, probs = _session.run(None, {_input_name: arr})
        
        # Voeg de voorspellingen toe aan de resultaten-tabel
        df["Voorspelde Categorie"] = [str(p) for p in preds]
        df["Betrouwbaarheid (%)"] = [round(float(np.max(probs[i])) * 100, 1) for i in range(len(probs))]
        
        # Toon de belangrijkste kolommen (gelimiteerd tot eerste 100 rijen voor snelle weergave op mobiel)
        preview_df = df[[text_col, "Voorspelde Categorie", "Betrouwbaarheid (%)"]].head(100)
        table_html = preview_df.to_html(classes="result-table", index=False)
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FlowSure Resultaten</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 15px; background-color: #f4f6f8; }}
                .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }}
                h2 {{ color: #111; margin-top: 10px; }}
                .result-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; }}
                .result-table th, .result-table td {{ border: 1px solid #e1e4e8; padding: 12px; text-align: left; }}
                .result-table th {{ background-color: #0070f3; color: white; position: sticky; top: 0; }}
                .result-table tr:nth-child(even) {{ background-color: #f8f9fa; }}
                .back-btn {{ display: inline-block; background-color: #333; color: white; text-decoration: none; padding: 10px 18px; border-radius: 6px; font-weight: bold; margin-bottom: 15px; font-size: 14px; }}
                .back-btn:hover {{ background-color: #555; }}
                .info {{ color: #666; font-size: 13px; margin-top: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <a href="/" class="back-btn">&larr; Terug</a>
                <h2>Voorspellingsresultaten</h2>
                <div class="info">Toont de eerste {len(preview_df)} van de {len(df)} rijen uit <strong>{file.filename}</strong>.</div>
                <div style="overflow-x: auto;">
                    {table_html}
                </div>
            </div>
        </body>
        </html>
        """
    except Exception as exc:
        return f"<h3>Er is een fout opgetreden tijdens het verwerken van de CSV:</h3><p>{str(exc)}</p><a href='/'>Terug</a>"