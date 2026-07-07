import os
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mangum import Mangum

app = FastAPI(title="FlowSure Cloud Responder API")

# Define request/response schemas
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    suggested_response: str
    matched_intent: str
    confidence: float

# Globals for the loaded model
_model = None
MODEL_DIR = os.getenv("MODEL_DIR", "/var/task/models")

def load_model():
    global _model
    if _model is None:
        try:
            print(f"Loading MLflow model from {MODEL_DIR}...")
            # Load the PyFunc model from the local directory
            _model = mlflow.pyfunc.load_model(MODEL_DIR)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

@app.on_event("startup")
def startup_event():
    """Load the model during fastAPI startup when running locally."""
    # AWS Lambda will likely initialize the model lazily on the first request 
    # to avoid timeout during the init phase, but local uvicorn can load it on startup.
    if not os.getenv("LAMBDA_TASK_ROOT"):
        load_model()

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Cloud Responder API is running"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Ensure model is loaded (crucial for AWS Lambda cold starts)
    if _model is None:
        load_model()

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # MLflow pyfunc expects a pandas DataFrame
    input_df = pd.DataFrame([{"text_clean": req.text}])
    
    # Run inference
    try:
        predictions = _model.predict(input_df)
        
        # Extract the first row of predictions
        result = predictions.iloc[0]
        
        return PredictResponse(
            suggested_response=str(result["suggested_response"]),
            matched_intent=str(result["matched_intent"]),
            confidence=float(result["confidence"])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# Mangum wrapper required for AWS Lambda
# This takes API Gateway events and translates them to FastAPI HTTP requests
handler = Mangum(app)

if __name__ == "__main__":
    # For local testing
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
