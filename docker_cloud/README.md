# FlowSure Cloud Responder — AWS Lambda Container

This directory contains the Dockerized **Cloud Model** (SentenceTransformers + PyTorch), optimized for deployment as a Serverless Container on AWS Lambda. It exposes a FastAPI endpoint via Mangum.

## 1. Running Locally (For Testing)

To test the model on your laptop without pushing to AWS, follow these steps:

1. **Pull the model from Databricks:**
   First, you need the MLflow model downloaded into the `models/` folder.
   ```bash
   mkdir models
   databricks fs cp -r dbfs:/Volumes/flowsure/mlops/artifacts/models/cloud_responder models/
   ```
   *(Note: The exact path might differ depending on your `_common.py` variables).*

2. **Build the Docker Image locally:**
   ```bash
   docker build -t flowsure-cloud-api .
   ```

3. **Run the container locally:**
   Because we are using the AWS Lambda base image, we have to run it locally using the AWS Lambda Runtime Emulator, OR we can override the entrypoint to run `uvicorn` directly for local testing:
   ```bash
   docker run -d -p 8000:8000 --entrypoint python flowsure-cloud-api -m uvicorn app:app --host 0.0.0.0 --port 8000
   ```

4. **Test the API via cURL:**
   ```bash
   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{"text": "How do I reset my password?"}'
   ```

## 2. Using the Model in the Cloud (AWS Lambda)

Once this container is deployed to AWS Lambda and connected to an **API Gateway**, you will receive a public HTTPS URL from AWS (e.g., `https://abc123xyz.execute-api.us-east-1.amazonaws.com/predict`).

You can integrate this directly into your frontend or customer service software (like Zendesk, Intercom, or a custom portal):

```javascript
fetch("https://<YOUR_API_GATEWAY_URL>/predict", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    text: "Can I get a refund for my last order?"
  })
})
.then(response => response.json())
.then(data => {
  console.log("Suggested response:", data.suggested_response);
  console.log("Intent matched:", data.matched_intent);
  console.log("Confidence:", data.confidence);
});
```

Because it runs on AWS Lambda, it costs nothing when idle, and gives you up to 1 million free requests per month.
