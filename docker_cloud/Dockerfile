# Use the official AWS Lambda Python 3.12 base image
FROM public.ecr.aws/lambda/python:3.12

# Set the HuggingFace cache directory to a writable /tmp directory
# (AWS Lambda only allows writing to /tmp)
ENV HF_HOME="/tmp/huggingface"
ENV MODEL_DIR="/var/task/models"
ENV MLFLOW_TRACKING_URI="file:///tmp/mlruns"
ENV HOME="/tmp"

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app and Mangum handler
COPY app.py ${LAMBDA_TASK_ROOT}

# The model artifacts will be copied into the image by the CI/CD pipeline
# or when running locally.
COPY models/ ${LAMBDA_TASK_ROOT}/models/

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]
