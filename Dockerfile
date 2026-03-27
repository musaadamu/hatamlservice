# ============================================
# HATA ML Service - AWS Lambda Container Image
# Optimized for local model inference with PyTorch
# ============================================
# Base image: AWS Lambda Python 3.11 runtime
FROM public.ecr.aws/lambda/python:3.11

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install dependencies (no cache to reduce image size)
# PyTorch and Transformers will be installed here
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py .
COPY main.py .
COPY app.py .
COPY services/ ./services/

# Copy .env file for default config (Lambda env vars will override)
COPY .env.lambda .env

# Create /tmp directories for Lambda (only /tmp is writable)
# /tmp/logs - for application logs
# /tmp/model_cache - for HuggingFace model cache
RUN mkdir -p /tmp/logs /tmp/model_cache

# Set HuggingFace cache directory to /tmp (Lambda writable directory)
ENV HF_HOME=/tmp/model_cache
ENV TRANSFORMERS_CACHE=/tmp/model_cache

# Lambda handler entry point
# Mangum wraps FastAPI's ASGI app for Lambda's event-driven invocation
CMD ["main.handler"]
