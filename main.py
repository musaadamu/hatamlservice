"""
HATA ML Service - FastAPI Application
Main entry point for the ML microservice
Supports both local (uvicorn) and AWS Lambda (Mangum) deployment
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from mangum import Mangum
from loguru import logger
import sys
import time
import numpy as np

from config import settings
from services.model_service import model_service
from services.explainability_service import ExplainabilityService
from services.bias_service import BiasDetectionService

# Configure logger
logger.remove()
logger.add(sys.stdout, level=settings.LOG_LEVEL)

# Only add file logging when NOT running on Lambda (Lambda has read-only filesystem except /tmp)
IS_LAMBDA = bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
if not IS_LAMBDA:
    logger.add("logs/ml_service.log", rotation="500 MB", level=settings.LOG_LEVEL)
else:
    logger.add("/tmp/ml_service.log", rotation="10 MB", level=settings.LOG_LEVEL)
    logger.info("Running on AWS Lambda — file logs go to /tmp/")

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
explainability_service = ExplainabilityService(model_service)
bias_service = BiasDetectionService()


# Request/Response Models
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    language: str = Field(..., pattern="^(ha|yo|ig|pcm)$")
    include_explanation: bool = True
    include_bias: bool = True


class PredictionResponse(BaseModel):
    prediction: dict
    explanation: Optional[dict] = None
    biasScore: Optional[dict] = None
    language: str
    language_name: str
    processing_time: float


class BatchPredictionRequest(BaseModel):
    texts: List[str]
    languages: List[str]
    include_explanation: bool = False
    include_bias: bool = False


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(i) for i in obj]
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.str_, str)):
        return str(obj)
    elif isinstance(obj, tuple):
        return [sanitize(i) for i in obj]
    return obj


# Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running",
        "runtime": "aws_lambda" if IS_LAMBDA else "local"
    }


@app.get("/ping")
async def ping():
    """Keep-warm endpoint for Lambda scheduled events (CloudWatch).
    Call this every 5 minutes to avoid cold starts."""
    return {"status": "warm", "timestamp": time.time()}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "inference_type": "huggingface_api" if settings.USE_HF_INFERENCE_API else "local",
        "api_endpoint": settings.HF_API_ENDPOINT,
        "supported_languages": settings.SUPPORTED_LANGUAGES,
        "model_name": settings.MODEL_NAME,
        "runtime": "aws_lambda" if IS_LAMBDA else "local"
    }


@app.get("/model-info")
async def get_model_info():
    """Get model information"""
    return model_service.get_model_info()


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction on input text
    
    Args:
        request: PredictionRequest with text and language
        
    Returns:
        PredictionResponse with prediction, explanation, and bias scores
    """
    try:
        start_time = time.time()
        
        # Validate language
        if request.language not in settings.SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language: {request.language}"
            )
        
        # Make prediction
        prediction_result = model_service.predict(request.text, request.language)
        
        # Generate explanation if requested
        explanation = None
        if request.include_explanation:
            explanation = explainability_service.explain_prediction(
                request.text,
                request.language
            )
        
        # Detect bias if requested
        bias_score = None
        if request.include_bias:
            bias_score = bias_service.detect_bias(request.text, request.language)
        
        processing_time = time.time() - start_time

        response = {
            "prediction": sanitize(prediction_result["prediction"]),
            "explanation": sanitize(explanation),
            "biasScore": sanitize(bias_score),
            "language": request.language,
            "language_name": settings.LANGUAGE_NAMES[request.language],
            "processing_time": round(processing_time, 4)
        }

        logger.info(f"Prediction completed in {processing_time:.4f}s")
        logger.info(f"Returning sanitized response: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict")
async def batch_predict(request: BatchPredictionRequest):
    """
    Make predictions on multiple texts
    
    Args:
        request: BatchPredictionRequest with texts and languages
        
    Returns:
        List of prediction results
    """
    try:
        if len(request.texts) != len(request.languages):
            raise HTTPException(
                status_code=400,
                detail="Number of texts must match number of languages"
            )
        
        results = model_service.batch_predict(request.texts, request.languages)
        return {"results": sanitize(results), "count": len(results)}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# AWS Lambda handler — Mangum adapts FastAPI's ASGI to Lambda's event interface
handler = Mangum(app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS
    )

