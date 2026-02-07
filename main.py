"""
HATA ML Service - FastAPI Application
Main entry point for the ML microservice
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
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
logger.add(sys.stderr, level=settings.LOG_LEVEL)
logger.add("logs/ml_service.log", rotation="500 MB", level=settings.LOG_LEVEL)

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
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_service.model is not None,
        "device": str(model_service.device),
        "supported_languages": settings.SUPPORTED_LANGUAGES
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS
    )

