"""
ML Service Configuration
"""

import os
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import List, Any, Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_TITLE: str = "HATA ML Service"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Human-AI Text Attribution ML Service for African Languages"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    WORKERS: int = 1
    RELOAD: bool = True
    
    # CORS - Allow backend and frontend to access ML service
    CORS_ORIGINS: Any = [
        "http://localhost:3000",
        "http://localhost:3001",
        "https://hatafrontend.vercel.app",
        "https://hatafrontend-1fv4sh97i-musa-adamus-projects.vercel.app",
        "https://hatabackend.onrender.com"
    ]
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        return [str(v)]

    # Model Configuration - Using HuggingFace Inference API for Render Free Tier
    MODEL_NAME: str = "msmaje/Quantizedphdhatamodel"
    HF_API_ENDPOINT: str = "https://api-inference.huggingface.co/models/msmaje/Quantizedphdhatamodel"
    USE_HF_INFERENCE_API: bool = True  # MUST be True for Render free tier (512MB RAM limit)
    MODEL_CACHE_DIR: str = "./model_cache"
    USE_DYNAMIC_QUANTIZATION: bool = False  # Not needed for API mode
    MAX_SEQUENCE_LENGTH: int = 512
    BATCH_SIZE: int = 8
    USE_HALF_PRECISION: bool = False

    # ONNX Configuration (not used in API mode)
    ONNX_FILE: str = "model_quantized.onnx"
    USE_ONNX: bool = False
    
    # Supported Languages
    SUPPORTED_LANGUAGES: List[str] = ["ha", "yo", "ig", "pcm"]
    
    # Language Names
    LANGUAGE_NAMES: dict = {
        "ha": "Hausa",
        "yo": "Yoruba",
        "ig": "Igbo",
        "pcm": "Nigerian Pidgin"
    }
    
    # Labels
    LABELS: dict = {
        0: "Human-written",
        1: "AI-generated"
    }
    
    # Explainability - Reduced for faster processing on CPU
    LIME_NUM_SAMPLES: int = 100  # Reduced from 1000 to 100 for 10x speed improvement
    LIME_NUM_FEATURES: int = 10
    
    # HF Hub - API Token for HuggingFace Inference API
    HF_TOKEN: Optional[str] = None
    
    # Inference API Timeout
    INFERENCE_API_TIMEOUT: int = 30  # seconds
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"


# Create settings instance
settings = Settings()

