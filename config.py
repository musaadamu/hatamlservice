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

    # Model Configuration - AWS Lambda Local Model Loading
    MODEL_NAME: str = "msmaje/Quantizedphdhatamodel"
    USE_HF_INFERENCE_API: bool = False  # Load model locally on AWS Lambda (3008MB RAM available)
    MODEL_CACHE_DIR: str = "/tmp/model_cache"  # AWS Lambda writable directory
    USE_DYNAMIC_QUANTIZATION: bool = True  # Use INT8 quantization for memory efficiency
    MAX_SEQUENCE_LENGTH: int = 512
    BATCH_SIZE: int = 8
    USE_HALF_PRECISION: bool = False  # Quantized model already optimized

    # ONNX Configuration (optional optimization)
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
    
    # Explainability - Optimized for AWS Lambda
    LIME_NUM_SAMPLES: int = 100  # Reduced from 1000 to 100 for 24x speed improvement
    LIME_NUM_FEATURES: int = 10

    # HF Hub - Token for downloading model from HuggingFace Hub
    HF_TOKEN: Optional[str] = None  # Required for downloading private/gated models
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"


# Create settings instance
settings = Settings()

