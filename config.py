"""
ML Service Configuration
Supports both HuggingFace Inference API (Render) and Local Model (development/Lambda)
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

    # -------------------------------------------------------
    # Model Configuration
    # -------------------------------------------------------
    # USE_HF_INFERENCE_API:
    #   true  → Call HuggingFace Inference API remotely (ideal for Render free tier)
    #   false → Load model locally with PyTorch (ideal for local dev / Lambda / GPU servers)
    # -------------------------------------------------------
    MODEL_NAME: str = "msmaje/phdhatamodel"
    USE_HF_INFERENCE_API: bool = True
    HF_API_ENDPOINT: str = "https://api-inference.huggingface.co/models/msmaje/phdhatamodel"
    INFERENCE_API_TIMEOUT: int = 30
    
    # Local model settings (used when USE_HF_INFERENCE_API=false)
    MODEL_CACHE_DIR: str = "./model_cache"
    LOCAL_MODEL_PATH: str = ""  # Path to local model dir (e.g. ../phdhatamodel)
    USE_DYNAMIC_QUANTIZATION: bool = False
    MAX_SEQUENCE_LENGTH: int = 512
    BATCH_SIZE: int = 8
    USE_HALF_PRECISION: bool = False

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
    
    # Explainability
    LIME_NUM_SAMPLES: int = 100
    LIME_NUM_FEATURES: int = 10

    # HF Hub Token — required for HF Inference API and downloading private models
    HF_TOKEN: Optional[str] = None
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"


# Create settings instance
settings = Settings()
