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
    
    # CORS
    CORS_ORIGINS: Any = ["http://localhost:3000", "http://localhost:3001", "https://hatafrontend.vercel.app"]
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        return [str(v)]

    # Model Configuration - Local Inference (Download from HuggingFace)
    MODEL_NAME: str = "msmaje/Quantized5000mbmodelhataphd"
    HF_API_ENDPOINT: str = "https://api-inference.huggingface.co/models/msmaje/Quantized5000mbmodelhataphd"
    USE_HF_INFERENCE_API: bool = False  # Set to False to load model locally on Render
    MODEL_CACHE_DIR: str = "./model_cache"
    MAX_SEQUENCE_LENGTH: int = 512
    BATCH_SIZE: int = 8  # Reduced for local inference memory limits
    USE_HALF_PRECISION: bool = False  # Set to True for FP16 inference (if supported)
    
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
    LIME_NUM_SAMPLES: int = 1000
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

