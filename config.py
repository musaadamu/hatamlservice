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

    # Model Configuration
    MODEL_PATH: str = "../phdhatamodel"
    MODEL_NAME: str = "msmaje/phdhatamodel"
    MODEL_SOURCE: str = "local"  # "local" or "hub"
    MAX_SEQUENCE_LENGTH: int = 512
    BATCH_SIZE: int = 16
    
    # Device Configuration
    DEVICE: str = "cuda"  # or "cpu"
    
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
    
    # HF Hub
    HF_TOKEN: Optional[str] = None
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"


# Create settings instance
settings = Settings()

