"""
ML Service Configuration
"""

import os
from pydantic_settings import BaseSettings
from typing import List


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
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001,https://hatafrontend.vercel.app").split(",")
    
    # Model Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "../phdhatamodel")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "msmaje/phdhatamodel")
    MODEL_SOURCE: str = os.getenv("MODEL_SOURCE", "local")  # "local" or "hub"
    MAX_SEQUENCE_LENGTH: int = 512
    BATCH_SIZE: int = 16
    
    # Device Configuration
    DEVICE: str = os.getenv("DEVICE", "cuda")  # or "cpu"
    
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
    LIME_NUM_SAMPLES: int = int(os.getenv("LIME_NUM_SAMPLES", "1000"))
    LIME_NUM_FEATURES: int = int(os.getenv("LIME_NUM_FEATURES", "10"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"


# Create settings instance
settings = Settings()

