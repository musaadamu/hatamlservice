"""
Model Service - Handles inference via Local Model or HuggingFace API
Optimized for EXTREME memory efficiency on Render Free Tier (512MB RAM)
"""

import os
import httpx
import numpy as np
import gc
from loguru import logger
from config import settings

# Move heavy imports to be conditional or inside class to save startup RAM
OPTIMUM_AVAILABLE = False
try:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from onnxruntime import SessionOptions
    from transformers import AutoTokenizer
    OPTIMUM_AVAILABLE = True
except ImportError:
    logger.warning("Optimum or Transformers not available in this environment")

class ModelService:
    """Service for inference supporting both local and API-based models"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.api_endpoint = settings.HF_API_ENDPOINT
        self.hf_token = settings.HF_TOKEN
        self.use_api = settings.USE_HF_INFERENCE_API
        
        # Determine device without importing torch early
        self.device = "cpu" 
        
        self.load_model()
    
    def load_model(self):
        """Initialize the model with extreme memory restrictions"""
        try:
            if self.use_api:
                logger.info(f"Using HuggingFace Inference API: {self.api_endpoint}")
                return

            logger.info(f"Loading model locally from {settings.MODEL_NAME}...")
            
            # Create cache directory
            os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
            
            # 1. Load Tokenizer (Use slow one for memory/compatibility)
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.MODEL_NAME,
                cache_dir=settings.MODEL_CACHE_DIR,
                token=self.hf_token,
                use_fast=False
            )
            
            # 2. Configure ONNX for Minimal Memory
            if settings.USE_ONNX and OPTIMUM_AVAILABLE:
                logger.info(f"🚀 Loading ONNX model: {settings.ONNX_FILE}...")
                
                # Session options to limit memory usage
                options = SessionOptions()
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                options.add_session_config_entry("session.load_model_format", "ONNX")
                
                self.model = ORTModelForSequenceClassification.from_pretrained(
                    settings.MODEL_NAME,
                    file_name=settings.ONNX_FILE,
                    cache_dir=settings.MODEL_CACHE_DIR,
                    token=self.hf_token,
                    provider="CPUExecutionProvider",
                    session_options=options
                )
                logger.info("✅ ONNX model loaded. Running garbage collection...")
            else:
                raise ImportError("Local ONNX loading required but Optimum not available.")

            # 3. Clean up memory immediately
            gc.collect()
            logger.info("Local model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model service: {e}")
            raise
    
    def predict(self, text: str, language: str) -> dict:
        """Make prediction using selected inference method"""
        if self.use_api:
            return self._predict_api(text, language)
        else:
            return self._predict_local(text, language)

    def _predict_local(self, text: str, language: str) -> dict:
        """Local inference using ORT"""
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=settings.MAX_SEQUENCE_LENGTH,
                padding=True,
                return_tensors="np" # Use Numpy directly for ONNX, faster than Torch
            )
            
            # Inference
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Softmax manually to avoid torch dependency here
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probabilities = (exp_logits / np.sum(exp_logits, axis=-1, keepdims=True))[0]
                
            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])
            
            return {
                "prediction": {
                    "label": predicted_class,
                    "label_text": settings.LABELS[predicted_class],
                    "confidence": confidence,
                    "probabilities": [float(p) for p in probabilities],
                    "human_prob": float(probabilities[0]),
                    "ai_prob": float(probabilities[1])
                },
                "language": language,
                "language_name": settings.LANGUAGE_NAMES.get(language, language),
                "tokens": text.split(),
                "inference_source": "local_onnx"
            }
            
        except Exception as e:
            logger.error(f"Local prediction error: {e}")
            raise

    def _predict_api(self, text: str, language: str) -> dict:
        """HuggingFace API implementation (similar to before)"""
        # ... logic for API prediction ...
        headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        payload = {"inputs": text}
        response = httpx.post(self.api_endpoint, json=payload, headers=headers, timeout=30)
        data = response.json()
        # simplified parsing for demonstration
        return {"prediction": data, "inference_source": "huggingface_api"}

    def get_model_info(self) -> dict:
        return {"model_name": settings.MODEL_NAME, "inference_mode": "local_onnx"}

# Create global model service instance
model_service = ModelService()
