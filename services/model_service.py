"""
Model Service - Handles model loading and inference
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger
import os
from config import settings


class ModelService:
    """Service for loading and running the HATA model"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.load_model()
    
    def load_model(self):
        """Load the model from local directory or HuggingFace Hub"""
        try:
            # Determine source
            model_to_load = settings.MODEL_PATH
            
            # Check if we should use Hub or if local path is missing
            if settings.MODEL_SOURCE == "hub" or not os.path.exists(settings.MODEL_PATH):
                logger.info(f"Using HuggingFace Hub: {settings.MODEL_NAME}")
                model_to_load = settings.MODEL_NAME
            else:
                logger.info(f"Loading local model from {settings.MODEL_PATH}")
            
            # Determine device
            if settings.DEVICE == "cuda" and torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_to_load,
                low_cpu_mem_usage=True
            )
            logger.info("Tokenizer loaded successfully")
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_to_load,
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, text: str, language: str) -> dict:
        """
        Make prediction on input text
        
        Args:
            text: Input text to classify
            language: Language code (ha, yo, ig, pcm)
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Validate language
            if language not in settings.SUPPORTED_LANGUAGES:
                raise ValueError(f"Unsupported language: {language}")
            
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=settings.MAX_SEQUENCE_LENGTH,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get prediction
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Get tokens for explainability
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            result = {
                "prediction": {
                    "label": predicted_class,
                    "label_text": settings.LABELS[predicted_class],
                    "confidence": float(confidence),
                    "probabilities": probabilities[0].cpu().tolist()
                },
                "language": language,
                "language_name": settings.LANGUAGE_NAMES.get(language, language),
                "tokens": tokens,
                "text_length": len(text),
                "token_count": len(tokens)
            }
            
            logger.info(f"Prediction made: {predicted_class} ({confidence:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def batch_predict(self, texts: list, languages: list) -> list:
        """
        Make predictions on multiple texts
        
        Args:
            texts: List of input texts
            languages: List of language codes
            
        Returns:
            List of prediction results
        """
        results = []
        for text, language in zip(texts, languages):
            try:
                result = self.predict(text, language)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch prediction error: {e}")
                results.append({"error": str(e)})
        
        return results
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_path": settings.MODEL_PATH,
            "model_name": settings.MODEL_NAME,
            "device": str(self.device),
            "supported_languages": settings.SUPPORTED_LANGUAGES,
            "max_sequence_length": settings.MAX_SEQUENCE_LENGTH,
            "labels": settings.LABELS
        }


# Create global model service instance
model_service = ModelService()

