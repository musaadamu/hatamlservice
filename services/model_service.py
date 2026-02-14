"""
Model Service - Handles inference via HuggingFace Inference API
Optimized for Render free tier (no heavy model loading)
"""

import httpx
from loguru import logger
from config import settings


class ModelService:
    """Service for inference via HuggingFace Inference API"""
    
    def __init__(self):
        self.model = None  # Not used with API approach
        self.tokenizer = None  # Not used with API approach
        self.device = None  # Not used with API approach
        self.api_endpoint = settings.HF_API_ENDPOINT
        self.hf_token = settings.HF_TOKEN
        self.load_model()
    
    def load_model(self):
        """Initialize HuggingFace Inference API connection"""
        try:
            if not self.hf_token:
                logger.warning("HF_TOKEN not set. Using HuggingFace Inference API without authentication.")
                logger.warning("This may result in rate limiting. Set HF_TOKEN in .env for better performance.")
            
            logger.info(f"Using HuggingFace Inference API")
            logger.info(f"Model endpoint: {self.api_endpoint}")
            logger.info("Model service initialized successfully (using API)")
            
        except Exception as e:
            logger.error(f"Error initializing model service: {e}")
            raise
    
    def _prepare_headers(self) -> dict:
        """Prepare headers for API request"""
        headers = {"Content-Type": "application/json"}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        return headers
    
    def predict(self, text: str, language: str) -> dict:
        """
        Make prediction via HuggingFace Inference API
        
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
            
            # Prepare payload for text classification
            payload = {
                "inputs": text,
                "parameters": {
                    "truncation": True
                }
            }
            
            headers = self._prepare_headers()
            
            # Call HuggingFace Inference API
            try:
                response = httpx.post(
                    self.api_endpoint,
                    json=payload,
                    headers=headers,
                    timeout=settings.INFERENCE_API_TIMEOUT
                )
                response.raise_for_status()
            except httpx.TimeoutException:
                logger.error("HuggingFace API request timed out")
                raise ValueError("Model inference timed out. Please try again.")
            except httpx.HTTPError as e:
                logger.error(f"HuggingFace API error: {e}")
                raise ValueError(f"Model inference failed: {str(e)}")
            
            # Parse response
            api_response = response.json()
            
            # Handle different response formats from HF API
            if isinstance(api_response, list) and len(api_response) > 0:
                # Standard text classification response format
                predictions = api_response[0]
            else:
                logger.error(f"Unexpected API response format: {api_response}")
                raise ValueError("Unexpected response from HuggingFace API")
            
            # Extract label and score
            # HF API returns list of dicts with 'label' and 'score'
            if not isinstance(predictions, list) or len(predictions) == 0:
                raise ValueError("Invalid prediction format from API")
            
            # Sort by score (descending) to get top prediction
            predictions = sorted(predictions, key=lambda x: x.get('score', 0), reverse=True)
            
            # Map label to our schema (0 = Human-written, 1 = AI-generated)
            top_prediction = predictions[0]
            label_text = top_prediction.get('label', 'UNKNOWN')
            confidence = top_prediction.get('score', 0.0)
            
            # Convert label text to label index
            # The model returns either "LABEL_0" (Human) or "LABEL_1" (AI)
            predicted_class = 0 if "LABEL_0" in label_text else 1
            
            # Build probabilities array [human_prob, ai_prob]
            probabilities = [0.0, 0.0]
            for pred in predictions[:2]:
                label = pred.get('label', '')
                score = pred.get('score', 0.0)
                if "LABEL_0" in label:
                    probabilities[0] = score
                elif "LABEL_1" in label:
                    probabilities[1] = score
            
            # Simple tokenization by splitting on whitespace
            tokens = text.split()
            
            result = {
                "prediction": {
                    "label": predicted_class,
                    "label_text": settings.LABELS[predicted_class],
                    "confidence": float(confidence),
                    "probabilities": probabilities,
                    "human_prob": float(probabilities[0]),
                    "ai_prob": float(probabilities[1])
                },
                "language": language,
                "language_name": settings.LANGUAGE_NAMES.get(language, language),
                "tokens": tokens,
                "text_length": len(text),
                "token_count": len(tokens),
                "inference_source": "huggingface_api"
            }
            
            logger.info(f"Prediction via HF API: {predicted_class} ({confidence:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def batch_predict(self, texts: list, languages: list) -> list:
        """
        Make predictions on multiple texts
        Uses sequential API calls (respects rate limits better)
        
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
                logger.error(f"Batch prediction error for text: {e}")
                results.append({
                    "error": str(e),
                    "language": language,
                    "inference_source": "huggingface_api"
                })
        
        return results
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": settings.MODEL_NAME,
            "inference_type": "huggingface_api",
            "api_endpoint": settings.HF_API_ENDPOINT,
            "supported_languages": settings.SUPPORTED_LANGUAGES,
            "max_sequence_length": settings.MAX_SEQUENCE_LENGTH,
            "labels": settings.LABELS,
            "note": "Running on HuggingFace Inference API - optimized for Render free tier"
        }


# Create global model service instance
model_service = ModelService()

