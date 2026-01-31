"""
Bias Detection Service
Detects gender, ethnic, and religious biases in text
"""

import re
from loguru import logger


class BiasDetectionService:
    """Service for detecting various types of bias in text"""
    
    def __init__(self):
        # Gender bias keywords (expandable)
        self.gender_keywords = {
            'male': ['man', 'men', 'boy', 'boys', 'male', 'he', 'him', 'his', 'father', 'son', 'brother', 'husband'],
            'female': ['woman', 'women', 'girl', 'girls', 'female', 'she', 'her', 'hers', 'mother', 'daughter', 'sister', 'wife']
        }
        
        # Ethnic bias keywords (Nigerian context)
        self.ethnic_keywords = [
            'hausa', 'yoruba', 'igbo', 'fulani', 'ijaw', 'kanuri', 'ibibio', 'tiv',
            'northerner', 'southerner', 'tribe', 'tribal', 'ethnic'
        ]
        
        # Religious bias keywords
        self.religious_keywords = [
            'muslim', 'islam', 'islamic', 'christian', 'christianity', 'church', 'mosque',
            'bible', 'quran', 'koran', 'allah', 'god', 'jesus', 'prophet', 'imam', 'pastor'
        ]
    
    def detect_bias(self, text: str, language: str) -> dict:
        """
        Detect all types of bias in text
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            Dictionary with bias scores
        """
        try:
            text_lower = text.lower()
            
            # Detect gender bias
            gender_bias = self._detect_gender_bias(text_lower)
            
            # Detect ethnic bias
            ethnic_bias = self._detect_ethnic_bias(text_lower)
            
            # Detect religious bias
            religious_bias = self._detect_religious_bias(text_lower)
            
            # Calculate overall bias
            overall_bias = (gender_bias + ethnic_bias + religious_bias) / 3
            
            result = {
                "genderBias": round(gender_bias, 4),
                "ethnicBias": round(ethnic_bias, 4),
                "religiousBias": round(religious_bias, 4),
                "overallBias": round(overall_bias, 4),
                "hasBias": overall_bias > 0.3,
                "biasLevel": self._get_bias_level(overall_bias)
            }
            
            logger.info(f"Bias detection completed: overall={overall_bias:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Bias detection error: {e}")
            return {
                "genderBias": 0.0,
                "ethnicBias": 0.0,
                "religiousBias": 0.0,
                "overallBias": 0.0,
                "hasBias": False,
                "biasLevel": "unknown",
                "error": str(e)
            }
    
    def _detect_gender_bias(self, text: str) -> float:
        """Detect gender bias based on keyword imbalance"""
        male_count = sum(text.count(word) for word in self.gender_keywords['male'])
        female_count = sum(text.count(word) for word in self.gender_keywords['female'])
        
        total = male_count + female_count
        if total == 0:
            return 0.0
        
        # Calculate imbalance (0 = balanced, 1 = completely imbalanced)
        imbalance = abs(male_count - female_count) / total
        return min(imbalance, 1.0)
    
    def _detect_ethnic_bias(self, text: str) -> float:
        """Detect ethnic bias based on keyword presence"""
        count = sum(text.count(word) for word in self.ethnic_keywords)
        
        # Normalize by text length (words)
        words = len(text.split())
        if words == 0:
            return 0.0
        
        # Score based on density of ethnic keywords
        density = count / words
        return min(density * 5, 1.0)  # Scale up and cap at 1.0
    
    def _detect_religious_bias(self, text: str) -> float:
        """Detect religious bias based on keyword presence"""
        count = sum(text.count(word) for word in self.religious_keywords)
        
        # Normalize by text length (words)
        words = len(text.split())
        if words == 0:
            return 0.0
        
        # Score based on density of religious keywords
        density = count / words
        return min(density * 5, 1.0)  # Scale up and cap at 1.0
    
    def _get_bias_level(self, score: float) -> str:
        """Convert bias score to categorical level"""
        if score < 0.2:
            return "low"
        elif score < 0.5:
            return "moderate"
        elif score < 0.7:
            return "high"
        else:
            return "very_high"
    
    def get_bias_details(self, text: str) -> dict:
        """Get detailed bias analysis with specific keywords found"""
        text_lower = text.lower()
        
        return {
            "gender_keywords_found": {
                "male": [w for w in self.gender_keywords['male'] if w in text_lower],
                "female": [w for w in self.gender_keywords['female'] if w in text_lower]
            },
            "ethnic_keywords_found": [w for w in self.ethnic_keywords if w in text_lower],
            "religious_keywords_found": [w for w in self.religious_keywords if w in text_lower]
        }

