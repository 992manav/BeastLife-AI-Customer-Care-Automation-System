import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import json
from src.core.config import get_settings
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text response from prompt."""
        pass
    
    @abstractmethod
    async def classify(self, text: str, categories: list) -> Dict[str, Any]:
        """Classify text into one of the given categories."""
        pass
    
    @abstractmethod
    async def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from text."""
        pass
    
    @abstractmethod
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        pass

class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""
    
    def __init__(self):
        try:
            import google.generativeai as genai
            self.genai = genai
            settings = get_settings()
            self.genai.configure(api_key=settings.gemini_api_key)
            self.model_name = settings.gemini_model
            self.model = self.genai.GenerativeModel(self.model_name)
            logger.info(f"Initialized Gemini provider with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini provider: {e}")
            raise
    
    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text using Gemini."""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(prompt)
            )
            return response.text
        except Exception as e:
            logger.error(f"Error generating text with Gemini: {e}")
            raise
    
    async def classify(self, text: str, categories: list) -> Dict[str, Any]:
        """Classify text using Gemini."""
        prompt = f"""Classify the following text into one of these categories: {', '.join(categories)}
        
Text: {text}

Respond in JSON format:
{{
    "category": "chosen category",
    "confidence": 0.0-1.0
}}"""
        
        try:
            response_text = await self.generate(prompt, temperature=0.2)
            # Parse JSON from response
            response_dict = json.loads(response_text)
            return response_dict
        except Exception as e:
            logger.error(f"Error classifying text with Gemini: {e}")
            return {"category": "unknown", "confidence": 0.0}
    
    async def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities using Gemini."""
        prompt = f"""Extract named entities and key information from the following text:

Text: {text}

Respond in JSON format with entity types and values:
{{
    "order_id": [],
    "product_name": [],
    "customer_issue": [],
    "phone": [],
    "email": []
}}"""
        
        try:
            response_text = await self.generate(prompt, temperature=0.1)
            response_dict = json.loads(response_text)
            return response_dict
        except Exception as e:
            logger.error(f"Error extracting entities with Gemini: {e}")
            return {}
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using Gemini."""
        prompt = f"""Analyze the sentiment of the following text:

Text: {text}

Respond in JSON format:
{{
    "sentiment": "positive|negative|neutral|critical",
    "confidence": 0.0-1.0,
    "score": -1.0 to 1.0
}}"""
        
        try:
            response_text = await self.generate(prompt, temperature=0.1)
            response_dict = json.loads(response_text)
            return response_dict
        except Exception as e:
            logger.error(f"Error analyzing sentiment with Gemini: {e}")
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "score": 0.0
            }

class GroqProvider(LLMProvider):
    """Groq API provider."""
    
    def __init__(self):
        try:
            from groq import Groq
            settings = get_settings()
            self.client = Groq(api_key=settings.groq_api_key)
            self.model_name = settings.groq_model
            logger.info(f"Initialized Groq provider with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq provider: {e}")
            raise
    
    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text using Groq."""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1024
                )
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text with Groq: {e}")
            raise
    
    async def classify(self, text: str, categories: list) -> Dict[str, Any]:
        """Classify text using Groq."""
        prompt = f"""Classify the following text into one of these categories: {', '.join(categories)}
        
Text: {text}

Respond in JSON format:
{{
    "category": "chosen category",
    "confidence": 0.0-1.0
}}"""
        
        try:
            response_text = await self.generate(prompt, temperature=0.2)
            # Extract JSON from response
            response_dict = json.loads(response_text)
            return response_dict
        except Exception as e:
            logger.error(f"Error classifying text with Groq: {e}")
            return {"category": "unknown", "confidence": 0.0}
    
    async def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities using Groq."""
        prompt = f"""Extract named entities and key information from the following text:

Text: {text}

Respond in JSON format with entity types and values:
{{
    "order_id": [],
    "product_name": [],
    "customer_issue": [],
    "phone": [],
    "email": []
}}"""
        
        try:
            response_text = await self.generate(prompt, temperature=0.1)
            response_dict = json.loads(response_text)
            return response_dict
        except Exception as e:
            logger.error(f"Error extracting entities with Groq: {e}")
            return {}
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using Groq."""
        prompt = f"""Analyze the sentiment of the following text:

Text: {text}

Respond in JSON format:
{{
    "sentiment": "positive|negative|neutral|critical",
    "confidence": 0.0-1.0,
    "score": -1.0 to 1.0
}}"""
        
        try:
            response_text = await self.generate(prompt, temperature=0.1)
            response_dict = json.loads(response_text)
            return response_dict
        except Exception as e:
            logger.error(f"Error analyzing sentiment with Groq: {e}")
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "score": 0.0
            }

def get_llm_provider() -> LLMProvider:
    """Factory function to get the appropriate LLM provider."""
    settings = get_settings()
    
    if settings.llm_provider == "gemini":
        return GeminiProvider()
    elif settings.llm_provider == "groq":
        return GroqProvider()
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
