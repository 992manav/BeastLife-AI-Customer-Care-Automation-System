import os
from typing import Literal
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Application configuration settings."""
    
    # LLM Configuration
    gemini_api_key: str = ""
    groq_api_key: str = ""
    llm_provider: Literal["gemini", "groq"] = "groq"
    
    # Database Configuration
    database_url: str = "sqlite:///./beastlife_care.db"
    
    # FastAPI Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_env: str = "development"
    
    # Streamlit Configuration
    streamlit_port: int = 8501
    
    # RAG Configuration
    faiss_index_path: str = "./data/faiss_index"
    docs_path: str = "./data/docs"
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"
    
    # Application Settings
    system_prompt_path: str = "./config/system_prompt.txt"
    
    # Model-specific settings
    gemini_model: str = "gemini-1.5-flash"
    groq_model: str = "mixtral-8x7b-32768"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

def get_settings() -> Settings:
    """Get application settings from environment variables."""
    return Settings()

def validate_llm_configuration():
    """Validate that required LLM credentials are configured."""
    settings = get_settings()
    
    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not set in environment variables")
    elif settings.llm_provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY not set in environment variables")
    else:
        raise ValueError(f"Invalid LLM_PROVIDER: {settings.llm_provider}")
    
    return settings
