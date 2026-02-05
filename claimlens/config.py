"""Configuration settings for ClaimLens pipeline."""

import os
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    OPENAI_API_KEY: str = ""
    TAVILY_API_KEY: str = ""
    SERPAPI_KEY: str = ""
    
    # LLM Configuration
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.1
    
    # Verification Configuration
    MAX_VERIFICATION_ITERATIONS: int = 3
    CONFIDENCE_THRESHOLD: float = 0.7
    MAX_EVIDENCE_PER_CLAIM: int = 5
    SEARCH_RESULTS_PER_QUERY: int = 5
    
    # Verifier Selection
    VERIFIER_TYPE: Literal["huggingface", "openai"] = "openai"
    HF_NLI_MODEL: str = "facebook/bart-large-mnli"
    
    # Search Provider
    SEARCH_PROVIDER: Literal["tavily", "serpapi"] = "tavily"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Security Configuration
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000"  # Comma-separated list
    API_KEY: str = ""  # Optional API key for authentication
    RATE_LIMIT_REQUESTS: int = 100  # Requests per minute
    RATE_LIMIT_WINDOW: int = 60  # Window in seconds
    MAX_TEXT_LENGTH: int = 10000  # Maximum input text length
    MAX_JOBS_STORED: int = 1000  # Maximum jobs to keep in memory
    DEBUG_MODE: bool = False  # Set to True only in development
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
