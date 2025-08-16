"""Configuration management for the Steam Game Recommender application."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """
    Application settings with free service configurations
    """
    # Application
    APP_NAME: str = "Steam Game Recommender"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    
    # Database (SQLite for local, PostgreSQL for production)
    DATABASE_URL: str = "sqlite:///./data/app.db"  # Free local storage
    # For production: Use Supabase free tier
    # DATABASE_URL: str = "postgresql://user:pass@db.supabase.co:5432/postgres"
    
    # Redis Cache (Optional - fallback to in-memory)
    REDIS_URL: Optional[str] = None  # Use Upstash free tier in production
    USE_REDIS: bool = False
    
    # Vector Database (ChromaDB - local and free)
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    CHROMA_COLLECTION: str = "game_embeddings"
    
    # Model Configuration
    MODEL_DIR: str = "./data/models"
    USE_GPU: bool = False  # Set to True if you have GPU
    BATCH_SIZE: int = 32
    
    # Steam API (free but rate-limited)
    STEAM_API_KEY: Optional[str] = None
    STEAM_RATE_LIMIT: int = 100000  # Daily limit
    
    # Authentication (using simple JWT - no external service needed)
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Free Monitoring (optional)
    ENABLE_METRICS: bool = False
    PROMETHEUS_PORT: int = 9090
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings() 