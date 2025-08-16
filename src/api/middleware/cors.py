"""CORS middleware configuration for the Steam Game Recommender API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.core.config import get_settings


def add_cors_middleware(app: FastAPI) -> None:
    """Add CORS middleware to the FastAPI application."""
    settings = get_settings()
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=settings.allowed_methods,
        allow_headers=settings.allowed_headers,
        expose_headers=["Content-Length", "Content-Type"],
        max_age=86400,  # 24 hours
    )


def get_cors_config() -> dict:
    """Get CORS configuration dictionary."""
    settings = get_settings()
    
    return {
        "allow_origins": settings.allowed_origins,
        "allow_credentials": True,
        "allow_methods": settings.allowed_methods,
        "allow_headers": settings.allowed_headers,
        "expose_headers": ["Content-Length", "Content-Type"],
        "max_age": 86400,
    } 