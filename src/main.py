"""Main FastAPI application entry point for the Steam Game Recommender."""

import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
from src.core.config import settings
from src.api.routes import recommendations, auth, users, games
from src.infrastructure.database.connection import engine, Base
from src.infrastructure.cache.redis_cache import cache
from src.infrastructure.ml.model_manager import model_manager
from src.infrastructure.ml.vector_store import VectorStore
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting up Steam Game Recommender...")
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized")
    
    # Initialize models
    logger.info(f"Loaded {len(model_manager.models)} models")
    
    # Initialize vector store
    app.state.vector_store = VectorStore()
    logger.info("Vector store initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await cache.clear()

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs" if settings.DEBUG else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    recommendations.router,
    prefix=f"{settings.API_PREFIX}/recommendations",
    tags=["recommendations"]
)
app.include_router(
    auth.router,
    prefix=f"{settings.API_PREFIX}/auth",
    tags=["authentication"]
)
app.include_router(
    users.router,
    prefix=f"{settings.API_PREFIX}/users",
    tags=["users"]
)
app.include_router(
    games.router,
    prefix=f"{settings.API_PREFIX}/games",
    tags=["games"]
)

# Serve static files (if you have a built frontend)
if os.path.exists("frontend/build"):
    app.mount("/", StaticFiles(directory="frontend/build", html=True), name="static")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(model_manager.models),
        "cache_available": cache.redis_client is not None
    }

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    ) 