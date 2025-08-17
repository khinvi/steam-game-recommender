"""Main FastAPI application entry point for the Steam Game Recommender."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import recommendations, games

app = FastAPI(
    title="Steam Game Recommender",
    description="AI-powered game recommendations using the Stanford SNAP Steam dataset",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(recommendations.router, prefix="/recommendations", tags=["recommendations"])
app.include_router(games.router, prefix="/games", tags=["games"])

@app.get("/")
async def root():
    return {
        "message": "Steam Game Recommender API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Steam Game Recommender",
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 