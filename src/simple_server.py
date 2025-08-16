"""Simple FastAPI server for the Steam Game Recommender - Database and Basic API only."""

import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from src.core.config import settings
from src.infrastructure.database.connection import engine, Base, SessionLocal
from src.infrastructure.database.models import User, Game, Recommendation
from src.recommendation_api import router as recommendation_router
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting up Steam Game Recommender (Simple Version)...")
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME + " (Simple Version)",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class UserCreate(BaseModel):
    username: str
    email: str
    hashed_password: str
    steam_id: Optional[str] = None
    preferences: dict = {}

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    steam_id: Optional[str] = None
    preferences: dict
    created_at: str

class GameCreate(BaseModel):
    steam_id: str
    name: str
    genres: List[str] = []
    tags: List[str] = []
    price: float = 0.0
    game_data: dict = {}

class GameResponse(BaseModel):
    id: int
    steam_id: str
    name: str
    genres: List[str]
    tags: List[str]
    price: float
    game_data: dict

class RecommendationCreate(BaseModel):
    user_id: int
    game_id: int
    score: float
    type: str

class RecommendationResponse(BaseModel):
    id: int
    user_id: int
    game_id: int
    score: float
    type: str
    created_at: str

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "simple",
        "database": "connected",
        "endpoints": [
            "/health",
            "/users",
            "/games", 
            "/recommendations",
            "/docs"
        ]
    }

# User endpoints
@app.post("/users", response_model=UserResponse)
def create_user(user: UserCreate):
    """Create a new user"""
    try:
        db = SessionLocal()
        db_user = User(
            username=user.username,
            email=user.email,
            hashed_password=user.hashed_password,
            steam_id=user.steam_id,
            preferences=user.preferences
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        result = UserResponse(
            id=db_user.id,
            username=db_user.username,
            email=db_user.email,
            steam_id=db_user.steam_id,
            preferences=db_user.preferences,
            created_at=db_user.created_at.isoformat()
        )
        db.close()
        return result
    except Exception as e:
        if 'db' in locals():
            db.rollback()
            db.close()
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")

@app.get("/users", response_model=List[UserResponse])
def get_users(skip: int = 0, limit: int = 100):
    """Get all users"""
    try:
        db = SessionLocal()
        users = db.query(User).offset(skip).limit(limit).all()
        result = [
            UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                steam_id=user.steam_id,
                preferences=user.preferences,
                created_at=user.created_at.isoformat()
            )
            for user in users
        ]
        db.close()
        return result
    except Exception as e:
        if 'db' in locals():
            db.close()
        raise HTTPException(status_code=500, detail=f"Failed to get users: {str(e)}")

@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int):
    """Get a specific user"""
    try:
        db = SessionLocal()
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            db.close()
            raise HTTPException(status_code=404, detail="User not found")
        
        result = UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            steam_id=user.steam_id,
            preferences=user.preferences,
            created_at=user.created_at.isoformat()
        )
        db.close()
        return result
    except Exception as e:
        if 'db' in locals():
            db.close()
        raise HTTPException(status_code=500, detail=f"Failed to get user: {str(e)}")

# Game endpoints
@app.post("/games", response_model=GameResponse)
def create_game(game: GameCreate):
    """Create a new game"""
    try:
        db = SessionLocal()
        db_game = Game(
            steam_id=game.steam_id,
            name=game.name,
            genres=game.genres,
            tags=game.tags,
            price=game.price,
            game_data=game.game_data
        )
        db.add(db_game)
        db.commit()
        db.refresh(db_game)
        
        result = GameResponse(
            id=db_game.id,
            steam_id=db_game.steam_id,
            name=db_game.name,
            genres=db_game.genres,
            tags=db_game.tags,
            price=db_game.price,
            game_data=db_game.game_data
        )
        db.close()
        return result
    except Exception as e:
        if 'db' in locals():
            db.rollback()
            db.close()
        raise HTTPException(status_code=500, detail=f"Failed to create game: {str(e)}")

@app.get("/games", response_model=List[GameResponse])
def get_games(skip: int = 0, limit: int = 100):
    """Get all games"""
    try:
        db = SessionLocal()
        games = db.query(Game).offset(skip).limit(limit).all()
        result = [
            GameResponse(
                id=game.id,
                steam_id=game.steam_id,
                name=game.name,
                genres=game.genres,
                tags=game.tags,
                price=game.price,
                game_data=game.game_data
            )
            for game in games
        ]
        db.close()
        return result
    except Exception as e:
        if 'db' in locals():
            db.close()
        raise HTTPException(status_code=500, detail=f"Failed to get games: {str(e)}")

@app.get("/games/{game_id}", response_model=GameResponse)
def get_game(game_id: int):
    """Get a specific game"""
    try:
        db = SessionLocal()
        game = db.query(Game).filter(Game.id == game_id).first()
        if game is None:
            db.close()
            raise HTTPException(status_code=404, detail="Game not found")
        
        result = GameResponse(
            id=game.id,
            steam_id=game.steam_id,
            name=game.name,
            genres=game.genres,
            tags=game.tags,
            price=game.price,
            game_data=game.game_data
        )
        db.close()
        return result
    except Exception as e:
        if 'db' in locals():
            db.close()
        raise HTTPException(status_code=500, detail=f"Failed to get game: {str(e)}")

# Recommendation endpoints
@app.post("/recommendations", response_model=RecommendationResponse)
def create_recommendation(rec: RecommendationCreate):
    """Create a new recommendation"""
    try:
        db = SessionLocal()
        db_rec = Recommendation(
            user_id=rec.user_id,
            game_id=rec.game_id,
            score=rec.score,
            type=rec.type
        )
        db.add(db_rec)
        db.commit()
        db.refresh(db_rec)
        
        result = RecommendationResponse(
            id=db_rec.id,
            user_id=db_rec.user_id,
            game_id=db_rec.game_id,
            score=db_rec.score,
            type=db_rec.type,
            created_at=db_rec.created_at.isoformat()
        )
        db.close()
        return result
    except Exception as e:
        if 'db' in locals():
            db.rollback()
            db.close()
        raise HTTPException(status_code=500, detail=f"Failed to create recommendation: {str(e)}")

@app.get("/recommendations", response_model=List[RecommendationResponse])
def get_recommendations(skip: int = 0, limit: int = 100):
    """Get all recommendations"""
    try:
        db = SessionLocal()
        recs = db.query(Recommendation).offset(skip).limit(limit).all()
        result = [
            RecommendationResponse(
                id=rec.id,
                user_id=rec.user_id,
                game_id=rec.game_id,
                score=rec.score,
                type=rec.type,
                created_at=rec.created_at.isoformat()
            )
            for rec in recs
        ]
        db.close()
        return result
    except Exception as e:
        if 'db' in locals():
            db.close()
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

@app.get("/recommendations/user/{user_id}", response_model=List[RecommendationResponse])
def get_user_recommendations(user_id: int):
    """Get recommendations for a specific user"""
    try:
        db = SessionLocal()
        recs = db.query(Recommendation).filter(Recommendation.user_id == user_id).all()
        result = [
            RecommendationResponse(
                id=rec.id,
                user_id=rec.user_id,
                game_id=rec.game_id,
                score=rec.score,
                type=rec.type,
                created_at=rec.created_at.isoformat()
            )
            for rec in recs
        ]
        db.close()
        return result
    except Exception as e:
        if 'db' in locals():
            db.close()
        raise HTTPException(status_code=500, detail=f"Failed to get user recommendations: {str(e)}")

# Include AI recommendation router (without prefix since it's already defined in the router)
app.include_router(recommendation_router, tags=["ai-recommendations"])

if __name__ == "__main__":
    uvicorn.run(
        "src.simple_server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    ) 