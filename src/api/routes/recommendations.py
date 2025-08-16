"""Recommendation routes for the Steam Game Recommender API."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from slowapi.util import get_remote_address
from slowapi import Limiter

from src.domain.entities.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendationType,
    UserGameInteraction,
    RecommendationFeedback
)
from src.api.dependencies import get_current_user
from src.domain.services.recommendation_service import RecommendationService
from src.infrastructure.cache.redis_cache import cache
from src.infrastructure.ml.model_manager import model_manager
import logging
from src.infrastructure.vector_store.vector_store import VectorStore

# Rate limiter for recommendation endpoints
limiter = Limiter(key_func=get_remote_address)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/generate", response_model=RecommendationResponse)
@limiter.limit("20/minute")
async def generate_recommendations(
    request: RecommendationRequest,
    current_user: dict = Depends(get_current_user)
) -> RecommendationResponse:
    """Generate personalized game recommendations for a user."""
    # Ensure the user is requesting recommendations for themselves
    if request.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only generate recommendations for yourself"
        )
    
    # This is a placeholder - implement actual recommendation service
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Recommendation generation not implemented yet"
    )


@router.get("/user/{user_id}", response_model=List[RecommendationResponse])
@limiter.limit("60/minute")
async def get_user_recommendations(
    user_id: int,
    recommendation_type: RecommendationType = Query(None, description="Filter by recommendation type"),
    limit: int = Query(10, ge=1, le=100, description="Number of recommendations to return"),
    current_user: dict = Depends(get_current_user)
) -> List[RecommendationResponse]:
    """Get recommendation history for a user."""
    # Users can only view their own recommendations or admins can view any
    if current_user.id != user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only view own recommendations"
        )
    
    # This is a placeholder - implement actual recommendation service
    return []


@router.get("/types")
@limiter.limit("100/minute")
async def get_recommendation_types() -> List[str]:
    """Get all available recommendation types."""
    return [rt.value for rt in RecommendationType]


@router.post("/feedback")
@limiter.limit("100/minute")
async def submit_recommendation_feedback(
    feedback: RecommendationFeedback,
    current_user: dict = Depends(get_current_user)
) -> dict:
    """Submit feedback on a recommendation."""
    # Ensure the user is submitting feedback for themselves
    if feedback.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only submit feedback for yourself"
        )
    
    # This is a placeholder - implement actual feedback service
    return {"message": "Feedback submitted successfully"}


@router.post("/interaction")
@limiter.limit("200/minute")
async def record_user_interaction(
    interaction: UserGameInteraction,
    current_user: dict = Depends(get_current_user)
) -> dict:
    """Record a user's interaction with a game."""
    # Ensure the user is recording interaction for themselves
    if interaction.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only record interactions for yourself"
        )
    
    # This is a placeholder - implement actual interaction service
    return {"message": "Interaction recorded successfully"}


@router.get("/explanation/{recommendation_id}")
@limiter.limit("100/minute")
async def get_recommendation_explanation(
    recommendation_id: str,
    current_user: dict = Depends(get_current_user)
) -> dict:
    """Get explanation for why a recommendation was made."""
    # This is a placeholder - implement actual explanation service
    return {
        "recommendation_id": recommendation_id,
        "explanation": "This recommendation was made based on your gaming preferences and similar users' behavior.",
        "factors": [
            "Genre preference match",
            "Similar user behavior",
            "Game popularity",
            "Price range compatibility"
        ]
    }


@router.get("/performance")
@limiter.limit("30/minute")
async def get_recommendation_performance(
    current_user: dict = Depends(get_current_user)
) -> dict:
    """Get recommendation system performance metrics (admin only)."""
    # This is a placeholder - implement actual performance metrics
    return {
        "overall_precision": 0.75,
        "overall_recall": 0.68,
        "user_satisfaction": 0.82,
        "diversity_score": 0.71,
        "novelty_score": 0.65,
        "total_recommendations": 15000,
        "active_users": 2500
    }


@router.post("/refresh")
@limiter.limit("10/minute")
async def refresh_recommendations(
    current_user: dict = Depends(get_current_user)
) -> dict:
    """Force refresh of user recommendations."""
    # This is a placeholder - implement actual refresh service
    return {"message": "Recommendations refresh initiated"}


@router.delete("/{recommendation_id}")
@limiter.limit("50/minute")
async def delete_recommendation(
    recommendation_id: str,
    current_user: dict = Depends(get_current_user)
) -> dict:
    """Delete a specific recommendation."""
    # This is a placeholder - implement actual deletion service
    return {"message": "Recommendation deleted successfully"} 

@router.get("/personalized")
async def get_personalized_recommendations(
    n: int = Query(10, ge=1, le=50),
    model_type: str = Query("svd", description="Model to use"),
    current_user: dict = Depends(get_current_user)
):
    """Get personalized recommendations for the current user"""
    
    user_id = current_user.id
    if not user_id:
        raise HTTPException(status_code=400, detail="Invalid user information")
    
    # Check cache first
    cache_key = f"recs:{user_id}:{model_type}:{n}"
    try:
        cached = await cache.get(cache_key)
        if cached:
            logger.info(f"Cache hit for user {user_id}")
            return {"recommendations": cached, "from_cache": True}
    except Exception as e:
        logger.warning(f"Cache error for user {user_id}: {e}")
    
    # Generate recommendations
    try:
        recommendations = model_manager.get_recommendations(
            user_id=user_id,
            model_type=model_type,
            n_recommendations=n
        )
        
        # Cache for 5 minutes
        try:
            await cache.set(cache_key, recommendations, ttl=300)
        except Exception as e:
            logger.warning(f"Failed to cache recommendations for user {user_id}: {e}")
        
        return {
            "recommendations": recommendations,
            "from_cache": False,
            "model_used": model_type
        }
    except Exception as e:
        logger.error(f"Error generating recommendations for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate recommendations")

@router.get("/similar/{game_id}")
async def get_similar_games(
    game_id: str,
    n: int = Query(10, ge=1, le=50)
):
    """Get games similar to a specific game"""
    
    try:
        # Use vector store for similarity search
        vector_store = VectorStore()
        
        # Get game embedding
        embedding = vector_store.get_game_embedding(game_id)
        if not embedding:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Search for similar games
        results = vector_store.search_similar_games(
            query_embedding=embedding,
            n_results=n + 1  # +1 to exclude the game itself
        )
        
        # Filter out the query game and format results
        similar_games = []
        for result_id, distance, metadata in zip(
            results['ids'],
            results['distances'],
            results['metadata']
        ):
            if result_id != game_id:  # Exclude the query game
                similar_games.append({
                    "game_id": result_id,
                    "similarity": 1 - distance,  # Convert distance to similarity
                    "metadata": metadata
                })
                if len(similar_games) >= n:
                    break
        
        return {"similar_games": similar_games}
    except Exception as e:
        logger.error(f"Error finding similar games: {e}")
        raise HTTPException(status_code=500, detail="Failed to find similar games")

@router.get("/trending")
async def get_trending_games(
    period: str = Query("week", regex="^(day|week|month)$"),
    n: int = Query(10, ge=1, le=50)
):
    """Get trending games"""
    
    # For now, return mock data (implement with real analytics later)
    trending = [
        {
            "game_id": f"trending_{i}",
            "name": f"Trending Game {i}",
            "trend_score": 100 - i * 5,
            "player_count": 10000 - i * 500
        }
        for i in range(n)
    ]
    
    return {"trending_games": trending, "period": period} 