"""Recommendation routes for the Steam Game Recommender API."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Query

from src.domain.entities.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendationType,
    UserGameInteraction,
    RecommendationFeedback
)
from src.domain.services.recommendation_service import RecommendationService
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize recommendation service
recommendation_service = RecommendationService()


@router.post("/generate", response_model=RecommendationResponse)
async def generate_recommendations(
    request: RecommendationRequest
) -> RecommendationResponse:
    """Generate personalized game recommendations for a user."""
    try:
        # Generate recommendations using the service
        recommendations = await recommendation_service.generate_recommendations(request)
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recommendations: {str(e)}"
        )


@router.get("/user/{user_id}", response_model=List[RecommendationResponse])
async def get_user_recommendations(
    user_id: int,
    recommendation_type: Optional[RecommendationType] = Query(None, description="Filter by recommendation type"),
    limit: int = Query(10, ge=1, le=100, description="Number of recommendations to return")
) -> List[RecommendationResponse]:
    """Get recommendation history for a user."""
    # Get recommendations from service
    recommendations = await recommendation_service.get_user_recommendations(
        user_id, recommendation_type, limit
    )
    return recommendations


@router.get("/types")
async def get_recommendation_types() -> List[str]:
    """Get all available recommendation types."""
    return [rt.value for rt in RecommendationType]


@router.get("/models")
async def get_available_models() -> dict:
    """Get available recommendation models and their performance metrics."""
    try:
        metrics = await recommendation_service.get_performance_metrics()
        return {
            "models": metrics.get("models_available", []),
            "performance": {
                "svd": "Best accuracy - 26% precision",
                "item_based": "Fast & simple - 8-12% precision", 
                "user_based": "Find similar players - 3-5% precision",
                "popularity": "Trending games - 5-8% precision",
                "hybrid": "Balanced approach - 15-20% precision"
            },
            "dataset_stats": {
                "total_users": metrics.get("total_users", 0),
                "total_games": metrics.get("total_games", 0),
                "total_interactions": metrics.get("total_interactions", 0),
                "data_source": metrics.get("data_source", "Unknown")
            }
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return {"error": "Failed to get model information"}


@router.get("/sample-users")
async def get_sample_users() -> List[dict]:
    """Get sample users for demo purposes."""
    try:
        sample_users = recommendation_service.get_sample_users()
        return sample_users
    except Exception as e:
        logger.error(f"Error getting sample users: {e}")
        return []


@router.post("/feedback")
async def submit_recommendation_feedback(
    feedback: RecommendationFeedback
) -> dict:
    """Submit feedback on a recommendation."""
    # Submit feedback using service
    await recommendation_service.submit_feedback(feedback)
    return {"message": "Feedback submitted successfully"}


@router.post("/interaction")
async def record_user_interaction(
    interaction: UserGameInteraction
) -> dict:
    """Record a user's interaction with a game."""
    # Record interaction using service
    await recommendation_service.record_user_interaction(interaction)
    return {"message": "Interaction recorded successfully"}


@router.get("/explanation/{recommendation_id}")
async def get_recommendation_explanation(
    recommendation_id: str
) -> dict:
    """Get explanation for why a recommendation was made."""
    try:
        explanation = await recommendation_service.get_recommendation_explanation(recommendation_id)
        return explanation
    except Exception as e:
        logger.error(f"Error getting explanation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get recommendation explanation"
        )


@router.get("/metrics")
async def get_performance_metrics() -> dict:
    """Get recommendation system performance metrics."""
    try:
        metrics = await recommendation_service.get_performance_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get performance metrics"
        )


@router.post("/refresh/{user_id}")
async def refresh_user_recommendations(
    user_id: int
) -> dict:
    """Force refresh of user recommendations."""
    # Refresh recommendations using service
    await recommendation_service.refresh_user_recommendations(user_id)
    return {"message": "Recommendations refreshed successfully"}


@router.delete("/{recommendation_id}")
async def delete_recommendation(
    recommendation_id: str
) -> dict:
    """Delete a specific recommendation."""
    try:
        # Delete recommendation using service
        success = await recommendation_service.delete_recommendation(recommendation_id)
        if success:
            return {"message": "Recommendation deleted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Recommendation not found"
            )
    except Exception as e:
        logger.error(f"Error deleting recommendation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete recommendation"
        ) 