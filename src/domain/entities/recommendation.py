"""Recommendation entity models for the Steam Game Recommender application."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class RecommendationType(str, Enum):
    """Types of recommendations."""
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"
    NEURAL = "neural"
    POPULARITY = "popularity"
    SIMILARITY = "similarity"


class RecommendationReason(str, Enum):
    """Reasons for recommendations."""
    SIMILAR_USERS = "similar_users"
    SIMILAR_GAMES = "similar_games"
    GENRE_PREFERENCE = "genre_preference"
    TAG_PREFERENCE = "tag_preference"
    POPULARITY = "popularity"
    RECENT_RELEASE = "recent_release"
    PRICE_RANGE = "price_range"
    PLATFORM_COMPATIBILITY = "platform_compatibility"


class RecommendationScore(BaseModel):
    """Recommendation score and metadata."""
    score: float = Field(..., description="Recommendation score (0-1)")
    confidence: float = Field(..., description="Confidence in the recommendation (0-1)")
    reason: RecommendationReason = Field(..., description="Reason for the recommendation")
    explanation: Optional[str] = Field(None, description="Human-readable explanation")
    
    @validator('score', 'confidence')
    def validate_score_range(cls, v):
        """Validate score and confidence are between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError("Score and confidence must be between 0 and 1")
        return v


class GameRecommendation(BaseModel):
    """Individual game recommendation."""
    game_id: int = Field(..., description="Recommended game ID")
    game_title: str = Field(..., description="Game title")
    game_image: Optional[str] = Field(None, description="Game header image")
    score: RecommendationScore = Field(..., description="Recommendation score and metadata")
    game_categories: List[str] = Field(default_factory=list, description="Game categories")
    game_tags: List[str] = Field(default_factory=list, description="Game tags")
    game_price: Optional[float] = Field(None, description="Game current price")
    game_rating: Optional[float] = Field(None, description="Game average rating")


class RecommendationRequest(BaseModel):
    """Request for generating recommendations."""
    user_id: int = Field(..., description="User ID requesting recommendations")
    recommendation_type: RecommendationType = Field(..., description="Type of recommendation to generate")
    limit: int = Field(10, description="Maximum number of recommendations")
    include_played: bool = Field(False, description="Include games user has already played")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")
    
    @validator('limit')
    def validate_limit(cls, v):
        """Validate recommendation limit."""
        if v < 1 or v > 100:
            raise ValueError("Limit must be between 1 and 100")
        return v


class RecommendationResponse(BaseModel):
    """Response containing game recommendations."""
    user_id: int = Field(..., description="User ID")
    recommendation_type: RecommendationType = Field(..., description="Type of recommendation generated")
    recommendations: List[GameRecommendation] = Field(..., description="List of game recommendations")
    total_count: int = Field(..., description="Total number of recommendations available")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="When recommendations were generated")
    model_version: Optional[str] = Field(None, description="ML model version used")
    processing_time: Optional[float] = Field(None, description="Time taken to generate recommendations in seconds")


class UserGameInteraction(BaseModel):
    """User interaction with a game."""
    user_id: int = Field(..., description="User ID")
    game_id: int = Field(..., description="Game ID")
    interaction_type: str = Field(..., description="Type of interaction (play, wishlist, purchase, etc.)")
    rating: Optional[float] = Field(None, description="User rating (1-5)")
    playtime_minutes: Optional[int] = Field(None, description="Playtime in minutes")
    last_played: Optional[datetime] = Field(None, description="Last played timestamp")
    interaction_date: datetime = Field(default_factory=datetime.utcnow, description="When interaction occurred")
    
    @validator('rating')
    def validate_rating(cls, v):
        """Validate rating range."""
        if v is not None and (v < 1 or v > 5):
            raise ValueError("Rating must be between 1 and 5")
        return v
    
    @validator('playtime_minutes')
    def validate_playtime(cls, v):
        """Validate playtime value."""
        if v is not None and v < 0:
            raise ValueError("Playtime must be non-negative")
        return v


class RecommendationFeedback(BaseModel):
    """User feedback on recommendations."""
    user_id: int = Field(..., description="User ID")
    recommendation_id: Optional[str] = Field(None, description="Recommendation ID")
    game_id: int = Field(..., description="Game ID from recommendation")
    feedback_type: str = Field(..., description="Type of feedback (like, dislike, neutral)")
    feedback_date: datetime = Field(default_factory=datetime.utcnow, description="When feedback was given")
    additional_notes: Optional[str] = Field(None, description="Additional user notes")
    
    @validator('feedback_type')
    def validate_feedback_type(cls, v):
        """Validate feedback type."""
        valid_types = ["like", "dislike", "neutral", "purchased", "wishlisted"]
        if v not in valid_types:
            raise ValueError(f"Feedback type must be one of: {valid_types}")
        return v


class RecommendationMetrics(BaseModel):
    """Metrics for recommendation quality."""
    precision_at_k: Optional[float] = Field(None, description="Precision at K")
    recall_at_k: Optional[float] = Field(None, description="Recall at K")
    ndcg_at_k: Optional[float] = Field(None, description="NDCG at K")
    diversity: Optional[float] = Field(None, description="Recommendation diversity")
    novelty: Optional[float] = Field(None, description="Recommendation novelty")
    coverage: Optional[float] = Field(None, description="Recommendation coverage")
    user_satisfaction: Optional[float] = Field(None, description="User satisfaction score")
    
    @validator('precision_at_k', 'recall_at_k', 'ndcg_at_k', 'diversity', 'novelty', 'coverage', 'user_satisfaction')
    def validate_metrics(cls, v):
        """Validate metric values are between 0 and 1."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Metrics must be between 0 and 1")
        return v


class RecommendationHistory(BaseModel):
    """Historical record of recommendations."""
    id: Optional[int] = Field(None, description="Recommendation history ID")
    user_id: int = Field(..., description="User ID")
    recommendation_type: RecommendationType = Field(..., description="Type of recommendation")
    recommendations: List[GameRecommendation] = Field(..., description="Generated recommendations")
    metrics: Optional[RecommendationMetrics] = Field(None, description="Recommendation quality metrics")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="When recommendations were generated")
    model_version: Optional[str] = Field(None, description="ML model version used")
    user_feedback: Optional[List[RecommendationFeedback]] = Field(None, description="User feedback on recommendations")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": 123,
                "recommendation_type": "hybrid",
                "recommendations": [
                    {
                        "game_id": 456,
                        "game_title": "The Witcher 3: Wild Hunt",
                        "score": {
                            "score": 0.95,
                            "confidence": 0.88,
                            "reason": "genre_preference",
                            "explanation": "Based on your love for RPG games"
                        },
                        "game_categories": ["RPG", "Action", "Adventure"],
                        "game_tags": ["Story Rich", "Open World"],
                        "game_price": 39.99,
                        "game_rating": 4.8
                    }
                ],
                "generated_at": "2024-01-15T10:30:00Z"
            }
        } 