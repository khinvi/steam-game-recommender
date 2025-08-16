"""Game entity models for the Steam Game Recommender application."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class GameCategory(str, Enum):
    """Game categories."""
    ACTION = "Action"
    ADVENTURE = "Adventure"
    RPG = "RPG"
    STRATEGY = "Strategy"
    SIMULATION = "Simulation"
    SPORTS = "Sports"
    RACING = "Racing"
    PUZZLE = "Puzzle"
    INDIE = "Indie"
    CASUAL = "Casual"
    ARCADE = "Arcade"
    PLATFORMER = "Platformer"
    SHOOTER = "Shooter"
    FIGHTING = "Fighting"
    STEALTH = "Stealth"
    SURVIVAL = "Survival"
    HORROR = "Horror"
    VISUAL_NOVEL = "Visual Novel"


class GamePlatform(str, Enum):
    """Gaming platforms."""
    PC = "PC"
    MAC = "Mac"
    LINUX = "Linux"
    STEAM_DECK = "Steam Deck"
    VR = "VR"


class GamePrice(BaseModel):
    """Game pricing information."""
    current_price: Optional[float] = Field(None, description="Current price in USD")
    original_price: Optional[float] = Field(None, description="Original price in USD")
    discount_percent: Optional[int] = Field(None, description="Discount percentage")
    is_free: bool = Field(False, description="Whether the game is free")
    currency: str = Field("USD", description="Price currency")
    
    @validator('discount_percent')
    def validate_discount(cls, v):
        """Validate discount percentage."""
        if v is not None and (v < 0 or v > 100):
            raise ValueError("Discount must be between 0 and 100")
        return v


class GameRequirements(BaseModel):
    """Game system requirements."""
    minimum: Optional[Dict[str, str]] = Field(None, description="Minimum system requirements")
    recommended: Optional[Dict[str, str]] = Field(None, description="Recommended system requirements")
    
    @validator('minimum', 'recommended')
    def validate_requirements(cls, v):
        """Validate requirements format."""
        if v and not isinstance(v, dict):
            raise ValueError("Requirements must be a dictionary")
        return v


class GameMetadata(BaseModel):
    """Additional game metadata."""
    steam_app_id: Optional[int] = Field(None, description="Steam App ID")
    steam_url: Optional[str] = Field(None, description="Steam store URL")
    website: Optional[str] = Field(None, description="Official website")
    support_url: Optional[str] = Field(None, description="Support URL")
    developer: Optional[str] = Field(None, description="Game developer")
    publisher: Optional[str] = Field(None, description="Game publisher")
    release_date: Optional[datetime] = Field(None, description="Release date")
    last_updated: Optional[datetime] = Field(None, description="Last update date")


class Game(BaseModel):
    """Game entity."""
    id: Optional[int] = Field(None, description="Internal game ID")
    title: str = Field(..., description="Game title")
    description: Optional[str] = Field(None, description="Game description")
    short_description: Optional[str] = Field(None, description="Short game description")
    categories: List[GameCategory] = Field(default_factory=list, description="Game categories")
    tags: List[str] = Field(default_factory=list, description="Game tags")
    platforms: List[GamePlatform] = Field(default_factory=list, description="Supported platforms")
    price: Optional[GamePrice] = Field(None, description="Game pricing information")
    requirements: Optional[GameRequirements] = Field(None, description="System requirements")
    metadata: Optional[GameMetadata] = Field(None, description="Additional metadata")
    
    # Game statistics
    rating: Optional[float] = Field(None, description="Average user rating (1-5)")
    rating_count: Optional[int] = Field(None, description="Number of ratings")
    playtime_median: Optional[int] = Field(None, description="Median playtime in minutes")
    playtime_mean: Optional[int] = Field(None, description="Mean playtime in minutes")
    owners_estimate: Optional[int] = Field(None, description="Estimated number of owners")
    
    # Media
    header_image: Optional[str] = Field(None, description="Header image URL")
    screenshots: List[str] = Field(default_factory=list, description="Screenshot URLs")
    background: Optional[str] = Field(None, description="Background image URL")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Record creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    @validator('rating')
    def validate_rating(cls, v):
        """Validate rating range."""
        if v is not None and (v < 1 or v > 5):
            raise ValueError("Rating must be between 1 and 5")
        return v
    
    @validator('playtime_median', 'playtime_mean')
    def validate_playtime(cls, v):
        """Validate playtime values."""
        if v is not None and v < 0:
            raise ValueError("Playtime must be non-negative")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "title": "The Witcher 3: Wild Hunt",
                "description": "An epic role-playing game with a gripping story and vast open world.",
                "short_description": "Epic RPG with gripping story and vast open world",
                "categories": ["RPG", "Action", "Adventure"],
                "tags": ["Story Rich", "Open World", "RPG", "Fantasy"],
                "platforms": ["PC", "Mac", "Linux"],
                "price": {
                    "current_price": 39.99,
                    "original_price": 59.99,
                    "discount_percent": 33,
                    "is_free": False,
                    "currency": "USD"
                },
                "rating": 4.8,
                "rating_count": 150000,
                "playtime_median": 4800,
                "playtime_mean": 5200
            }
        }


class GameCreate(BaseModel):
    """Model for creating a new game."""
    title: str = Field(..., description="Game title")
    description: Optional[str] = Field(None, description="Game description")
    short_description: Optional[str] = Field(None, description="Short game description")
    categories: List[GameCategory] = Field(default_factory=list, description="Game categories")
    tags: List[str] = Field(default_factory=list, description="Game tags")
    platforms: List[GamePlatform] = Field(default_factory=list, description="Supported platforms")
    steam_app_id: Optional[int] = Field(None, description="Steam App ID")


class GameUpdate(BaseModel):
    """Model for updating game information."""
    title: Optional[str] = Field(None, description="Game title")
    description: Optional[str] = Field(None, description="Game description")
    short_description: Optional[str] = Field(None, description="Short game description")
    categories: Optional[List[GameCategory]] = Field(None, description="Game categories")
    tags: Optional[List[str]] = Field(None, description="Game tags")
    platforms: Optional[List[GamePlatform]] = Field(None, description="Supported platforms")
    price: Optional[GamePrice] = Field(None, description="Game pricing information")
    requirements: Optional[GameRequirements] = Field(None, description="System requirements")
    rating: Optional[float] = Field(None, description="Average user rating")
    playtime_median: Optional[int] = Field(None, description="Median playtime in minutes")
    playtime_mean: Optional[int] = Field(None, description="Mean playtime in minutes")


class GameResponse(BaseModel):
    """Model for game response."""
    id: int = Field(..., description="Internal game ID")
    title: str = Field(..., description="Game title")
    description: Optional[str] = Field(None, description="Game description")
    short_description: Optional[str] = Field(None, description="Short game description")
    categories: List[GameCategory] = Field(..., description="Game categories")
    tags: List[str] = Field(..., description="Game tags")
    platforms: List[GamePlatform] = Field(..., description="Supported platforms")
    price: Optional[GamePrice] = Field(None, description="Game pricing information")
    requirements: Optional[GameRequirements] = Field(None, description="System requirements")
    metadata: Optional[GameMetadata] = Field(None, description="Additional metadata")
    rating: Optional[float] = Field(None, description="Average user rating")
    rating_count: Optional[int] = Field(None, description="Number of ratings")
    playtime_median: Optional[int] = Field(None, description="Median playtime in minutes")
    playtime_mean: Optional[int] = Field(None, description="Mean playtime in minutes")
    owners_estimate: Optional[int] = Field(None, description="Estimated number of owners")
    header_image: Optional[str] = Field(None, description="Header image URL")
    screenshots: List[str] = Field(..., description="Screenshot URLs")
    background: Optional[str] = Field(None, description="Background image URL")
    created_at: datetime = Field(..., description="Record creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class GameSearch(BaseModel):
    """Model for game search parameters."""
    query: Optional[str] = Field(None, description="Search query")
    categories: Optional[List[GameCategory]] = Field(None, description="Filter by categories")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    platforms: Optional[List[GamePlatform]] = Field(None, description="Filter by platforms")
    min_price: Optional[float] = Field(None, description="Minimum price")
    max_price: Optional[float] = Field(None, description="Maximum price")
    min_rating: Optional[float] = Field(None, description="Minimum rating")
    sort_by: Optional[str] = Field("relevance", description="Sort field")
    sort_order: Optional[str] = Field("desc", description="Sort order (asc/desc)")
    page: int = Field(1, description="Page number")
    page_size: int = Field(20, description="Items per page")
    
    @validator('min_price', 'max_price')
    def validate_price_range(cls, v):
        """Validate price values."""
        if v is not None and v < 0:
            raise ValueError("Price must be non-negative")
        return v
    
    @validator('min_rating')
    def validate_min_rating(cls, v):
        """Validate minimum rating."""
        if v is not None and (v < 1 or v > 5):
            raise ValueError("Rating must be between 1 and 5")
        return v 