"""User entity models for the Steam Game Recommender application."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class UserRole(str, Enum):
    """User roles in the system."""
    USER = "user"
    ADMIN = "admin"
    MODERATOR = "moderator"


class UserPreferences(BaseModel):
    """User gaming preferences."""
    favorite_genres: List[str] = Field(default_factory=list, description="Favorite game genres")
    favorite_tags: List[str] = Field(default_factory=list, description="Favorite game tags")
    preferred_platforms: List[str] = Field(default_factory=list, description="Preferred gaming platforms")
    max_price: Optional[float] = Field(None, description="Maximum price willing to pay for games")
    playtime_preference: Optional[str] = Field(None, description="Preferred playtime (casual, moderate, hardcore)")
    multiplayer_preference: Optional[bool] = Field(None, description="Preference for multiplayer games")
    
    @validator('favorite_genres', 'favorite_tags', 'preferred_platforms')
    def validate_lists(cls, v):
        """Validate that lists contain only strings."""
        if not all(isinstance(item, str) for item in v):
            raise ValueError("All items must be strings")
        return v


class UserProfile(BaseModel):
    """User profile information."""
    steam_id: Optional[str] = Field(None, description="Steam ID if connected")
    display_name: str = Field(..., description="User display name")
    avatar_url: Optional[str] = Field(None, description="User avatar URL")
    bio: Optional[str] = Field(None, description="User bio")
    location: Optional[str] = Field(None, description="User location")
    join_date: Optional[datetime] = Field(None, description="When user joined the platform")
    last_active: Optional[datetime] = Field(None, description="Last active timestamp")
    
    @validator('steam_id')
    def validate_steam_id(cls, v):
        """Validate Steam ID format."""
        if v and not v.isdigit():
            raise ValueError("Steam ID must be numeric")
        return v


class User(BaseModel):
    """User entity."""
    id: Optional[int] = Field(None, description="Internal user ID")
    email: str = Field(..., description="User email address")
    username: str = Field(..., description="Unique username")
    hashed_password: str = Field(..., description="Hashed password")
    role: UserRole = Field(UserRole.USER, description="User role")
    is_active: bool = Field(True, description="Whether user account is active")
    is_verified: bool = Field(False, description="Whether email is verified")
    profile: Optional[UserProfile] = Field(None, description="User profile information")
    preferences: Optional[UserPreferences] = Field(None, description="User gaming preferences")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Account creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    @validator('email')
    def validate_email(cls, v):
        """Basic email validation."""
        if '@' not in v or '.' not in v:
            raise ValueError("Invalid email format")
        return v.lower()
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format."""
        if len(v) < 3 or len(v) > 30:
            raise ValueError("Username must be between 3 and 30 characters")
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Username can only contain letters, numbers, underscores, and hyphens")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "username": "gamer123",
                "hashed_password": "hashed_password_here",
                "role": "user",
                "is_active": True,
                "is_verified": False,
                "profile": {
                    "display_name": "Gamer123",
                    "bio": "Passionate gamer who loves RPGs and strategy games"
                },
                "preferences": {
                    "favorite_genres": ["RPG", "Strategy"],
                    "favorite_tags": ["Story Rich", "Open World"],
                    "preferred_platforms": ["PC", "Steam Deck"]
                }
            }
        }


class UserCreate(BaseModel):
    """Model for creating a new user."""
    email: str = Field(..., description="User email address")
    username: str = Field(..., description="Unique username")
    password: str = Field(..., description="Plain text password")
    display_name: Optional[str] = Field(None, description="User display name")
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class UserUpdate(BaseModel):
    """Model for updating user information."""
    email: Optional[str] = Field(None, description="User email address")
    username: Optional[str] = Field(None, description="Unique username")
    display_name: Optional[str] = Field(None, description="User display name")
    bio: Optional[str] = Field(None, description="User bio")
    location: Optional[str] = Field(None, description="User location")
    avatar_url: Optional[str] = Field(None, description="User avatar URL")
    preferences: Optional[UserPreferences] = Field(None, description="User gaming preferences")


class UserResponse(BaseModel):
    """Model for user response (without sensitive information)."""
    id: int = Field(..., description="Internal user ID")
    email: str = Field(..., description="User email address")
    username: str = Field(..., description="Unique username")
    role: UserRole = Field(..., description="User role")
    is_active: bool = Field(..., description="Whether user account is active")
    is_verified: bool = Field(..., description="Whether email is verified")
    profile: Optional[UserProfile] = Field(None, description="User profile information")
    preferences: Optional[UserPreferences] = Field(None, description="User gaming preferences")
    created_at: datetime = Field(..., description="Account creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class UserLogin(BaseModel):
    """Model for user login."""
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="Plain text password")


class UserToken(BaseModel):
    """Model for user authentication token."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    refresh_token: Optional[str] = Field(None, description="Refresh token for getting new access token") 