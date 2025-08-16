"""User management routes for the Steam Game Recommender API."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from slowapi.util import get_remote_address
from slowapi import Limiter

from src.domain.entities.user import (
    UserResponse,
    UserUpdate,
    UserPreferences
)
from src.api.dependencies import get_current_user, get_current_admin_user
from src.domain.services.user_service import UserService

# Rate limiter for user endpoints
limiter = Limiter(key_func=get_remote_address)

router = APIRouter()


@router.get("/", response_model=List[UserResponse])
@limiter.limit("30/minute")
async def get_users(
    skip: int = Query(0, ge=0, description="Number of users to skip"),
    limit: int = Query(100, ge=1, le=100, description="Maximum number of users to return"),
    current_admin: UserResponse = Depends(get_current_admin_user)
) -> List[UserResponse]:
    """Get all users (admin only)."""
    # This is a placeholder - implement actual user service
    return []


@router.get("/{user_id}", response_model=UserResponse)
@limiter.limit("60/minute")
async def get_user(
    user_id: int,
    current_user: UserResponse = Depends(get_current_user)
) -> UserResponse:
    """Get a specific user by ID."""
    # Users can only view their own profile or admins can view any
    if current_user.id != user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only view own profile"
        )
    
    # This is a placeholder - implement actual user service
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="User not found"
    )


@router.put("/{user_id}", response_model=UserResponse)
@limiter.limit("30/minute")
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    current_user: UserResponse = Depends(get_current_user)
) -> UserResponse:
    """Update a user profile."""
    # Users can only update their own profile or admins can update any
    if current_user.id != user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only update own profile"
        )
    
    # This is a placeholder - implement actual user service
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="User not found"
    )


@router.delete("/{user_id}")
@limiter.limit("10/minute")
async def delete_user(
    user_id: int,
    current_admin: UserResponse = Depends(get_current_admin_user)
) -> dict:
    """Delete a user (admin only)."""
    # This is a placeholder - implement actual user service
    return {"message": "User deleted successfully"}


@router.get("/{user_id}/preferences")
@limiter.limit("60/minute")
async def get_user_preferences(
    user_id: int,
    current_user: UserResponse = Depends(get_current_user)
) -> UserPreferences:
    """Get user gaming preferences."""
    # Users can only view their own preferences or admins can view any
    if current_user.id != user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only view own preferences"
        )
    
    # This is a placeholder - implement actual user service
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="User preferences not found"
    )


@router.put("/{user_id}/preferences")
@limiter.limit("30/minute")
async def update_user_preferences(
    user_id: int,
    preferences: UserPreferences,
    current_user: UserResponse = Depends(get_current_user)
) -> UserPreferences:
    """Update user gaming preferences."""
    # Users can only update their own preferences
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only update own preferences"
        )
    
    # This is a placeholder - implement actual user service
    return preferences


@router.get("/{user_id}/stats")
@limiter.limit("60/minute")
async def get_user_stats(
    user_id: int,
    current_user: UserResponse = Depends(get_current_user)
) -> dict:
    """Get user statistics and activity."""
    # Users can only view their own stats or admins can view any
    if current_user.id != user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only view own statistics"
        )
    
    # This is a placeholder - implement actual user service
    return {
        "total_games_played": 0,
        "total_playtime_hours": 0,
        "favorite_genres": [],
        "average_rating": 0.0,
        "last_active": None
    } 