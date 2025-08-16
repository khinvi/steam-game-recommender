"""Authentication routes for the Steam Game Recommender API."""

from datetime import timedelta
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from slowapi.util import get_remote_address
from slowapi import Limiter

from src.core.config import get_settings
from src.core.constants import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    SECURITY_BCRYPT_ROUNDS,
    SUCCESS_USER_CREATED,
    ERROR_INVALID_CREDENTIALS
)
from src.domain.entities.user import (
    UserCreate,
    UserResponse,
    UserLogin,
    UserToken
)
from src.api.dependencies import get_current_user
from src.domain.services.auth_service import AuthService

# Rate limiter for auth endpoints
limiter = Limiter(key_func=get_remote_address)

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("5/minute")
async def register(
    user_data: UserCreate,
    request: Any,
    auth_service: AuthService = Depends()
) -> UserResponse:
    """Register a new user."""
    try:
        user = await auth_service.register_user(user_data)
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/login", response_model=UserToken)
@limiter.limit("10/minute")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthService = Depends()
) -> UserToken:
    """Authenticate user and return access token."""
    try:
        token = await auth_service.authenticate_user(
            email=form_data.username,
            password=form_data.password
        )
        return token
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_INVALID_CREDENTIALS
        )


@router.post("/refresh", response_model=UserToken)
@limiter.limit("20/minute")
async def refresh_token(
    refresh_token: str,
    auth_service: AuthService = Depends()
) -> UserToken:
    """Refresh access token using refresh token."""
    try:
        token = await auth_service.refresh_token(refresh_token)
        return token
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )


@router.post("/logout")
@limiter.limit("20/minute")
async def logout(
    current_user: UserResponse = Depends(get_current_user),
    auth_service: AuthService = Depends()
) -> dict:
    """Logout user and invalidate tokens."""
    await auth_service.logout_user(current_user.id)
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: UserResponse = Depends(get_current_user)
) -> UserResponse:
    """Get current user information."""
    return current_user


@router.post("/verify-email")
@limiter.limit("5/minute")
async def verify_email(
    token: str,
    auth_service: AuthService = Depends()
) -> dict:
    """Verify user email address."""
    try:
        await auth_service.verify_email(token)
        return {"message": "Email verified successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/forgot-password")
@limiter.limit("3/minute")
async def forgot_password(
    email: str,
    auth_service: AuthService = Depends()
) -> dict:
    """Send password reset email."""
    try:
        await auth_service.send_password_reset_email(email)
        return {"message": "Password reset email sent"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/reset-password")
@limiter.limit("5/minute")
async def reset_password(
    token: str,
    new_password: str,
    auth_service: AuthService = Depends()
) -> dict:
    """Reset user password using reset token."""
    try:
        await auth_service.reset_password(token, new_password)
        return {"message": "Password reset successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/change-password")
@limiter.limit("5/minute")
async def change_password(
    current_password: str,
    new_password: str,
    current_user: UserResponse = Depends(get_current_user),
    auth_service: AuthService = Depends()
) -> dict:
    """Change user password."""
    try:
        await auth_service.change_password(
            current_user.id,
            current_password,
            new_password
        )
        return {"message": "Password changed successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        ) 