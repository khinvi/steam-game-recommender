"""Authentication service for the Steam Game Recommender application."""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext

from src.core.config import get_settings
from src.core.exceptions import AuthenticationError
from src.domain.entities.user import (
    User,
    UserCreate,
    UserResponse,
    UserToken
)
from src.domain.repositories.base import BaseRepository

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """Service for handling authentication and user management."""
    
    def __init__(self, user_repository: BaseRepository):
        self.user_repository = user_repository
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(
            to_encode,
            settings.secret_key,
            algorithm=settings.algorithm
        )
        return encoded_jwt
    
    def create_refresh_token(self, data: dict) -> str:
        """Create a JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=7)  # 7 days
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(
            to_encode,
            settings.secret_key,
            algorithm=settings.algorithm
        )
        return encoded_jwt
    
    async def authenticate_user(self, email: str, password: str) -> UserToken:
        """Authenticate a user and return access token."""
        # This is a placeholder - you'll need to implement actual user lookup
        # from your database
        user = await self._get_user_by_email(email)
        if not user:
            raise AuthenticationError("Invalid email or password")
        
        if not self.verify_password(password, user.hashed_password):
            raise AuthenticationError("Invalid email or password")
        
        if not user.is_active:
            raise AuthenticationError("User account is disabled")
        
        # Create access token
        access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
        access_token = self.create_access_token(
            data={"sub": str(user.id)},
            expires_delta=access_token_expires
        )
        
        # Create refresh token
        refresh_token = self.create_refresh_token(
            data={"sub": str(user.id)}
        )
        
        return UserToken(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.access_token_expire_minutes * 60,
            refresh_token=refresh_token
        )
    
    async def register_user(self, user_data: UserCreate) -> UserResponse:
        """Register a new user."""
        # Check if user already exists
        existing_user = await self._get_user_by_email(user_data.email)
        if existing_user:
            raise ValueError("User with this email already exists")
        
        existing_username = await self._get_user_by_username(user_data.username)
        if existing_username:
            raise ValueError("Username already taken")
        
        # Hash password
        hashed_password = self.get_password_hash(user_data.password)
        
        # Create user object
        user = User(
            email=user_data.email,
            username=user_data.username,
            hashed_password=hashed_password,
            profile=user_data.display_name
        )
        
        # Save user to database (placeholder)
        # saved_user = await self.user_repository.create(user)
        
        # For now, return a mock user
        return UserResponse(
            id=1,
            email=user.email,
            username=user.username,
            role=user.role,
            is_active=user.is_active,
            is_verified=user.is_verified,
            profile=user.profile,
            preferences=user.preferences,
            created_at=user.created_at,
            updated_at=user.updated_at
        )
    
    async def refresh_token(self, refresh_token: str) -> UserToken:
        """Refresh an access token using a refresh token."""
        try:
            payload = jwt.decode(
                refresh_token,
                settings.secret_key,
                algorithms=[settings.algorithm]
            )
            user_id: str = payload.get("sub")
            token_type: str = payload.get("type")
            
            if user_id is None or token_type != "refresh":
                raise ValueError("Invalid refresh token")
            
            # Verify user exists and is active
            user = await self._get_user_by_id(int(user_id))
            if not user or not user.is_active:
                raise ValueError("User not found or inactive")
            
            # Create new access token
            access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
            access_token = self.create_access_token(
                data={"sub": user_id},
                expires_delta=access_token_expires
            )
            
            # Create new refresh token
            new_refresh_token = self.create_refresh_token(
                data={"sub": user_id}
            )
            
            return UserToken(
                access_token=access_token,
                token_type="bearer",
                expires_in=settings.access_token_expire_minutes * 60,
                refresh_token=new_refresh_token
            )
            
        except JWTError:
            raise ValueError("Invalid refresh token")
    
    async def logout_user(self, user_id: int) -> None:
        """Logout a user (invalidate tokens)."""
        # In a real implementation, you might want to:
        # 1. Add the token to a blacklist
        # 2. Update user's last logout time
        # 3. Log the logout event
        pass
    
    async def verify_email(self, token: str) -> None:
        """Verify a user's email address."""
        # This is a placeholder - implement email verification logic
        pass
    
    async def send_password_reset_email(self, email: str) -> None:
        """Send a password reset email."""
        # This is a placeholder - implement password reset logic
        pass
    
    async def reset_password(self, token: str, new_password: str) -> None:
        """Reset a user's password using a reset token."""
        # This is a placeholder - implement password reset logic
        pass
    
    async def change_password(self, user_id: int, current_password: str, new_password: str) -> None:
        """Change a user's password."""
        # This is a placeholder - implement password change logic
        pass
    
    async def get_user_by_id(self, user_id: int) -> Optional[UserResponse]:
        """Get a user by ID."""
        user = await self._get_user_by_id(user_id)
        if not user:
            return None
        
        return UserResponse(
            id=user.id,
            email=user.email,
            username=user.username,
            role=user.role,
            is_active=user.is_active,
            is_verified=user.is_verified,
            profile=user.profile,
            preferences=user.preferences,
            created_at=user.created_at,
            updated_at=user.updated_at
        )
    
    async def _get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email (placeholder implementation)."""
        # This is a placeholder - implement actual database lookup
        return None
    
    async def _get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username (placeholder implementation)."""
        # This is a placeholder - implement actual database lookup
        return None
    
    async def _get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get a user by ID (placeholder implementation)."""
        # This is a placeholder - implement actual database lookup
        return None 