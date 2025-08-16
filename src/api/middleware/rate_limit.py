"""Rate limiting middleware for the Steam Game Recommender API."""

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from src.core.config import get_settings


def create_limiter() -> Limiter:
    """Create and configure the rate limiter."""
    settings = get_settings()
    
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[
            f"{settings.rate_limit_per_minute}/minute",
            f"{settings.rate_limit_per_hour}/hour"
        ]
    )
    
    return limiter


def add_rate_limit_exception_handler(app, limiter: Limiter) -> None:
    """Add rate limit exception handler to the FastAPI application."""
    
    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        """Handle rate limit exceeded exceptions."""
        retry_after = exc.retry_after
        
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "Rate limit exceeded",
                    "details": {
                        "retry_after": retry_after,
                        "limit": exc.limit,
                        "reset": exc.reset
                    }
                }
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(exc.limit),
                "X-RateLimit-Remaining": str(exc.remaining),
                "X-RateLimit-Reset": str(exc.reset)
            }
        )


def get_rate_limit_config() -> dict:
    """Get rate limiting configuration."""
    settings = get_settings()
    
    return {
        "default_limits": [
            f"{settings.rate_limit_per_minute}/minute",
            f"{settings.rate_limit_per_hour}/hour"
        ],
        "storage_uri": "memory://",  # Use in-memory storage for simplicity
        "strategy": "fixed-window",  # Fixed window rate limiting strategy
    }


# Rate limit decorators for specific endpoints
def rate_limit_by_user(request: Request) -> str:
    """Rate limit key function based on user ID."""
    # Extract user ID from request (e.g., from JWT token)
    # For now, fall back to IP address
    return get_remote_address(request)


def rate_limit_by_endpoint(request: Request) -> str:
    """Rate limit key function based on endpoint."""
    return f"{request.method}:{request.url.path}"


def rate_limit_by_user_and_endpoint(request: Request) -> str:
    """Rate limit key function based on user ID and endpoint."""
    # Extract user ID from request (e.g., from JWT token)
    # For now, fall back to IP address + endpoint
    return f"{get_remote_address(request)}:{request.method}:{request.url.path}" 