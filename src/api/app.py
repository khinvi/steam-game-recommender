"""FastAPI application configuration and router setup."""

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.core.config import get_settings
from src.api.routes import auth, recommendations, users, games
from src.api.middleware.error_handler import add_error_handlers


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="A comprehensive Steam game recommendation system with ML models and web API",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=settings.allowed_methods,
        allow_headers=settings.allowed_headers,
    )
    
    # Add rate limiting
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # Add error handlers
    add_error_handlers(app)
    
    # Include API routes
    app.include_router(api_router, prefix="/api/v1")
    
    return app


# Create the main API router
api_router = APIRouter()

# Include route modules
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(users.router, prefix="/users", tags=["Users"])
api_router.include_router(games.router, prefix="/games", tags=["Games"])
api_router.include_router(recommendations.router, prefix="/recommendations", tags=["Recommendations"])


@api_router.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Steam Game Recommender API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    } 