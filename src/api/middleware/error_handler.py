"""Error handling middleware for the Steam Game Recommender API."""

import logging
from typing import Dict, Any
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.core.exceptions import SteamRecommenderException
from src.core.constants import (
    ERROR_INTERNAL_SERVER,
    ERROR_VALIDATION_FAILED,
    ERROR_RESOURCE_NOT_FOUND
)

logger = logging.getLogger(__name__)


def add_error_handlers(app: FastAPI) -> None:
    """Add error handlers to the FastAPI application."""
    
    @app.exception_handler(SteamRecommenderException)
    async def steam_recommender_exception_handler(
        request: Request, 
        exc: SteamRecommenderException
    ) -> JSONResponse:
        """Handle custom Steam Recommender exceptions."""
        logger.error(
            f"Steam Recommender Exception: {exc.message}",
            extra={
                "error_code": exc.error_code,
                "details": exc.details,
                "status_code": exc.status_code,
                "path": request.url.path,
                "method": request.method
            }
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "details": exc.details,
                    "path": request.url.path,
                    "timestamp": exc.created_at.isoformat() if hasattr(exc, 'created_at') else None
                }
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, 
        exc: RequestValidationError
    ) -> JSONResponse:
        """Handle request validation errors."""
        logger.warning(
            f"Validation Error: {exc.errors()}",
            extra={
                "path": request.url.path,
                "method": request.method,
                "errors": exc.errors()
            }
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": ERROR_VALIDATION_FAILED,
                    "details": {
                        "validation_errors": exc.errors(),
                        "path": request.url.path
                    }
                }
            }
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, 
        exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle HTTP exceptions."""
        logger.warning(
            f"HTTP Exception: {exc.detail}",
            extra={
                "status_code": exc.status_code,
                "path": request.url.path,
                "method": request.method
            }
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": exc.detail,
                    "path": request.url.path
                }
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request, 
        exc: Exception
    ) -> JSONResponse:
        """Handle general exceptions."""
        logger.error(
            f"Unhandled Exception: {str(exc)}",
            extra={
                "exception_type": type(exc).__name__,
                "path": request.url.path,
                "method": request.method
            },
            exc_info=True
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": ERROR_INTERNAL_SERVER,
                    "path": request.url.path
                }
            }
        )


def create_error_response(
    error_code: str,
    message: str,
    status_code: int = 500,
    details: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Create a standardized error response."""
    error_response = {
        "error": {
            "code": error_code,
            "message": message,
            "status_code": status_code
        }
    }
    
    if details:
        error_response["error"]["details"] = details
    
    return error_response


def log_error(
    error: Exception,
    request: Request,
    context: Dict[str, Any] = None
) -> None:
    """Log error with context information."""
    log_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "path": request.url.path,
        "method": request.method,
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
    }
    
    if context:
        log_data.update(context)
    
    logger.error("API Error", extra=log_data) 