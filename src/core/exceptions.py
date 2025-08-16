"""Custom exceptions for the Steam Game Recommender application."""

from typing import Any, Dict, Optional


class SteamRecommenderException(Exception):
    """Base exception for the Steam Game Recommender application."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)


class ValidationError(SteamRecommenderException):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details,
            status_code=422
        )


class AuthenticationError(SteamRecommenderException):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401
        )


class AuthorizationError(SteamRecommenderException):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403
        )


class ResourceNotFoundError(SteamRecommenderException):
    """Raised when a requested resource is not found."""
    
    def __init__(self, resource_type: str, resource_id: str):
        message = f"{resource_type} with id '{resource_id}' not found"
        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            details={"resource_type": resource_type, "resource_id": resource_id},
            status_code=404
        )


class RateLimitError(SteamRecommenderException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, retry_after: Optional[int] = None):
        message = "Rate limit exceeded"
        details = {}
        if retry_after:
            details["retry_after"] = retry_after
            message += f". Please try again in {retry_after} seconds"
        
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details,
            status_code=429
        )


class ExternalAPIError(SteamRecommenderException):
    """Raised when an external API call fails."""
    
    def __init__(self, api_name: str, error_details: Optional[Dict[str, Any]] = None):
        message = f"External API '{api_name}' call failed"
        super().__init__(
            message=message,
            error_code="EXTERNAL_API_ERROR",
            details={"api_name": api_name, **error_details} if error_details else {"api_name": api_name},
            status_code=502
        )


class ModelError(SteamRecommenderException):
    """Raised when ML model operations fail."""
    
    def __init__(self, model_name: str, operation: str, error_details: Optional[Dict[str, Any]] = None):
        message = f"Model '{model_name}' {operation} failed"
        super().__init__(
            message=message,
            error_code="MODEL_ERROR",
            details={
                "model_name": model_name,
                "operation": operation,
                **error_details
            } if error_details else {"model_name": model_name, "operation": operation},
            status_code=500
        )


class DatabaseError(SteamRecommenderException):
    """Raised when database operations fail."""
    
    def __init__(self, operation: str, error_details: Optional[Dict[str, Any]] = None):
        message = f"Database {operation} failed"
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            details={"operation": operation, **error_details} if error_details else {"operation": operation},
            status_code=500
        )


class CacheError(SteamRecommenderException):
    """Raised when cache operations fail."""
    
    def __init__(self, operation: str, error_details: Optional[Dict[str, Any]] = None):
        message = f"Cache {operation} failed"
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            details={"operation": operation, **error_details} if error_details else {"operation": operation},
            status_code=500
        )


class ConfigurationError(SteamRecommenderException):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, config_key: str, message: Optional[str] = None):
        if not message:
            message = f"Configuration error: '{config_key}' is required"
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details={"config_key": config_key},
            status_code=500
        ) 