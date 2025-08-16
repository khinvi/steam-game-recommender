"""Basic tests to verify project setup and imports."""

import pytest
from pathlib import Path


def test_project_structure():
    """Test that the project structure is set up correctly."""
    # Check that main directories exist
    assert Path("src").exists(), "src directory should exist"
    assert Path("src/api").exists(), "src/api directory should exist"
    assert Path("src/core").exists(), "src/core directory should exist"
    assert Path("src/domain").exists(), "src/domain directory should exist"
    assert Path("src/infrastructure").exists(), "src/infrastructure directory should exist"
    assert Path("src/models").exists(), "src/models directory should exist"
    
    # Check that main files exist
    assert Path("src/main.py").exists(), "src/main.py should exist"
    assert Path("src/api/app.py").exists(), "src/api/app.py should exist"
    assert Path("src/core/config.py").exists(), "src/core/config.py should exist"
    assert Path("requirements.txt").exists(), "requirements.txt should exist"
    assert Path("pyproject.toml").exists(), "pyproject.toml should exist"


def test_core_imports():
    """Test that core modules can be imported."""
    try:
        from src.core.config import get_settings
        from src.core.constants import PROJECT_NAME
        from src.core.exceptions import SteamRecommenderException
        
        # Test that we can get settings
        settings = get_settings()
        assert hasattr(settings, 'APP_NAME')
        assert hasattr(settings, 'DEBUG')
        
        # Test constants
        assert PROJECT_NAME == "Steam Game Recommender"
        
        # Test exceptions
        assert issubclass(SteamRecommenderException, Exception)
        
    except ImportError as e:
        pytest.fail(f"Failed to import core modules: {e}")


def test_domain_imports():
    """Test that domain entities can be imported."""
    try:
        from src.domain.entities.user import User, UserCreate, UserResponse
        from src.domain.entities.game import Game, GameCreate, GameResponse
        from src.domain.entities.recommendation import RecommendationRequest, RecommendationResponse
        
        # Test that entities exist
        assert User is not None
        assert UserCreate is not None
        assert UserResponse is not None
        assert Game is not None
        assert GameCreate is not None
        assert GameResponse is not None
        assert RecommendationRequest is not None
        assert RecommendationResponse is not None
        
    except ImportError as e:
        pytest.fail(f"Failed to import domain entities: {e}")


def test_api_imports():
    """Test that API modules can be imported."""
    try:
        from src.api.app import create_app, api_router
        from src.api.dependencies import get_current_user, get_current_active_user
        
        # Test that API components exist
        assert create_app is not None
        assert api_router is not None
        assert get_current_user is not None
        assert get_current_active_user is not None
        
    except ImportError as e:
        pytest.fail(f"Failed to import API modules: {e}")


def test_infrastructure_imports():
    """Test that infrastructure modules can be imported."""
    try:
        from src.infrastructure.ml.model_manager import get_model_manager
        from src.infrastructure.cache.redis_cache import get_cache
        
        # Test that infrastructure components exist
        assert get_model_manager is not None
        assert get_cache is not None
        
    except ImportError as e:
        pytest.fail(f"Failed to import infrastructure modules: {e}")


def test_configuration():
    """Test that configuration is properly set up."""
    try:
        from src.core.config import get_settings
        
        settings = get_settings()
        
        # Test required settings
        assert hasattr(settings, 'APP_NAME')
        assert hasattr(settings, 'DEBUG')
        assert hasattr(settings, 'API_HOST')
        assert hasattr(settings, 'API_PORT')
        assert hasattr(settings, 'DATABASE_URL')
        assert hasattr(settings, 'SECRET_KEY')
        
        # Test default values
        assert settings.APP_NAME == "Steam Game Recommender"
        assert settings.API_HOST == "0.0.0.0"
        assert settings.API_PORT == 8000
        
    except Exception as e:
        pytest.fail(f"Configuration test failed: {e}")


def test_environment_file():
    """Test that environment configuration file exists."""
    env_file = Path(".env.example")
    assert env_file.exists(), ".env.example should exist"
    
    # Check that it contains required variables
    env_content = env_file.read_text()
    required_vars = [
        "APP_NAME",
        "DEBUG",
        "HOST",
        "PORT",
        "DATABASE_URL",
        "SECRET_KEY"
    ]
    
    for var in required_vars:
        assert var in env_content, f"Environment variable {var} should be defined in .env.example"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"]) 