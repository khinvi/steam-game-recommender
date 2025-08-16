# Testing Guide for Steam Game Recommender

This guide explains how to test different components of your Steam game recommender system.

## 🚀 Quick Start Testing

### 1. Run the Quick Test Script
The easiest way to test your system is to run the quick test script:

```bash
python test_quick.py
```

This script will test:
- ✅ Module imports
- ✅ Configuration loading
- ✅ Database creation
- ✅ Model creation and relationships

### 2. Run Individual Tests
You can also run specific test functions:

```bash
python -c "
import sys
sys.path.insert(0, 'src')
from src.core.config import get_settings
print('Config loaded:', get_settings().APP_NAME)
"
```

## 🧪 Comprehensive Testing with pytest

### 1. Install pytest (if not already installed)
```bash
pip install pytest pytest-asyncio
```

### 2. Run all tests
```bash
pytest tests/ -v
```

### 3. Run specific test files
```bash
# Test database functionality
pytest tests/test_database.py -v

# Test basic setup
pytest tests/test_basic_setup.py -v

# Test data processing
pytest tests/test_data.py -v

# Test ML models
pytest tests/test_models.py -v
```

### 4. Run tests with coverage
```bash
pip install pytest-cov
pytest tests/ --cov=src --cov-report=html
```

## 🗄️ Database Testing

### Test Database Connection
```python
from src.infrastructure.database.connection import get_db, engine
from src.infrastructure.database.models import Base

# Create tables
Base.metadata.create_all(bind=engine)

# Test connection
with engine.connect() as conn:
    result = conn.execute("SELECT 1").scalar()
    print(f"Database connection: {'✅' if result == 1 else '❌'}")
```

### Test Model Creation
```python
from src.infrastructure.database.connection import SessionLocal
from src.infrastructure.database.models import User, Game

# Create session
db = SessionLocal()

# Create test data
user = User(username="testuser", email="test@example.com", hashed_password="hash")
game = Game(steam_id="123", name="Test Game")

# Add to database
db.add(user)
db.add(game)
db.commit()

print(f"Created user: {user.username}")
print(f"Created game: {game.name}")

# Clean up
db.delete(user)
db.delete(game)
db.commit()
db.close()
```

## 🔧 Configuration Testing

### Test Settings Loading
```python
from src.core.config import get_settings, settings

# Test configuration
config = get_settings()
print(f"App Name: {config.APP_NAME}")
print(f"Database URL: {config.DATABASE_URL}")
print(f"Debug Mode: {config.DEBUG}")
print(f"Environment: {config.ENVIRONMENT}")
```

## 🌐 API Testing

### Test FastAPI App
```python
from src.api.app import create_app
from fastapi.testclient import TestClient

# Create test client
app = create_app()
client = TestClient(app)

# Test health endpoint
response = client.get("/health")
print(f"Health check: {response.status_code}")
```

## 📊 ML Model Testing

### Test Model Loading
```python
from src.infrastructure.ml.model_manager import get_model_manager

# Test model manager
try:
    model_manager = get_model_manager()
    print("✅ Model manager loaded successfully")
except Exception as e:
    print(f"❌ Model manager failed: {e}")
```

## 🐳 Docker Testing

### Test with Docker Compose
```bash
# Build and run services
cd deployment/docker
docker-compose up --build

# Test API endpoints
curl http://localhost:8000/health
```

## 🔍 Manual Testing

### 1. Test Database Creation
```bash
# Create data directory
mkdir -p data

# Run Python and test database
python3 -c "
import sys
sys.path.insert(0, 'src')
from src.infrastructure.database.connection import engine, Base
from src.infrastructure.database.models import User, Game

# Create tables
Base.metadata.create_all(bind=engine)
print('✅ Database tables created')

# Test connection
with engine.connect() as conn:
    result = conn.execute('SELECT 1').scalar()
    print(f'✅ Database connection: {result}')
"
```

### 2. Test Configuration
```bash
python3 -c "
import sys
sys.path.insert(0, 'src')
from src.core.config import get_settings

settings = get_settings()
print(f'App: {settings.APP_NAME}')
print(f'Database: {settings.DATABASE_URL}')
print(f'Debug: {settings.DEBUG}')
"
```

## 🚨 Common Issues and Solutions

### Import Errors
If you get import errors:
```bash
# Make sure you're in the project root
cd /Users/arnavkhinvasara/steam-game-recommender

# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Database Errors
If database tests fail:
```bash
# Check if data directory exists
ls -la data/

# Remove old database and recreate
rm -f data/app.db
python test_quick.py
```

### Missing Dependencies
If you get missing module errors:
```bash
# Install requirements
pip install -r requirements.txt

# Or install specific packages
pip install sqlalchemy pydantic fastapi
```

## 📝 Test Results Interpretation

### ✅ Successful Tests
- All modules import correctly
- Database tables are created
- Models can be instantiated
- Relationships work properly
- Configuration loads correctly

### ❌ Failed Tests
- Check error messages for specific issues
- Verify all dependencies are installed
- Ensure you're in the correct directory
- Check file permissions for database creation

## 🎯 Next Steps After Testing

1. **Fix any failing tests** - Address import errors, missing dependencies, etc.
2. **Run the full test suite** - Use pytest for comprehensive testing
3. **Test the API** - Start the FastAPI server and test endpoints
4. **Test with real data** - Load some sample Steam data
5. **Test the frontend** - Run the React app and test the UI

## 🆘 Getting Help

If tests continue to fail:
1. Check the error messages carefully
2. Verify your Python environment and dependencies
3. Ensure all files are in the correct locations
4. Check file permissions and paths
5. Review the project structure matches the expected layout

Happy testing! 🎮✨ 