# Steam Game Recommender

A comprehensive Steam game recommendation system with ML models, FastAPI web API, and modern web technologies.

## 🚀 Features

- **Multiple ML Models**: Collaborative filtering, content-based, hybrid, and neural approaches
- **FastAPI Backend**: High-performance async API with automatic documentation
- **Authentication & Authorization**: JWT-based user management with role-based access
- **Real-time Recommendations**: Personalized game suggestions based on user preferences
- **Caching & Performance**: Redis caching with in-memory fallback
- **Vector Database**: ChromaDB integration for semantic search
- **Docker Support**: Containerized deployment with Docker Compose
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment
- **Modern Architecture**: Clean architecture with domain-driven design

## 🏗️ Project Structure

```
steam-game-recommender/
├── .github/workflows/          # GitHub Actions CI/CD
├── src/                        # Main application source
│   ├── api/                   # FastAPI application layer
│   │   ├── routes/            # API endpoints
│   │   ├── middleware/        # CORS, rate limiting, error handling
│   │   └── dependencies.py    # Dependency injection
│   ├── core/                  # Core configuration and constants
│   ├── domain/                # Business logic and entities
│   │   ├── entities/          # Data models (User, Game, Recommendation)
│   │   ├── repositories/      # Data access interfaces
│   │   └── services/          # Business logic services
│   ├── infrastructure/        # External integrations
│   │   ├── database/          # Database connections and models
│   │   ├── cache/             # Redis and in-memory caching
│   │   ├── ml/                # ML model management
│   │   └── external/          # External API clients
│   ├── models/                # ML model implementations
│   │   ├── classical/         # Traditional ML approaches
│   │   ├── neural/            # Neural network models
│   │   └── ensemble/          # Model ensemble methods
│   └── utils/                 # Utility functions and helpers
├── data/                      # Data storage and models
├── frontend/                  # React/Next.js frontend (planned)
├── deployment/                # Deployment configurations
│   ├── docker/                # Docker and Docker Compose
│   └── scripts/               # Deployment and setup scripts
├── notebooks/                 # Jupyter notebooks for experimentation
└── tests/                     # Test suite
```

## 🛠️ Technology Stack

### Backend
- **Python 3.8+** with async/await support
- **FastAPI** for high-performance web API
- **SQLAlchemy** for database ORM
- **Redis** for caching and session storage
- **ChromaDB** for vector storage and similarity search
- **Pydantic** for data validation and serialization

### ML & Data Science
- **Scikit-learn** for traditional ML algorithms
- **NumPy & Pandas** for data manipulation
- **Matplotlib & Seaborn** for visualization
- **Optional**: PyTorch for neural networks

### Infrastructure
- **Docker** for containerization
- **Docker Compose** for local development
- **GitHub Actions** for CI/CD
- **SQLite** for local development, **PostgreSQL** for production

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker and Docker Compose
- Redis (optional, will fallback to in-memory)

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/steam-game-recommender.git
   cd steam-game-recommender
   ```

2. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run with Docker Compose (recommended)**
   ```bash
   docker-compose -f deployment/docker/docker-compose.yml up --build
   ```

5. **Or run locally**
   ```bash
   python -m src.main
   ```

6. **Access the API**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Docker Development

```bash
# Build and run all services
docker-compose -f deployment/docker/docker-compose.yml up --build

# Run specific services
docker-compose -f deployment/docker/docker-compose.yml up steam-recommender redis

# View logs
docker-compose -f deployment/docker/docker-compose.yml logs -f steam-recommender
```

## 📚 API Documentation

### Authentication Endpoints
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User authentication
- `POST /api/v1/auth/refresh` - Token refresh
- `GET /api/v1/auth/me` - Current user info

### User Management
- `GET /api/v1/users/{user_id}` - Get user profile
- `PUT /api/v1/users/{user_id}` - Update user profile
- `GET /api/v1/users/{user_id}/preferences` - Get user preferences

### Game Management
- `GET /api/v1/games/` - List games with filtering
- `GET /api/v1/games/{game_id}` - Get game details
- `POST /api/v1/games/search` - Search games
- `GET /api/v1/games/categories` - Get game categories

### Recommendations
- `POST /api/v1/recommendations/generate` - Generate recommendations
- `GET /api/v1/recommendations/user/{user_id}` - Get user recommendations
- `POST /api/v1/recommendations/feedback` - Submit feedback

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Application
APP_NAME=Steam Game Recommender
DEBUG=true
ENVIRONMENT=development

# Server
HOST=0.0.0.0
PORT=8000

# Database
DATABASE_URL=sqlite:///./data/steam_recommender.db

# Redis
REDIS_URL=redis://localhost:6379

# Steam API
STEAM_API_KEY=your_steam_api_key_here

# Security
SECRET_KEY=your_secret_key_here
```

### ML Model Configuration

Models are stored in `data/models/` and can be configured via:

```python
from src.infrastructure.ml.model_manager import get_model_manager

model_manager = get_model_manager()
model = model_manager.get_model("collaborative_filtering")
```

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_recommendations.py

# Run async tests
pytest --asyncio-mode=auto
```

### Test Structure
- `tests/` - Test suite
- `tests/conftest.py` - Test configuration and fixtures
- `tests/test_*.py` - Test modules for each component

## 🚀 Deployment

### Production Deployment

1. **Environment Setup**
   ```bash
   export ENVIRONMENT=production
   export DEBUG=false
   export SECRET_KEY=your_secure_secret_key
   ```

2. **Database Migration**
   ```bash
   alembic upgrade head
   ```

3. **Run with Production Server**
   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

### Docker Production

```bash
# Build production image
docker build -f deployment/docker/Dockerfile -t steam-recommender:prod .

# Run production container
docker run -d \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e DATABASE_URL=postgresql://user:pass@host/db \
  steam-recommender:prod
```

### Cloud Deployment

The project includes configuration for:
- **Render.com** - `deployment/render.yaml`
- **Docker Hub** - Automated builds via GitHub Actions
- **Kubernetes** - Ready for K8s deployment

## 🔍 ML Models

### Available Models

1. **Collaborative Filtering**
   - User-based collaborative filtering
   - Item-based collaborative filtering
   - Matrix factorization (SVD)

2. **Content-Based**
   - Genre and tag-based filtering
   - TF-IDF text similarity
   - Feature-based recommendations

3. **Hybrid Approaches**
   - Weighted combination of multiple models
   - Ensemble methods
   - Context-aware recommendations

4. **Neural Networks**
   - Two-tower models for user-item embeddings
   - Deep learning for sequential patterns
   - Transformer-based approaches

### Model Training

```bash
# Train all models
python -m src.scripts.train_models

# Train specific model
python -m src.scripts.train_models --model collaborative_filtering

# Evaluate models
python -m src.scripts.evaluate_models
```

## 📊 Data Sources

### Steam Data
- Game metadata and reviews
- User playtime and ratings
- Genre and tag information
- Price and availability data

### Data Processing
- Automated data collection scripts
- Data cleaning and preprocessing
- Feature engineering pipelines
- Regular data updates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
isort src/ tests/

# Run linting
flake8 src/ tests/
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Steam for providing game data and APIs
- FastAPI community for the excellent web framework
- Scikit-learn team for ML tools
- Open source community for various libraries and tools

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/steam-game-recommender/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/steam-game-recommender/discussions)
- **Documentation**: [API Docs](http://localhost:8000/docs) (when running locally)

---

**Happy Gaming! 🎮**
