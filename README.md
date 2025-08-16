# Steam Game Recommender System

![Status](https://img.shields.io/badge/Status-In_Progress-yellow)

A comprehensive Steam game recommendation system that uses collaborative filtering and matrix factorization techniques, built with modern web technologies and production-ready architecture.

![Steam Logo](https://store.steampowered.com/favicon.ico) 

## 🎯 Overview

This project is an extension from a [CSE258](https://cseweb.ucsd.edu/classes/fa24/cse258-b/) project that I built with @wanderman12345 and @aaron-wu1. By analyzing interaction data such as user playtime patterns and game metadata, the system can suggest new games that users might enjoy based on their gaming history.

The system has evolved from a research prototype to a production-ready application with:
- **Multiple ML Models**: Collaborative filtering, content-based, hybrid, and neural approaches
- **FastAPI Backend**: High-performance async API with automatic documentation
- **Authentication & Authorization**: JWT-based user management with role-based access
- **Real-time Recommendations**: Personalized game suggestions based on user preferences
- **Caching & Performance**: Redis caching with in-memory fallback
- **Vector Database**: ChromaDB integration for semantic search
- **Docker Support**: Containerized deployment with Docker Compose
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment
- **Modern Architecture**: Clean architecture with domain-driven design

## 🏗️ Repository Structure

```
steam-game-recommender/
├── .github/workflows/          # GitHub Actions CI/CD
├── data/                       # Data storage (will be git-ignored)
│   ├── README.md               # Instructions for data download
│   ├── reviews_v2.json.gz      # Raw review data (downloaded)
│   ├── items_v2.json.gz        # Raw item metadata (downloaded)
│   ├── bundles.json            # Bundle data (downloaded)
│   └── processed/              # Processed data files
│       ├── train_interactions.csv # Training data
│       ├── test_interactions.csv  # Testing data
│       └── interaction_matrix.csv # User-item interaction matrix
├── src/                        # Main application source
│   ├── api/                    # FastAPI application layer
│   │   ├── routes/             # API endpoints
│   │   ├── middleware/         # CORS, rate limiting, error handling
│   │   └── dependencies.py     # Dependency injection
│   ├── core/                   # Core configuration and constants
│   ├── domain/                 # Business logic and entities
│   │   ├── entities/           # Data models (User, Game, Recommendation)
│   │   ├── repositories/       # Data access interfaces
│   │   └── services/           # Business logic services
│   ├── infrastructure/         # External integrations
│   │   ├── database/           # Database connections and models
│   │   ├── cache/              # Redis and in-memory caching
│   │   ├── ml/                 # ML model management
│   │   └── external/           # External API clients
│   ├── models/                 # ML model implementations
│   │   ├── classical/          # Traditional ML approaches
│   │   ├── neural/             # Neural network models
│   │   └── ensemble/           # Model ensemble methods
│   └── utils/                  # Utility functions and helpers
├── notebooks/                  # Jupyter notebooks for experimentation
│   ├── 01_data_exploration.py  # Initial data analysis
│   ├── 02_preprocessing.py     # Data preprocessing steps
│   ├── 03_baseline_model.py    # Baseline model implementation
│   └── 04_advanced_models.py   # SVD and other models
├── deployment/                  # Deployment configurations
│   ├── docker/                 # Docker and Docker Compose
│   └── scripts/                # Deployment and setup scripts
├── scripts/                     # Scripts for various tasks
│   ├── download_data.py        # Script to download data
│   ├── train_model.py          # Script to train models
│   └── web_demo.py             # Streamlit web interface demo
├── tests/                       # Test suite
├── models/                      # Saved model files (will be git-ignored)
├── requirements.txt             # Project dependencies
├── setup.py                     # Package setup
└── .gitignore                   # Git ignore file
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

## 📊 Data

We used the Steam Video Game and Bundle Data from Professor Julian McAuley's research repository. It is linked at the bottom. The dataset includes 7.79 million reviews from over 2.5 million users as well as information about more than 15,000 games and 615 game bundles.

### Downloading Data

You can download the dataset using the provided script:

```bash
python scripts/download_data.py
```

This script will download the following files to the `data/` directory:
- `reviews_v2.json.gz` (1.3GB): Review data including user IDs, item IDs, and playtime
- `items_v2.json.gz` (2.7MB): Game metadata including titles and genres
- `bundles.json` (92KB): Game bundle information

Alternatively, you can manually download the files from the following URLs:
- Review Data: [https://snap.stanford.edu/data/steam/steam_reviews.json.gz](https://snap.stanford.edu/data/steam/steam_reviews.json.gz)
- Item Metadata: [https://snap.stanford.edu/data/steam/steam_games.json.gz](https://snap.stanford.edu/data/steam/steam_games.json.gz)
- Bundle Data: [https://snap.stanford.edu/data/steam/bundle_data.json.gz](https://snap.stanford.edu/data/steam/bundle_data.json.gz)

### Data Format

Each data file contains JSON objects, one per line. The format of each file is described in detail in `data/README.md`.

## 🔧 Usage Guide

### Data Exploration and Preprocessing

Start by exploring and preprocessing the data using the provided Jupyter notebooks:

```bash
# Launch Jupyter notebook
jupyter notebook
```

Then navigate to and run the following notebooks in order:

1. **Data Exploration** (`notebooks/01_data_exploration.py`):
   - Examines the structure and characteristics of the Steam dataset
   - Analyzes user behavior and game popularity
   - Visualizes key patterns in the data

2. **Data Preprocessing** (`notebooks/02_preprocessing.py`):
   - Cleans and transforms the raw data
   - Normalizes playtime information
   - Splits data into training and testing sets
   - Creates the user-item interaction matrix
   - Saves preprocessed data to `data/processed/`

### Training Models

You can train recommendation models using either the Jupyter notebooks or the command-line script:

#### Using Jupyter Notebooks:

3. **Baseline Models** (`notebooks/03_baseline_model.py`):
   - Implements and evaluates user-based and item-based collaborative filtering
   - Compares their performance using precision@k and hit rate

4. **Advanced Models** (`notebooks/04_advanced_models.py`):
   - Implements and evaluates SVD models
   - Tunes hyperparameters (number of latent factors)
   - Analyzes model performance in detail

#### Using Command-Line Script:

```bash
# Train a user-based collaborative filtering model
python scripts/train_model.py --model user_cf

# Train an item-based collaborative filtering model
python scripts/train_model.py --model item_cf

# Train an SVD model with 50 latent factors
python scripts/train_model.py --model svd --n_factors 50

# Train with a sample of the data for faster iteration
python scripts/train_model.py --model svd --n_factors 20 --sample_size 100000
```

Trained models will be saved to the `models/` directory.

### Generating Recommendations

You can generate recommendations programmatically or using the web demo:

#### Programmatically:

```python
import pickle
import pandas as pd

# Load a trained model
with open('models/svd_model_50_factors.pkl', 'rb') as f:
    model = pickle.load(f)

# Generate recommendations for a user
user_id = 'YOUR_USER_ID'  # Replace with an actual user ID
recommendations = model.recommend(user_id, k=10)

# Display recommendations
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. Item: {rec['item_id']}, Score: {rec['score']:.4f}")
```

### Evaluating Models

To evaluate a model on test data:

```python
import pickle
import pandas as pd
from src.evaluation.metrics import evaluate_model

# Load a trained model
with open('models/svd_model_50_factors.pkl', 'rb') as f:
    model = pickle.load(f)

# Load test data
test_df = pd.read_csv('data/processed/test_interactions.csv')

# Evaluate the model
metrics = evaluate_model(model, test_df, k=10)
print(f"Precision@10: {metrics['precision_at_k']:.4f}")
print(f"Hit Rate: {metrics['hit_rate']:.4f}")
```

### Web Demo

The repository includes an interactive web demo built with Streamlit that allows you to explore the recommendation system visually:

```bash
# Run the web demo
streamlit run scripts/web_demo.py
```

The web demo provides the following features:
- Load actual Steam data or generate sample data
- Visualize playtime and genre distributions
- Select a trained model for recommendations
- Choose a user and generate personalized game recommendations
- View and visualize the recommendations with game details

## 🤖 ML Models

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

### Baseline Models

- **User-Based Collaborative Filtering**: Recommends games based on similar users' preferences. Implemented in `src/models/cosine_similarity.py` with `mode="user"`.
- **Item-Based Collaborative Filtering**: Recommends games similar to those the user has already played. Implemented in `src/models/cosine_similarity.py` with `mode="item"`.

### Advanced Models

- **Singular Value Decomposition (SVD)**: Decomposes the user-item interaction matrix into latent factors to uncover hidden patterns and relationships. Implemented in `src/models/svd.py`.
- **Hybrid Models**: Combines multiple recommendation approaches to leverage the strengths of each. Implemented in `src/models/hybrid.py`:
  - **Content-Based Hybrid Model**: Integrates collaborative filtering with content-based features from game metadata (genres).

## 📈 Results

According to our evaluation, the SVD model significantly outperforms the baseline models:

- **Baseline (Cosine Similarity)**:
  - User-based CF: ~0.3% precision@k
  - Item-based CF: ~8% precision@k

- **SVD Model**:
  - Precision@k: ~26%
  - Hit Rate: ~89%

The SVD model's superior performance can be attributed to its ability to handle sparsity, capture latent factors, and reduce noise in the data.

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

To run the unit tests:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_models.py

# Run with verbose output
python -m pytest -v tests/
```

The tests verify the functionality of data processing modules and recommendation models using small test datasets.

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

## 🔍 Data Sources

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

## 🔮 Future Work

Ideas for extending the project:

- Implement deep learning approaches like Neural Collaborative Filtering
- Incorporate temporal dynamics to capture evolving user preferences
- Extend the system to recommend game bundles based on user preferences
- Add more features from game metadata (tags, descriptions, etc.)
- Implement context-aware recommendations based on gaming sessions
- Develop cold-start handling strategies for new users and games

## 📚 References

If you use this code or dataset, please cite the following papers:

- **Self-attentive sequential recommendation** Wang-Cheng Kang, Julian McAuley *ICDM*, 2018
- **Item recommendation on monotonic behavior chains** Mengting Wan, Julian McAuley *RecSys*, 2018
- **Generating and personalizing bundle recommendations on Steam** Apurva Pathak, Kshitiz Gupta, Julian McAuley *SIGIR*, 2017

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

- [Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/) for providing the Steam dataset
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
