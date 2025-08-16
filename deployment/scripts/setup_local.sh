#!/bin/bash

echo "🚀 Setting up Steam Game Recommender System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo -e "${RED}Error: Python 3.8+ is required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $python_version detected${NC}"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data/{models,processed,cache,chroma}
mkdir -p logs
mkdir -p frontend/build

# Copy environment template
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo -e "${YELLOW}Please update .env with your configuration${NC}"
fi

# Initialize database
echo "Initializing database..."
python -c "
from src.infrastructure.database.connection import engine, Base
Base.metadata.create_all(bind=engine)
print('Database initialized successfully')
"

# Download sample data if not exists
if [ ! -f "data/processed/interaction_matrix.csv" ]; then
    echo "Downloading sample data..."
    python deployment/scripts/download_sample_data.py
fi

# Train initial models
echo "Training initial models..."
python deployment/scripts/train_initial_models.py

echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "To start the application:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Start the server: python -m src.main"
echo "  3. Open browser: http://localhost:8000"
echo ""
echo "To use Docker instead:"
echo "  docker-compose -f deployment/docker/docker-compose.yml up" 