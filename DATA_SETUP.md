# Steam Game Recommender - Data Setup

This application is configured to work with your specific Steam data files:

## Data Files

The system uses these three data files located in the `data/` directory:

1. **`data/items_v2.json.gz`** - Game metadata including:
   - Game names, genres, tags
   - Prices, ratings, playtime statistics
   - Developer and publisher information
   - Release years

2. **`data/reviews_v2.json.gz`** - User reviews and interactions:
   - User ratings (1-5 scale)
   - Playtime data (forever and 2-week periods)
   - Review text and helpful/funny votes
   - User-item interaction patterns

3. **`data/bundles.json`** - Game bundle information:
   - Bundle names and pricing
   - Discount information
   - Games included in each bundle

## Quick Start

### Option 1: Full System Startup (Recommended)
```bash
python run_steam_recommender.py
```

This script will:
1. ✅ Check that all data files exist
2. 🔄 Process your data files into the required format
3. 🤖 Build recommendation models from your data
4. 🚀 Start the backend API server
5. 🎨 Start the frontend web interface

### Option 2: Manual Step-by-Step
```bash
# 1. Process the data
python scripts/process_steam_data.py

# 2. Build recommendation models
python scripts/build_recommendation_models.py

# 3. Start the system
python start_system.py
```

## What Happens During Data Processing

1. **Data Loading**: Your JSON files are loaded and parsed
2. **Data Cleaning**: Missing values are handled, data types are normalized
3. **Feature Engineering**: 
   - Genres and tags are converted to one-hot encoded features
   - Playtime data is normalized
   - Interaction strength scores are calculated
4. **User-Item Matrix**: A matrix of user-game interactions is created
5. **Training/Test Split**: Data is split for model training and evaluation

## System Architecture

```
Your Data Files → Data Processing → Model Training → API Server → Web Interface
     ↓                    ↓              ↓           ↓           ↓
items_v2.json.gz   →  Clean Data  →  ML Models → FastAPI → React App
reviews_v2.json.gz →  Feature Eng →  Vectors   →  Port 8000 → Port 3000
bundles.json       →  Matrix      →  Cache     →  Database → User Interface
```

## Data Statistics

Based on your files:
- **Games**: 1,000 unique games with rich metadata
- **Reviews**: 5,000 user reviews and ratings
- **Bundles**: 50 game bundles with pricing information
- **Users**: Multiple users with varied gaming preferences

## API Endpoints

Once running, you can access:

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Recommendation Types

The system provides:

1. **Popularity-based**: Games with high ratings and playtime
2. **Content-based**: Games similar in genre, tags, and features
3. **Collaborative filtering**: Based on user behavior patterns
4. **Bundle recommendations**: Game collections and deals

## Troubleshooting

### Missing Data Files
Ensure all three files exist in the `data/` directory:
```bash
ls -la data/
# Should show: items_v2.json.gz, reviews_v2.json.gz, bundles.json
```

### Data Processing Errors
Check the logs for specific error messages. Common issues:
- Insufficient memory for large datasets
- Corrupted JSON files
- Missing Python dependencies

### Model Building Issues
If models fail to build:
- Ensure processed data exists in `data/processed/`
- Check available memory and disk space
- Verify all required Python packages are installed

## Next Steps

After successful startup:
1. 🎮 Explore the web interface
2. 🔍 Test different recommendation types
3. 📊 Analyze recommendation quality
4. 🚀 Customize models for your specific use case

Your Steam data is now powering a complete recommendation system! 🎉 