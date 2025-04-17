# Steam Game Recommender System

A comprehensive recommendation system for Steam games utilizing collaborative filtering and matrix factorization techniques.

![Steam Logo](https://store.steampowered.com/favicon.ico) 

## Overview

This project builds a recommendation system for Steam games using user interaction data from the Steam platform. By analyzing user playtime patterns and game metadata, the system can suggest new games that users might enjoy based on their gaming history.

Key features:
- Data preprocessing for Steam game interaction data
- Baseline recommendation models using cosine similarity
- Advanced recommendation models using Singular Value Decomposition (SVD)
- Hybrid models combining collaborative filtering with content-based features
- Comprehensive evaluation metrics including precision@k and hit rate
- Interactive web demo for exploring recommendations
- Extensive documentation and step-by-step notebooks

## Repository Structure

```
steam-game-recommender/
├── data/                          # Data storage (will be git-ignored)
│   ├── README.md                  # Instructions for data download
│   ├── reviews_v2.json.gz         # Raw review data (downloaded)
│   ├── items_v2.json.gz           # Raw item metadata (downloaded)
│   ├── bundles.json               # Bundle data (downloaded)
│   └── processed/                 # Processed data files
│       ├── train_interactions.csv # Training data
│       ├── test_interactions.csv  # Testing data
│       └── interaction_matrix.csv # User-item interaction matrix
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb  # Initial data analysis
│   ├── 02_preprocessing.ipynb     # Data preprocessing steps
│   ├── 03_baseline_model.ipynb    # Baseline model implementation
│   └── 04_advanced_models.ipynb   # SVD and other models
├── src/                           # Source code
│   ├── __init__.py                # Package initialization
│   ├── data/                      # Data processing modules
│   │   ├── __init__.py
│   │   ├── loader.py              # Data loading utilities
│   │   └── preprocessor.py        # Data preprocessing functions
│   ├── models/                    # Recommendation models
│   │   ├── __init__.py
│   │   ├── base.py                # Base model class
│   │   ├── cosine_similarity.py   # Cosine similarity model
│   │   ├── svd.py                 # SVD model implementation
│   │   └── hybrid.py              # Hybrid model implementation
│   ├── evaluation/                # Evaluation modules
│   │   ├── __init__.py
│   │   └── metrics.py             # Evaluation metrics
│   └── utils/                     # Utility modules
│       ├── __init__.py
│       └── visualization.py       # Visualization utilities
├── scripts/                       # Scripts for various tasks
│   ├── download_data.py           # Script to download data
│   ├── train_model.py             # Script to train models
│   └── web_demo.py                # Streamlit web interface demo
├── tests/                         # Unit tests
│   ├── __init__.py                # Test package initialization
│   ├── test_data.py               # Tests for data processing
│   └── test_models.py             # Tests for recommendation models
├── models/                        # Saved model files (will be git-ignored)
│   ├── user_based_cf.pkl          # Saved user-based model
│   ├── item_based_cf.pkl          # Saved item-based model
│   └── svd_model_50_factors.pkl   # Saved SVD model
├── requirements.txt               # Project dependencies
├── setup.py                       # Package setup
├── .gitignore                     # Git ignore file
└── README.md                      # Project README
```

## Installation

Follow these steps to set up the project environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/steam-game-recommender.git
cd steam-game-recommender

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode (optional)
pip install -e .
```

## Data

The project uses the Steam Video Game and Bundle Data from Professor Julian McAuley's research repository. The dataset includes:

- 7.79 million reviews from over 2.5 million users
- Information about more than 15,000 games
- 615 game bundles

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

## Usage Guide

### Data Exploration and Preprocessing

Start by exploring and preprocessing the data using the provided Jupyter notebooks:

```bash
# Launch Jupyter notebook
jupyter notebook
```

Then navigate to and run the following notebooks in order:

1. **Data Exploration** (`notebooks/01_data_exploration.ipynb`):
   - Examines the structure and characteristics of the Steam dataset
   - Analyzes user behavior and game popularity
   - Visualizes key patterns in the data

2. **Data Preprocessing** (`notebooks/02_preprocessing.ipynb`):
   - Cleans and transforms the raw data
   - Normalizes playtime information
   - Splits data into training and testing sets
   - Creates the user-item interaction matrix
   - Saves preprocessed data to `data/processed/`

### Training Models

You can train recommendation models using either the Jupyter notebooks or the command-line script:

#### Using Jupyter Notebooks:

3. **Baseline Models** (`notebooks/03_baseline_model.ipynb`):
   - Implements and evaluates user-based and item-based collaborative filtering
   - Compares their performance using precision@k and hit rate

4. **Advanced Models** (`notebooks/04_advanced_models.ipynb`):
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

## Models

### Baseline Models

- **User-Based Collaborative Filtering**: Recommends games based on similar users' preferences. Implemented in `src/models/cosine_similarity.py` with `mode="user"`.
- **Item-Based Collaborative Filtering**: Recommends games similar to those the user has already played. Implemented in `src/models/cosine_similarity.py` with `mode="item"`.

### Advanced Models

- **Singular Value Decomposition (SVD)**: Decomposes the user-item interaction matrix into latent factors to uncover hidden patterns and relationships. Implemented in `src/models/svd.py`.
- **Hybrid Models**: Combines multiple recommendation approaches to leverage the strengths of each. Implemented in `src/models/hybrid.py`:
  - **Content-Based Hybrid Model**: Integrates collaborative filtering with content-based features from game metadata (genres).

## Results

According to our evaluation, the SVD model significantly outperforms the baseline models:

- **Baseline (Cosine Similarity)**:
  - User-based CF: ~0.3% precision@k
  - Item-based CF: ~8% precision@k

- **SVD Model**:
  - Precision@k: ~26%
  - Hit Rate: ~89%

The SVD model's superior performance can be attributed to its ability to handle sparsity, capture latent factors, and reduce noise in the data.

## Testing

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

## Future Work

Ideas for extending the project:

- Implement deep learning approaches like Neural Collaborative Filtering
- Incorporate temporal dynamics to capture evolving user preferences
- Extend the system to recommend game bundles based on user preferences
- Add more features from game metadata (tags, descriptions, etc.)
- Implement context-aware recommendations based on gaming sessions
- Develop cold-start handling strategies for new users and games

## References

If you use this code or dataset, please cite the following papers:

- **Self-attentive sequential recommendation** Wang-Cheng Kang, Julian McAuley *ICDM*, 2018
- **Item recommendation on monotonic behavior chains** Mengting Wan, Julian McAuley *RecSys*, 2018
- **Generating and personalizing bundle recommendations on Steam** Apurva Pathak, Kshitiz Gupta, Julian McAuley *SIGIR*, 2017

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/) for providing the Steam dataset
