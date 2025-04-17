# Steam Game Recommender System

A comprehensive recommendation system for Steam games utilizing collaborative filtering and matrix factorization techniques.

## Overview

This project builds a recommendation system for Steam games using user interaction data from the Steam platform. By analyzing user playtime patterns and game metadata, the system can suggest new games that users might enjoy based on their gaming history.

Key features:
- Data preprocessing for Steam game interaction data
- Baseline recommendation models using cosine similarity
- Advanced recommendation models using Singular Value Decomposition (SVD)
- Comprehensive evaluation metrics including precision@k and hit rate
- Interactive Jupyter notebooks for exploration and analysis

## Dataset

The project uses the Steam Video Game and Bundle Data from Professor Julian McAuley's research repository. The dataset includes:

- 7.79 million reviews from over 2.5 million users
- Information about more than 15,000 games
- 615 game bundles

To download the dataset, run:

```bash
python scripts/download_data.py
```

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

```bash
# Clone the repository
git clone https://github.com/yourusername/steam-game-recommender.git
cd steam-game-recommender

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

1. **Download the data**:
   ```bash
   python scripts/download_data.py
   ```

2. **Explore and preprocess the data**:
   Run the Jupyter notebooks in the `notebooks/` directory:
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   jupyter notebook notebooks/02_preprocessing.ipynb
   ```

3. **Train and evaluate models**:
   ```bash
   jupyter notebook notebooks/03_baseline_model.ipynb
   jupyter notebook notebooks/04_advanced_models.ipynb
   ```
   
   Or use the command-line script:
   ```bash
   python scripts/train_model.py --model svd --n_factors 50
   ```

4. **Run the web demo**:
   ```bash
   pip install streamlit
   streamlit run scripts/web_demo.py
   ```

## Models

### Baseline Models

- **User-Based Collaborative Filtering**: Recommends games based on similar users' preferences.
- **Item-Based Collaborative Filtering**: Recommends games similar to those the user has already played.

### Advanced Models

- **Singular Value Decomposition (SVD)**: Decomposes the user-item interaction matrix into latent factors to uncover hidden patterns and relationships.
- **Hybrid Models**: Combines multiple recommendation approaches to leverage the strengths of each:
  - **Content-Based Hybrid Model**: Integrates collaborative filtering with content-based features from game metadata (genres).

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

## Results

According to our evaluation, the SVD model significantly outperforms the baseline models:

- **Baseline (Cosine Similarity)**:
  - User-based CF: ~0.3% precision@k
  - Item-based CF: ~8% precision@k

- **SVD Model**:
  - Precision@k: ~26%
  - Hit Rate: ~89%

The SVD model's superior performance can be attributed to its ability to handle sparsity, capture latent factors, and reduce noise in the data.

## Future Work

- Implement hybrid models combining collaborative filtering with content-based features
- Explore deep learning approaches like Neural Collaborative Filtering
- Incorporate temporal dynamics to capture evolving user preferences
- Extend the system to recommend game bundles based on user preferences

## References

If you use this code or dataset, please cite the following papers:

- **Self-attentive sequential recommendation** Wang-Cheng Kang, Julian McAuley *ICDM*, 2018
- **Item recommendation on monotonic behavior chains** Mengting Wan, Julian McAuley *RecSys*, 2018
- **Generating and personalizing bundle recommendations on Steam** Apurva Pathak, Kshitiz Gupta, Julian McAuley *SIGIR*, 2017

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/) for providing the Steam dataset
