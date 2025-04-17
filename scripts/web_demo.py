#!/usr/bin/env python
"""
Web demo for the Steam Game Recommender System.
"""
import os
import sys
import argparse
import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_steam_data, convert_to_dataframes
from src.models.svd import SVDModel
from src.models.cosine_similarity import CosineSimilarityModel
from src.models.hybrid import HybridModel, ContentBasedHybridModel
from src.utils.visualization import (
    plot_recommendations_for_user,
    plot_playtime_distribution,
    plot_genre_distribution
)

def load_model(model_path):
    """Load a trained model from a pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_game_info(item_id, metadata_df):
    """Get information about a game from the metadata."""
    if metadata_df is None or item_id not in metadata_df['item_id'].values:
        return f"Game ID: {item_id}", None
    
    game_data = metadata_df[metadata_df['item_id'] == item_id].iloc[0]
    game_name = game_data.get('item_name', f"Game ID: {item_id}")
    
    # Get genre if available
    genre = game_data.get('genre', None)
    
    return game_name, genre

def main():
    """Main function for the web demo."""
    st.set_page_config(
        page_title="Steam Game Recommender",
        page_icon="🎮",
        layout="wide"
    )
    
    st.title("Steam Game Recommender System")
    st.markdown("""
    This demo showcases the Steam Game Recommender System, which recommends games
    based on user interactions and game metadata. The system uses collaborative filtering
    and matrix factorization techniques to generate personalized recommendations.
    """)
    
    # Sidebar for model selection and settings
    st.sidebar.header("Settings")
    
    # Model selection
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    
    if not os.path.exists(models_dir) or not os.listdir(models_dir):
        st.sidebar.warning("No trained models found. Please train models first.")
        model_files = []
    else:
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    model_name = st.sidebar.selectbox(
        "Select a model",
        model_files,
        index=0 if model_files else None
    )
    
    # Number of recommendations
    k = st.sidebar.slider(
        "Number of recommendations",
        min_value=1,
        max_value=20,
        value=10
    )
    
    # Load data
    st.header("Data")
    
    data_tab1, data_tab2 = st.tabs(["Load from files", "Use sample data"])
    
    with data_tab1:
        data_dir = st.text_input(
            "Data directory",
            value=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        )
        
        if st.button("Load data"):
            with st.spinner("Loading data..."):
                try:
                    raw_data = load_steam_data(data_dir)
                    dfs = convert_to_dataframes(raw_data)
                    
                    if 'reviews' in dfs:
                        reviews_df = dfs['reviews']
                        st.success(f"Loaded {len(reviews_df):,} reviews")
                    else:
                        st.error("No reviews data found")
                        reviews_df = None
                    
                    if 'metadata' in dfs:
                        metadata_df = dfs['metadata']
                        st.success(f"Loaded metadata for {len(metadata_df):,} games")
                    else:
                        st.warning("No metadata found")
                        metadata_df = None
                except Exception as e:
                    st.error(f"Error loading data: {e}")
                    reviews_df = None
                    metadata_df = None
    
    with data_tab2:
        st.markdown("Use sample data for demonstration purposes.")
        
        if st.button("Generate sample data"):
            # Create sample data
            sample_reviews = []
            users = ["U1", "U2", "U3", "U4", "U5"]
            games = ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10"]
            
            import random
            for user in users:
                # Each user plays 3-6 random games
                num_games = random.randint(3, 6)
                user_games = random.sample(games, num_games)
                
                for game in user_games:
                    # Random playtime between 1 and 100 hours (in minutes)
                    playtime = random.randint(60, 6000)
                    
                    sample_reviews.append({
                        'user_id': user,
                        'item_id': game,
                        'playtime_forever': playtime
                    })
            
            reviews_df = pd.DataFrame(sample_reviews)
            st.success(f"Generated {len(reviews_df):,} sample reviews")
            
            # Generate sample metadata
            sample_metadata = []
            game_names = [
                "Action Adventure", "RPG Quest", "Strategy Master",
                "Simulation City", "Racing Pro", "Puzzle Challenge",
                "Sports Champion", "FPS Shooter", "Platformer Classic", "Horror Survival"
            ]
            
            genres = [
                "Action, Adventure", "RPG, Adventure", "Strategy, Simulation",
                "Simulation, Management", "Racing, Sport", "Puzzle, Casual",
                "Sport, Simulation", "Action, FPS", "Platformer, Action", "Horror, Survival"
            ]
            
            for i, game in enumerate(games):
                sample_metadata.append({
                    'item_id': game,
                    'item_name': game_names[i],
                    'genre': genres[i]
                })
            
            metadata_df = pd.DataFrame(sample_metadata)
            st.success(f"Generated metadata for {len(metadata_df):,} games")
    
    # Data visualization
    if 'reviews_df' in locals() and reviews_df is not None:
        st.header("Data Visualization")
        viz_tab1, viz_tab2 = st.tabs(["Playtime Distribution", "Genre Distribution"])
        
        with viz_tab1:
            if 'playtime_forever' in reviews_df.columns:
                fig = plot_playtime_distribution(reviews_df)
                st.pyplot(fig)
            else:
                st.warning("No playtime data available")
        
        with viz_tab2:
            if 'metadata_df' in locals() and metadata_df is not None and 'genre' in metadata_df.columns:
                fig = plot_genre_distribution(metadata_df)
                st.pyplot(fig)
            else:
                st.warning("No genre data available")
    
    # Model loading and recommendation
    st.header("Recommendations")
    
    if model_name and os.path.exists(os.path.join(models_dir, model_name)):
        model_path = os.path.join(models_dir, model_name)
        model = load_model(model_path)
        
        if model is not None:
            st.success(f"Loaded model: {model_name}")
            
            # Get a list of available users
            if 'reviews_df' in locals() and reviews_df is not None:
                available_users = sorted(reviews_df['user_id'].unique())
                
                if available_users:
                    user_id = st.selectbox("Select a user", available_users)
                    
                    if st.button(f"Generate Recommendations for User {user_id}"):
                        try:
                            with st.spinner("Generating recommendations..."):
                                recommendations = model.recommend(user_id, k=k)
                                
                                if recommendations:
                                    st.subheader(f"Top {len(recommendations)} Recommendations:")
                                    
                                    # Create a DataFrame for displaying recommendations
                                    rec_data = []
                                    
                                    for i, rec in enumerate(recommendations, 1):
                                        item_id = rec['item_id']
                                        score = rec['score']
                                        
                                        # Get game information if metadata is available
                                        if 'metadata_df' in locals() and metadata_df is not None:
                                            game_name, genre = get_game_info(item_id, metadata_df)
                                        else:
                                            game_name = f"Game {item_id}"
                                            genre = None
                                        
                                        rec_data.append({
                                            'Rank': i,
                                            'Game ID': item_id,
                                            'Game Name': game_name,
                                            'Genre': genre,
                                            'Score': score
                                        })
                                    
                                    rec_df = pd.DataFrame(rec_data)
                                    st.dataframe(rec_df)
                                    
                                    # Visualize recommendations
                                    if 'metadata_df' in locals() and metadata_df is not None:
                                        fig = plot_recommendations_for_user(user_id, recommendations, metadata_df)
                                        st.pyplot(fig)
                                    
                                    # Show user's playtime history if available
                                    if 'reviews_df' in locals() and reviews_df is not None:
                                        st.subheader(f"User {user_id}'s Playtime History:")
                                        user_history = reviews_df[reviews_df['user_id'] == user_id]
                                        
                                        # Join with metadata if available
                                        if 'metadata_df' in locals() and metadata_df is not None:
                                            user_history = user_history.merge(
                                                metadata_df[['item_id', 'item_name', 'genre']], 
                                                on='item_id', 
                                                how='left'
                                            )
                                        
                                        # Display user history
                                        st.dataframe(user_history)
                                else:
                                    st.warning("No recommendations generated. The user may not have any interactions in the training data.")
                        except Exception as e:
                            st.error(f"Error generating recommendations: {e}")
                else:
                    st.warning("No users available. Please load data first.")
            else:
                st.warning("Please load data first.")
        else:
            st.error("Failed to load model.")
    else:
        st.warning("Please select a trained model first.")
    
    # Footer
    st.markdown("---")
    st.markdown("Steam Game Recommender System | 2025")

if __name__ == "__main__":
    main()