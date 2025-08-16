# src/infrastructure/ml/vector_store.py
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional
from src.core.config import settings
import os

class VectorStore:
    """
    ChromaDB vector store for game embeddings (completely free and local)
    """
    
    def __init__(self):
        # Create directory if it doesn't exist
        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
        
        # Initialize ChromaDB client (persistent local storage)
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=settings.CHROMA_PERSIST_DIR
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_game_embeddings(
        self,
        game_ids: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict]] = None
    ):
        """Add game embeddings to vector store"""
        self.collection.add(
            ids=game_ids,
            embeddings=embeddings,
            metadatas=metadata or [{} for _ in game_ids]
        )
    
    def search_similar_games(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        filter_dict: Optional[Dict] = None
    ) -> Dict:
        """Search for similar games"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict
        )
        
        return {
            'ids': results['ids'][0] if results['ids'] else [],
            'distances': results['distances'][0] if results['distances'] else [],
            'metadata': results['metadatas'][0] if results['metadatas'] else []
        }
    
    def get_game_embedding(self, game_id: str) -> Optional[List[float]]:
        """Get embedding for a specific game"""
        result = self.collection.get(
            ids=[game_id],
            include=['embeddings']
        )
        
        if result['embeddings']:
            return result['embeddings'][0]
        return None
    
    def update_game_embedding(
        self,
        game_id: str,
        embedding: List[float],
        metadata: Optional[Dict] = None
    ):
        """Update game embedding"""
        self.collection.update(
            ids=[game_id],
            embeddings=[embedding],
            metadatas=[metadata] if metadata else None
        ) 