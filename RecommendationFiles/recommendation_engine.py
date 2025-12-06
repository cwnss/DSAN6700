"""
Fashion Recommendation Engine

This module implements a recommendation engine for fashion items using:
- Dimensionality reduction (PCA)
- FAISS for efficient similarity search
- Ranking and filtering logic
"""

import numpy as np
import faiss
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional, Dict, Any
import pickle
import time
import os


class FashionRecommendationEngine:
    """
    A recommendation engine for fashion items using embeddings, dimensionality reduction,
    and FAISS for efficient similarity search.
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        n_components: int = 128,
        use_pca: bool = True,
        index_type: str = "L2"
    ):
        """
        Initialize the recommendation engine.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim) - original embeddings
            labels: Array of shape (n_samples,) - class labels for each item
            class_names: List of class name strings
            n_components: Number of dimensions after PCA reduction
            use_pca: Whether to apply PCA dimensionality reduction
            index_type: Type of FAISS index ("L2" or "cosine")
        """
        self.embeddings = embeddings
        self.labels = labels
        self.class_names = class_names
        self.n_components = n_components
        self.use_pca = use_pca
        self.index_type = index_type
        
        # Components to be initialized
        self.pca = None
        self.scaler = None
        self.reduced_embeddings = None
        self.index = None
        self.item_metadata = None
        
        # Statistics
        self.original_dim = embeddings.shape[1]
        self.n_samples = embeddings.shape[0]
        
    def _apply_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply PCA dimensionality reduction to embeddings.
        
        Args:
            embeddings: Original embeddings
            
        Returns:
            Reduced embeddings
        """
        if not self.use_pca:
            return embeddings
            
        print(f"Applying PCA: {self.original_dim} -> {self.n_components} dimensions...")
        self.pca = PCA(n_components=self.n_components, random_state=42)
        reduced = self.pca.fit_transform(embeddings)
        
        # Calculate explained variance
        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        print(f"Explained variance: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
        
        return reduced
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings for cosine similarity (if needed).
        
        Args:
            embeddings: Embeddings to normalize
            
        Returns:
            Normalized embeddings
        """
        if self.index_type == "cosine":
            # L2 normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            return embeddings / norms
        return embeddings
    
    def build_index(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Build the FAISS index for similarity search.
        
        Args:
            metadata: Optional dictionary with item metadata (e.g., item IDs, categories)
        """
        print("Building recommendation index...")
        start_time = time.time()
        
        # Apply PCA if enabled
        self.reduced_embeddings = self._apply_pca(self.embeddings)
        reduced_dim = self.reduced_embeddings.shape[1]
        
        # Normalize if using cosine similarity
        normalized_embeddings = self._normalize_embeddings(self.reduced_embeddings)
        
        # Convert to float32 for FAISS
        embeddings_f32 = normalized_embeddings.astype('float32')
        
        # Create FAISS index
        if self.index_type == "L2":
            # L2 distance index
            self.index = faiss.IndexFlatL2(reduced_dim)
        elif self.index_type == "cosine":
            # Inner product index (for normalized vectors, inner product = cosine similarity)
            self.index = faiss.IndexFlatIP(reduced_dim)
        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")
        
        # Add embeddings to index
        self.index.add(embeddings_f32)
        
        # Store metadata if provided
        self.item_metadata = metadata
        
        build_time = time.time() - start_time
        print(f"Index built in {build_time:.4f} seconds")
        print(f"Index contains {self.index.ntotal} items")
        print(f"Embedding dimension: {reduced_dim}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_by_class: Optional[List[int]] = None,
        exclude_indices: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar items.
        
        Args:
            query_embedding: Query embedding vector (original dimension)
            k: Number of results to return
            filter_by_class: Optional list of class indices to filter by
            exclude_indices: Optional list of indices to exclude from results
            
        Returns:
            Tuple of (distances, indices) for top-k results
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Transform query embedding
        query_reshaped = query_embedding.reshape(1, -1)
        
        # Apply PCA if used
        if self.use_pca and self.pca is not None:
            query_reduced = self.pca.transform(query_reshaped)
        else:
            query_reduced = query_reshaped
        
        # Normalize if using cosine similarity
        query_normalized = self._normalize_embeddings(query_reduced)
        query_f32 = query_normalized.astype('float32')
        
        # Search
        search_k = k * 10 if filter_by_class or exclude_indices else k
        distances, indices = self.index.search(query_f32, search_k)
        
        # Apply filters
        if filter_by_class is not None or exclude_indices is not None:
            filtered_indices = []
            filtered_distances = []
            
            for i, idx in enumerate(indices[0]):
                if exclude_indices and idx in exclude_indices:
                    continue
                if filter_by_class and self.labels[idx] not in filter_by_class:
                    continue
                filtered_indices.append(idx)
                filtered_distances.append(distances[0][i])
                if len(filtered_indices) >= k:
                    break
            
            return np.array([filtered_distances]), np.array([filtered_indices])
        
        return distances, indices
    
    def recommend(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_by_class: Optional[List[int]] = None,
        exclude_indices: Optional[List[int]] = None,
        return_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations with metadata.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of recommendations
            filter_by_class: Optional list of class indices to filter by
            exclude_indices: Optional list of indices to exclude
            return_metadata: Whether to include metadata in results
            
        Returns:
            List of recommendation dictionaries
        """
        distances, indices = self.search(
            query_embedding, k, filter_by_class, exclude_indices
        )
        
        recommendations = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            rec = {
                'index': int(idx),
                'distance': float(dist),
                'label': int(self.labels[idx]),
                'class_name': self.class_names[self.labels[idx]],
                'rank': i + 1
            }
            
            if return_metadata and self.item_metadata:
                for key, values in self.item_metadata.items():
                    rec[key] = values[idx]
            
            recommendations.append(rec)
        
        return recommendations
    
    def save(self, filepath: str):
        """Save the recommendation engine to disk."""
        save_dict = {
            'reduced_embeddings': self.reduced_embeddings,
            'labels': self.labels,
            'class_names': self.class_names,
            'n_components': self.n_components,
            'use_pca': self.use_pca,
            'index_type': self.index_type,
            'pca': self.pca,
            'item_metadata': self.item_metadata,
            'original_dim': self.original_dim
        }
        
        # Save FAISS index separately
        index_path = filepath.replace('.pkl', '_index.faiss')
        faiss.write_index(self.index, index_path)
        
        # Save other components
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"Saved recommendation engine to {filepath}")
        print(f"Saved FAISS index to {index_path}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load the recommendation engine from disk."""
        index_path = filepath.replace('.pkl', '_index.faiss')
        
        # Load components
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Load FAISS index
        index = faiss.read_index(index_path)
        
        # Reconstruct embeddings (we need original for PCA transform)
        # Note: This requires the original embeddings to be passed
        # For now, we'll use reduced embeddings
        engine = cls.__new__(cls)
        engine.reduced_embeddings = save_dict['reduced_embeddings']
        engine.labels = save_dict['labels']
        engine.class_names = save_dict['class_names']
        engine.n_components = save_dict['n_components']
        engine.use_pca = save_dict['use_pca']
        engine.index_type = save_dict['index_type']
        engine.pca = save_dict['pca']
        engine.item_metadata = save_dict['item_metadata']
        engine.original_dim = save_dict['original_dim']
        engine.index = index
        engine.n_samples = engine.reduced_embeddings.shape[0]
        
        print(f"Loaded recommendation engine from {filepath}")
        return engine


def benchmark_search_speed(
    engine: FashionRecommendationEngine,
    n_queries: int = 100,
    k: int = 10
) -> Dict[str, float]:
    """
    Benchmark search speed of the recommendation engine.
    
    Args:
        engine: The recommendation engine
        n_queries: Number of queries to test
        k: Number of results per query
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"Benchmarking search speed with {n_queries} queries...")
    
    # Generate random query embeddings
    query_dim = engine.original_dim
    queries = np.random.randn(n_queries, query_dim).astype('float32')
    
    # Warm-up
    _ = engine.search(queries[0], k=k)
    
    # Benchmark
    start_time = time.time()
    for query in queries:
        _ = engine.search(query, k=k)
    total_time = time.time() - start_time
    
    avg_time = total_time / n_queries
    queries_per_sec = n_queries / total_time
    
    results = {
        'total_time': total_time,
        'avg_time_per_query': avg_time,
        'queries_per_second': queries_per_sec,
        'n_queries': n_queries,
        'k': k
    }
    
    print(f"Average time per query: {avg_time*1000:.2f} ms")
    print(f"Queries per second: {queries_per_sec:.2f}")
    
    return results


def analyze_memory_usage(engine: FashionRecommendationEngine) -> Dict[str, Any]:
    """
    Analyze memory usage of the recommendation engine.
    
    Args:
        engine: The recommendation engine
        
    Returns:
        Dictionary with memory usage information
    """
    import sys
    
    memory_info = {
        'original_embeddings_size_mb': engine.embeddings.nbytes / (1024**2),
        'reduced_embeddings_size_mb': engine.reduced_embeddings.nbytes / (1024**2) if engine.reduced_embeddings is not None else 0,
        'labels_size_mb': engine.labels.nbytes / (1024**2),
        'n_samples': engine.n_samples,
        'original_dim': engine.original_dim,
        'reduced_dim': engine.reduced_embeddings.shape[1] if engine.reduced_embeddings is not None else engine.original_dim,
        'compression_ratio': engine.original_dim / (engine.reduced_embeddings.shape[1] if engine.reduced_embeddings is not None else engine.original_dim)
    }
    
    if engine.index is not None:
        memory_info['index_size_mb'] = engine.index.ntotal * engine.index.d * 4 / (1024**2)  # float32 = 4 bytes
    
    print("Memory Usage Analysis:")
    print(f"  Original embeddings: {memory_info['original_embeddings_size_mb']:.2f} MB")
    if engine.reduced_embeddings is not None:
        print(f"  Reduced embeddings: {memory_info['reduced_embeddings_size_mb']:.2f} MB")
        print(f"  Compression ratio: {memory_info['compression_ratio']:.2f}x")
    if 'index_size_mb' in memory_info:
        print(f"  FAISS index: {memory_info['index_size_mb']:.2f} MB")
    
    return memory_info

