"""
Example usage of the Fashion Recommendation Engine

This script demonstrates how to:
1. Load embeddings and labels
2. Build the recommendation engine
3. Perform similarity searches
4. Get recommendations with filtering
5. Benchmark performance
"""

import numpy as np
from recommendation_engine import (
    FashionRecommendationEngine,
    benchmark_search_speed,
    analyze_memory_usage
)


def load_data():
    """Load embeddings and labels from the CV team's output files."""
    print("Loading embeddings and labels...")
    
    # Load embeddings
    embeddings = np.load('../ComputerVisionFiles/fashion_mnist_resnet50_embeddings.npy')
    print(f"Loaded embeddings: {embeddings.shape}")
    
    # Load labels
    labels = np.load('../ComputerVisionFiles/fashion_mnist_labels.npy')
    print(f"Loaded labels: {labels.shape}")
    
    # Load class names
    with open('../ComputerVisionFiles/fashion_mnist_classes.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(class_names)} class names: {class_names}")
    
    return embeddings, labels, class_names


def main():
    """Main example usage."""
    # Load data
    embeddings, labels, class_names = load_data()
    
    # Initialize recommendation engine
    print("\n" + "="*60)
    print("Initializing Recommendation Engine")
    print("="*60)
    
    engine = FashionRecommendationEngine(
        embeddings=embeddings,
        labels=labels,
        class_names=class_names,
        n_components=128,  # Reduce from 2048 to 128 dimensions
        use_pca=True,
        index_type="L2"  # Use L2 distance
    )
    
    # Build index
    print("\n" + "="*60)
    print("Building FAISS Index")
    print("="*60)
    engine.build_index()
    
    # Memory analysis
    print("\n" + "="*60)
    print("Memory Usage Analysis")
    print("="*60)
    memory_info = analyze_memory_usage(engine)
    
    # Example 1: Basic similarity search
    print("\n" + "="*60)
    print("Example 1: Basic Similarity Search")
    print("="*60)
    
    # Use a random item as query
    query_idx = 0
    query_embedding = embeddings[query_idx]
    query_label = labels[query_idx]
    query_class = class_names[query_label]
    
    print(f"Query item: Index {query_idx}, Class: {query_class}")
    
    recommendations = engine.recommend(
        query_embedding=query_embedding,
        k=10
    )
    
    print(f"\nTop {len(recommendations)} recommendations:")
    for rec in recommendations:
        print(f"  Rank {rec['rank']}: Index {rec['index']}, "
              f"Class: {rec['class_name']}, Distance: {rec['distance']:.4f}")
    
    # Example 2: Filter by class
    print("\n" + "="*60)
    print("Example 2: Filter by Class")
    print("="*60)
    
    # Find items similar to query but from different classes
    target_class_idx = (query_label + 1) % len(class_names)
    target_class = class_names[target_class_idx]
    
    print(f"Query item: Index {query_idx}, Class: {query_class}")
    print(f"Filtering for class: {target_class}")
    
    recommendations = engine.recommend(
        query_embedding=query_embedding,
        k=5,
        filter_by_class=[target_class_idx]
    )
    
    print(f"\nTop {len(recommendations)} recommendations from {target_class}:")
    for rec in recommendations:
        print(f"  Rank {rec['rank']}: Index {rec['index']}, "
              f"Class: {rec['class_name']}, Distance: {rec['distance']:.4f}")
    
    # Example 3: Exclude certain items
    print("\n" + "="*60)
    print("Example 3: Exclude Items")
    print("="*60)
    
    # Exclude the query item itself and a few others
    exclude_indices = [query_idx, 1, 2, 3]
    
    recommendations = engine.recommend(
        query_embedding=query_embedding,
        k=10,
        exclude_indices=exclude_indices
    )
    
    print(f"Query item: Index {query_idx}, Class: {query_class}")
    print(f"Excluded indices: {exclude_indices}")
    print(f"\nTop {len(recommendations)} recommendations (excluding specified items):")
    for rec in recommendations:
        print(f"  Rank {rec['rank']}: Index {rec['index']}, "
              f"Class: {rec['class_name']}, Distance: {rec['distance']:.4f}")
    
    # Example 4: Benchmark search speed
    print("\n" + "="*60)
    print("Example 4: Performance Benchmark")
    print("="*60)
    
    benchmark_results = benchmark_search_speed(
        engine=engine,
        n_queries=100,
        k=10
    )
    
    # Example 5: Save and load
    print("\n" + "="*60)
    print("Example 5: Save and Load")
    print("="*60)
    
    save_path = 'recommendation_engine.pkl'
    print(f"Saving engine to {save_path}...")
    engine.save(save_path)
    
    print(f"\nLoading engine from {save_path}...")
    loaded_engine = FashionRecommendationEngine.load(save_path)
    
    # Test loaded engine
    test_recommendations = loaded_engine.recommend(
        query_embedding=query_embedding,
        k=5
    )
    print(f"\nTest search with loaded engine returned {len(test_recommendations)} results")
    
    print("\n" + "="*60)
    print("Example Usage Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

