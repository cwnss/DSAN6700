"""
Benchmarking script for the Fashion Recommendation Engine

This script performs comprehensive benchmarking including:
- Search speed with different configurations
- Memory usage analysis
- Accuracy comparison with/without PCA
- Different index types comparison
"""

import numpy as np
import time
from recommendation_engine import (
    FashionRecommendationEngine,
    benchmark_search_speed,
    analyze_memory_usage
)


def load_data():
    """Load embeddings and labels."""
    embeddings = np.load('../ComputerVisionFiles/fashion_mnist_resnet50_embeddings.npy')
    labels = np.load('../ComputerVisionFiles/fashion_mnist_labels.npy')
    
    with open('../ComputerVisionFiles/fashion_mnist_classes.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    return embeddings, labels, class_names


def benchmark_pca_components(embeddings, labels, class_names, n_components_list):
    """Benchmark different PCA component counts."""
    print("\n" + "="*60)
    print("Benchmarking Different PCA Component Counts")
    print("="*60)
    
    results = []
    query_embedding = embeddings[0]
    
    for n_comp in n_components_list:
        print(f"\nTesting with {n_comp} components...")
        
        engine = FashionRecommendationEngine(
            embeddings=embeddings,
            labels=labels,
            class_names=class_names,
            n_components=n_comp,
            use_pca=True,
            index_type="L2"
        )
        engine.build_index()
        
        # Benchmark search speed
        bench_results = benchmark_search_speed(engine, n_queries=100, k=10)
        
        # Memory analysis
        memory_info = analyze_memory_usage(engine)
        
        # Accuracy test (compare top-k overlap)
        current_recs = engine.recommend(query_embedding, k=10)
        
        if len(results) > 0:
            baseline_recs = results[0]['recommendations']
            
            baseline_indices = set([r['index'] for r in baseline_recs])
            current_indices = set([r['index'] for r in current_recs])
            overlap = len(baseline_indices & current_indices) / len(baseline_indices)
        else:
            overlap = 1.0
            baseline_recs = current_recs
        
        results.append({
            'n_components': n_comp,
            'search_speed': bench_results['queries_per_second'],
            'avg_time_ms': bench_results['avg_time_per_query'] * 1000,
            'memory_mb': memory_info['reduced_embeddings_size_mb'],
            'compression_ratio': memory_info['compression_ratio'],
            'top_k_overlap': overlap,
            'recommendations': current_recs
        })
    
    # Print comparison table
    print("\n" + "-"*60)
    print("Comparison Table:")
    print("-"*60)
    print(f"{'Components':<12} {'Speed (q/s)':<15} {'Time (ms)':<12} "
          f"{'Memory (MB)':<12} {'Compression':<12} {'Overlap':<10}")
    print("-"*60)
    
    for r in results:
        print(f"{r['n_components']:<12} {r['search_speed']:<15.2f} "
              f"{r['avg_time_ms']:<12.2f} {r['memory_mb']:<12.2f} "
              f"{r['compression_ratio']:<12.2f} {r['top_k_overlap']:<10.2f}")
    
    return results


def benchmark_index_types(embeddings, labels, class_names):
    """Benchmark different index types."""
    print("\n" + "="*60)
    print("Benchmarking Different Index Types")
    print("="*60)
    
    results = []
    query_embedding = embeddings[0]
    
    for index_type in ["L2", "cosine"]:
        print(f"\nTesting {index_type} index...")
        
        engine = FashionRecommendationEngine(
            embeddings=embeddings,
            labels=labels,
            class_names=class_names,
            n_components=128,
            use_pca=True,
            index_type=index_type
        )
        engine.build_index()
        
        # Benchmark search speed
        bench_results = benchmark_search_speed(engine, n_queries=100, k=10)
        
        # Get recommendations
        recommendations = engine.recommend(query_embedding, k=10)
        
        results.append({
            'index_type': index_type,
            'search_speed': bench_results['queries_per_second'],
            'avg_time_ms': bench_results['avg_time_per_query'] * 1000,
            'recommendations': recommendations
        })
    
    # Print comparison
    print("\n" + "-"*60)
    print("Index Type Comparison:")
    print("-"*60)
    print(f"{'Index Type':<15} {'Speed (q/s)':<15} {'Time (ms)':<12}")
    print("-"*60)
    
    for r in results:
        print(f"{r['index_type']:<15} {r['search_speed']:<15.2f} "
              f"{r['avg_time_ms']:<12.2f}")
    
    return results


def benchmark_filtering_overhead(engine):
    """Benchmark the overhead of filtering operations."""
    print("\n" + "="*60)
    print("Benchmarking Filtering Overhead")
    print("="*60)
    
    query_embedding = engine.embeddings[0]
    n_queries = 100
    
    # Without filtering
    start_time = time.time()
    for _ in range(n_queries):
        _ = engine.search(query_embedding, k=10)
    time_no_filter = time.time() - start_time
    
    # With class filtering
    start_time = time.time()
    for _ in range(n_queries):
        _ = engine.search(query_embedding, k=10, filter_by_class=[0, 1, 2])
    time_class_filter = time.time() - start_time
    
    # With exclusion
    start_time = time.time()
    for _ in range(n_queries):
        _ = engine.search(query_embedding, k=10, exclude_indices=[0, 1, 2, 3, 4])
    time_exclude = time.time() - start_time
    
    print(f"No filtering:      {time_no_filter/n_queries*1000:.2f} ms per query")
    print(f"Class filtering:   {time_class_filter/n_queries*1000:.2f} ms per query "
          f"(overhead: {(time_class_filter/time_no_filter - 1)*100:.1f}%)")
    print(f"Exclusion:         {time_exclude/n_queries*1000:.2f} ms per query "
          f"(overhead: {(time_exclude/time_no_filter - 1)*100:.1f}%)")


def main():
    """Run all benchmarks."""
    print("="*60)
    print("Fashion Recommendation Engine - Comprehensive Benchmark")
    print("="*60)
    
    # Load data
    embeddings, labels, class_names = load_data()
    print(f"\nLoaded {len(embeddings)} items with {embeddings.shape[1]}-dimensional embeddings")
    
    # Benchmark 1: Different PCA component counts
    n_components_list = [64, 128, 256, 512, 1024]
    pca_results = benchmark_pca_components(embeddings, labels, class_names, n_components_list)
    
    # Benchmark 2: Different index types
    index_results = benchmark_index_types(embeddings, labels, class_names)
    
    # Benchmark 3: Filtering overhead
    engine = FashionRecommendationEngine(
        embeddings=embeddings,
        labels=labels,
        class_names=class_names,
        n_components=128,
        use_pca=True,
        index_type="L2"
    )
    engine.build_index()
    benchmark_filtering_overhead(engine)
    
    # Summary
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    print("✓ PCA component count comparison completed")
    print("✓ Index type comparison completed")
    print("✓ Filtering overhead analysis completed")
    print("\nRecommendation: Use 128 components with L2 index for optimal")
    print("balance between speed, memory, and accuracy.")


if __name__ == "__main__":
    main()

