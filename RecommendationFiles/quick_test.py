#!/usr/bin/env python3
"""
Quick test script to verify the recommendation engine works.
Run this first to make sure everything is set up correctly.
"""

import sys
import os

def check_files():
    """Check if required data files exist."""
    print("1. Checking data files...")
    
    files_required = [
        '../ComputerVisionFiles/fashion_mnist_resnet50_embeddings.npy',
        '../ComputerVisionFiles/fashion_mnist_labels.npy',
        '../ComputerVisionFiles/fashion_mnist_classes.txt'
    ]
    
    missing_files = []
    for file in files_required:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"   Found: {file}")
    
    if missing_files:
        print("\n   Missing files:")
        for file in missing_files:
            print(f"      - {file}")
        print("\n   Solution: Run 'git lfs pull' to download data files")
        return False
    
    return True


def check_packages():
    """Check if required packages are installed."""
    print("\n2. Checking Python packages...")
    
    packages = {
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'faiss': 'faiss-cpu',
        'torch': 'torch'
    }
    
    missing = []
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"   {package} installed")
        except ImportError:
            print(f"   {package} NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\n   Missing packages: {', '.join(missing)}")
        print(f"   Solution: Run 'pip install {' '.join(missing)}'")
        return False
    
    return True


def run_quick_test():
    """Run a quick functionality test."""
    print("\n3. Running quick functionality test...")
    
    try:
        import numpy as np
        from recommendation_engine import FashionRecommendationEngine
        
        # Load data
        print("   Loading embeddings and labels...")
        embeddings = np.load('../ComputerVisionFiles/fashion_mnist_resnet50_embeddings.npy')
        labels = np.load('../ComputerVisionFiles/fashion_mnist_labels.npy')
        
        with open('../ComputerVisionFiles/fashion_mnist_classes.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        
        print(f"   Loaded {len(embeddings)} items with {embeddings.shape[1]}D embeddings")
        
        # Create engine
        print("   Building recommendation engine...")
        engine = FashionRecommendationEngine(
            embeddings=embeddings,
            labels=labels,
            class_names=class_names,
            n_components=128,
            use_pca=True,
            index_type="L2"
        )
        engine.build_index()
        
        # Test search
        print("   Testing similarity search...")
        query_embedding = embeddings[0]
        recommendations = engine.recommend(query_embedding, k=5)
        
        print(f"   Got {len(recommendations)} recommendations")
        print(f"   Top result: Item {recommendations[0]['index']}, "
              f"Class: {recommendations[0]['class_name']}, "
              f"Distance: {recommendations[0]['distance']:.4f}")
        
        # Memory check
        from recommendation_engine import analyze_memory_usage
        memory_info = analyze_memory_usage(engine)
        
        print("\n   Memory usage:")
        print(f"   - Original: {memory_info['original_embeddings_size_mb']:.2f} MB")
        print(f"   - Reduced: {memory_info['reduced_embeddings_size_mb']:.2f} MB")
        print(f"   - Compression: {memory_info['compression_ratio']:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"\n   Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks."""
    print("="*60)
    print("Fashion Recommendation Engine - Quick Test")
    print("="*60)
    print()
    
    # Check files
    if not check_files():
        print("\nFile check failed. Please fix issues above.")
        sys.exit(1)
    
    # Check packages
    if not check_packages():
        print("\nPackage check failed. Please install missing packages.")
        sys.exit(1)
    
    # Run test
    if not run_quick_test():
        print("\nFunctionality test failed.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("All checks passed! The recommendation engine works!")
    print("="*60)
    print("\nNext steps:")
    print("  - Run 'python test_integration.py' for full integration tests")
    print("  - Run 'python example_usage.py' to see detailed examples")
    print("  - Run 'python benchmark.py' for performance analysis")
    print()


if __name__ == "__main__":
    main()

