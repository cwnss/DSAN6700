"""
Integration test for the recommendation engine

This script verifies that the recommendation engine works correctly
with the CV team's embeddings.
"""

import numpy as np
from recommendation_engine import FashionRecommendationEngine


def test_basic_functionality():
    """Test basic recommendation functionality."""
    print("Testing basic functionality...")
    
    # Load data
    embeddings = np.load('../ComputerVisionFiles/fashion_mnist_resnet50_embeddings.npy')
    labels = np.load('../ComputerVisionFiles/fashion_mnist_labels.npy')
    
    with open('../ComputerVisionFiles/fashion_mnist_classes.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Create engine
    engine = FashionRecommendationEngine(
        embeddings=embeddings,
        labels=labels,
        class_names=class_names,
        n_components=128,
        use_pca=True,
        index_type="L2"
    )
    
    # Build index
    engine.build_index()
    
    # Test search
    query_idx = 0
    query_embedding = embeddings[query_idx]
    
    recommendations = engine.recommend(query_embedding, k=5)
    
    assert len(recommendations) == 5, "Should return 5 recommendations"
    assert recommendations[0]['rank'] == 1, "First recommendation should have rank 1"
    assert recommendations[0]['index'] == query_idx, "Query item should be most similar to itself"
    
    print("✓ Basic functionality test passed")
    return True


def test_filtering():
    """Test filtering functionality."""
    print("Testing filtering...")
    
    embeddings = np.load('../ComputerVisionFiles/fashion_mnist_resnet50_embeddings.npy')
    labels = np.load('../ComputerVisionFiles/fashion_mnist_labels.npy')
    
    with open('../ComputerVisionFiles/fashion_mnist_classes.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    engine = FashionRecommendationEngine(
        embeddings=embeddings,
        labels=labels,
        class_names=class_names,
        n_components=128
    )
    engine.build_index()
    
    query_embedding = embeddings[0]
    
    # Test class filtering
    recommendations = engine.recommend(
        query_embedding,
        k=10,
        filter_by_class=[0, 1]
    )
    
    for rec in recommendations:
        assert rec['label'] in [0, 1], f"All results should be from classes 0 or 1, got {rec['label']}"
    
    # Test exclusion
    recommendations = engine.recommend(
        query_embedding,
        k=10,
        exclude_indices=[0, 1, 2]
    )
    
    for rec in recommendations:
        assert rec['index'] not in [0, 1, 2], f"Excluded indices should not appear in results"
    
    print("✓ Filtering test passed")
    return True


def test_save_load():
    """Test save and load functionality."""
    print("Testing save/load...")
    
    embeddings = np.load('../ComputerVisionFiles/fashion_mnist_resnet50_embeddings.npy')
    labels = np.load('../ComputerVisionFiles/fashion_mnist_labels.npy')
    
    with open('../ComputerVisionFiles/fashion_mnist_classes.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Create and save
    engine1 = FashionRecommendationEngine(
        embeddings=embeddings,
        labels=labels,
        class_names=class_names,
        n_components=128
    )
    engine1.build_index()
    engine1.save('test_engine.pkl')
    
    # Load
    engine2 = FashionRecommendationEngine.load('test_engine.pkl')
    
    # Test that loaded engine works
    query_embedding = embeddings[0]
    rec1 = engine1.recommend(query_embedding, k=5)
    rec2 = engine2.recommend(query_embedding, k=5)
    
    # Results should be the same
    assert len(rec1) == len(rec2), "Loaded engine should return same number of results"
    for r1, r2 in zip(rec1, rec2):
        assert r1['index'] == r2['index'], "Results should match"
    
    print("✓ Save/load test passed")
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("Running Integration Tests")
    print("="*60)
    
    try:
        test_basic_functionality()
        test_filtering()
        test_save_load()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()

