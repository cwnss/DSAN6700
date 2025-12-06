# Fashion Recommendation Engine

A comprehensive recommendation engine for fashion items using computer vision embeddings, dimensionality reduction, and efficient similarity search. This module implements similarity search, dimensionality reduction (PCA), and recommendation logic with FAISS indexing.

## Features

- **Dimensionality Reduction**: PCA-based reduction from 2048D to configurable dimensions (default: 128D, ~95% variance retained)
- **FAISS Integration**: High-performance similarity search using FAISS (L2 or cosine similarity)
- **Flexible Filtering**: Filter by class, exclude specific items, combine multiple filters
- **Metadata Support**: Optional item metadata integration for advanced filtering
- **Performance Optimized**: Benchmarked for speed (~10,000+ queries/second) and memory efficiency
- **Save/Load**: Persist trained models for production use

## Installation

```bash
pip install -r requirements.txt
```

- `fashion_mnist_labels.npy` (stored with Git LFS): Array of label indices for each embedded image.

## Quick Start

### Step 1: Ensure Data Files Are Available

The embedding files are stored with Git LFS. If you haven't pulled them yet:

```bash
cd ..  # Go to project root
git lfs install
git lfs pull
cd RecommendationFiles  # Return to this directory
```

### Step 2: Quick Test

Run the quick test script to verify everything works:

```bash
python quick_test.py
```

This will check:
- ✅ All data files exist
- ✅ Required packages are installed
- ✅ Basic functionality works
- ✅ Memory usage statistics

### Step 3: Basic Usage

```python
import numpy as np
from recommendation_engine import FashionRecommendationEngine

# Load embeddings and labels from CV team's output
embeddings = np.load('../ComputerVisionFiles/fashion_mnist_resnet50_embeddings.npy')
labels = np.load('../ComputerVisionFiles/fashion_mnist_labels.npy')

with open('../ComputerVisionFiles/fashion_mnist_classes.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Initialize engine
engine = FashionRecommendationEngine(
    embeddings=embeddings,
    labels=labels,
    class_names=class_names,
    n_components=128,  # Reduce to 128 dimensions
    use_pca=True,
    index_type="L2"   # or "cosine"
)

# Build the index
engine.build_index()

# Get recommendations
query_embedding = embeddings[0]  # Your query item
recommendations = engine.recommend(query_embedding, k=10)

for rec in recommendations:
    print(f"Item {rec['index']}: {rec['class_name']} (distance: {rec['distance']:.4f})")
```

## Usage Examples

### Basic Similarity Search

```python
recommendations = engine.recommend(
    query_embedding=query_embedding,
    k=10
)
```

### Filter by Class

```python
# Find similar items but only from specific classes
recommendations = engine.recommend(
    query_embedding=query_embedding,
    k=10,
    filter_by_class=[0, 1, 2]  # Only these class indices
)
```

### Exclude Items

```python
# Exclude specific items from results
recommendations = engine.recommend(
    query_embedding=query_embedding,
    k=10,
    exclude_indices=[0, 1, 2]  # Exclude these item indices
)
```

### Combined Filtering

```python
recommendations = engine.recommend(
    query_embedding=query_embedding,
    k=10,
    filter_by_class=[0, 1],
    exclude_indices=[5, 6]
)
```

## Running Scripts

### Integration Tests

```bash
python test_integration.py
```

Expected output: "All tests passed! ✓"

### Example Usage

```bash
python example_usage.py
```

This shows:
- Loading embeddings
- Building the index
- Memory usage analysis
- Example searches with filtering
- Save/load functionality

### Performance Benchmarking

```bash
python benchmark.py
```

This tests:
- Different PCA component counts (64, 128, 256, 512, 1024)
- Different index types (L2 vs cosine)
- Filtering overhead
- Memory usage

### Automated Test Script

```bash
chmod +x run_tests.sh
./run_tests.sh
```

Or on Windows:
```bash
bash run_tests.sh
```

## API Reference

### FashionRecommendationEngine

#### `__init__(embeddings, labels, class_names, n_components=128, use_pca=True, index_type="L2")`

Initialize the recommendation engine.

**Parameters:**
- `embeddings` (np.ndarray): Original embeddings, shape (n_samples, embedding_dim)
- `labels` (np.ndarray): Class labels for each item, shape (n_samples,)
- `class_names` (List[str]): List of class name strings
- `n_components` (int): Number of dimensions after PCA (default: 128)
- `use_pca` (bool): Whether to apply PCA (default: True)
- `index_type` (str): FAISS index type - "L2" or "cosine" (default: "L2")

#### `build_index(metadata=None)`

Build the FAISS index for similarity search.

**Parameters:**
- `metadata` (Optional[Dict]): Optional item metadata dictionary

#### `recommend(query_embedding, k=10, filter_by_class=None, exclude_indices=None, return_metadata=False)`

Get recommendations with metadata.

**Parameters:**
- `query_embedding` (np.ndarray): Query embedding vector (original dimension)
- `k` (int): Number of recommendations (default: 10)
- `filter_by_class` (Optional[List[int]]): Filter by class indices
- `exclude_indices` (Optional[List[int]]): Exclude item indices
- `return_metadata` (bool): Include metadata in results

**Returns:**
- List of recommendation dictionaries with keys: `index`, `distance`, `label`, `class_name`, `rank`

#### `search(query_embedding, k=10, filter_by_class=None, exclude_indices=None)`

Low-level search function returning distances and indices.

**Returns:**
- Tuple of (distances, indices) arrays

#### `save(filepath)`

Save the engine to disk (saves both model and FAISS index).

#### `load(filepath)`

Load the engine from disk.

## Performance Characteristics

Based on Fashion MNIST dataset (60,000 items, 2048D embeddings):

- **PCA Reduction**: 2048D → 128D (explains ~95% variance)
- **Memory**: ~30MB (reduced) vs ~500MB (original)
- **Compression Ratio**: ~32x reduction
- **Search Speed**: ~10,000+ queries/second
- **Index Build Time**: <1 second
- **Average Query Time**: <1ms per query

## Integration with CV Pipeline

The recommendation engine integrates seamlessly with the CV team's ResNet-50 embeddings:

```python
# 1. Load CV embeddings
embeddings = np.load('../ComputerVisionFiles/fashion_mnist_resnet50_embeddings.npy')
labels = np.load('../ComputerVisionFiles/fashion_mnist_labels.npy')

with open('../ComputerVisionFiles/fashion_mnist_classes.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# 2. Build engine
engine = FashionRecommendationEngine(
    embeddings=embeddings,
    labels=labels,
    class_names=class_names,
    n_components=128
)
engine.build_index()

# 3. For a new user upload, get embedding from CV pipeline
# (using the get_embedding_for_pil function from ResNet.ipynb)
user_image_embedding = get_embedding_for_pil(user_uploaded_image)

# 4. Get recommendations
recommendations = engine.recommend(user_image_embedding, k=10)

# 5. Return top-k similar items to user
```

## Architecture

```
┌─────────────────┐
│  CV Embeddings  │ (2048D)
│  (ResNet-50)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      PCA        │ (2048D → 128D)
│  Dimensionality │
│    Reduction    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FAISS Index    │ (L2 or Cosine)
│  Similarity     │
│     Search      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Ranking &      │
│  Filtering      │
└─────────────────┘
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'faiss'"

**Solution:**
```bash
# For CPU version (recommended)
pip install faiss-cpu

# For GPU version (if you have CUDA)
pip install faiss-gpu
```

### Issue: "FileNotFoundError: fashion_mnist_resnet50_embeddings.npy"

**Solution:**
```bash
cd ..  # Go to project root
git lfs install
git lfs pull
cd RecommendationFiles
```

### Issue: Memory errors with large datasets

**Solution:** The PCA reduction helps, but if you still have issues:
- Reduce `n_components` in the engine initialization (try 64 or 96)
- Process data in batches

### Issue: Slow performance

**Solution:**
- Make sure you're using the reduced embeddings (PCA enabled)
- Verify FAISS is installed correctly
- Check that you're not accidentally using full 2048D embeddings

## Quick Verification

Run this Python one-liner to quickly verify:

```bash
python -c "
import numpy as np
from recommendation_engine import FashionRecommendationEngine

embeddings = np.load('../ComputerVisionFiles/fashion_mnist_resnet50_embeddings.npy')
labels = np.load('../ComputerVisionFiles/fashion_mnist_labels.npy')
with open('../ComputerVisionFiles/fashion_mnist_classes.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

engine = FashionRecommendationEngine(embeddings, labels, class_names, n_components=128)
engine.build_index()
recs = engine.recommend(embeddings[0], k=5)
print('✓ Success! Engine works. Got', len(recs), 'recommendations')
"
```

## Notes

- The engine uses L2-normalized embeddings for cosine similarity when `index_type="cosine"`
- PCA is applied before indexing to reduce memory and improve search speed
- Filtering is done post-search for flexibility
- The engine can handle metadata for additional filtering/ranking criteria
- Saved models include both the PCA transformer and FAISS index

## Future Enhancements

- Support for approximate nearest neighbor (ANN) indices for even faster search
- Multi-query batch search optimization
- Weighted similarity combining multiple features
- Integration with user preference learning
- GPU acceleration support

## File Structure

```
RecommendationFiles/
├── recommendation_engine.py    # Main engine module
├── example_usage.py            # Example usage script
├── benchmark.py                # Performance benchmarking
├── test_integration.py         # Integration tests
├── quick_test.py              # Quick verification script
├── run_tests.sh               # Automated test runner
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```
