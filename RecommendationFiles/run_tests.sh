#!/bin/bash

# Quick test script to verify the recommendation engine works

echo "=========================================="
echo "Fashion Recommendation Engine - Test Runner"
echo "=========================================="
echo ""

# Check if data files exist
echo "1. Checking data files..."
if [ ! -f "../ComputerVisionFiles/fashion_mnist_resnet50_embeddings.npy" ]; then
    echo "   ❌ ERROR: embeddings file not found!"
    echo "   Please run: git lfs pull"
    exit 1
fi

if [ ! -f "../ComputerVisionFiles/fashion_mnist_labels.npy" ]; then
    echo "   ❌ ERROR: labels file not found!"
    echo "   Please run: git lfs pull"
    exit 1
fi

if [ ! -f "../ComputerVisionFiles/fashion_mnist_classes.txt" ]; then
    echo "   ❌ ERROR: classes file not found!"
    exit 1
fi

echo "   ✓ All data files found"
echo ""

# Check if required packages are installed
echo "2. Checking Python packages..."
python3 -c "import numpy; import sklearn; import faiss; import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "   ⚠️  Some packages missing. Installing..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "   ❌ ERROR: Failed to install packages"
        exit 1
    fi
else
    echo "   ✓ All required packages installed"
fi
echo ""

# Run integration tests
echo "3. Running integration tests..."
python3 test_integration.py
if [ $? -eq 0 ]; then
    echo "   ✓ Integration tests passed"
else
    echo "   ❌ Integration tests failed"
    exit 1
fi
echo ""

# Run example usage (quick version)
echo "4. Running example usage (this may take a minute)..."
python3 example_usage.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ Example usage completed successfully"
else
    echo "   ❌ Example usage failed"
    exit 1
fi
echo ""

echo "=========================================="
echo "✅ All tests passed! Everything works!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  - Run 'python3 example_usage.py' to see detailed output"
echo "  - Run 'python3 benchmark.py' for performance analysis"

