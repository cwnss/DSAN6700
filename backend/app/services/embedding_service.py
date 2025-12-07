"""
Embedding service for:
- Converting images to vectors (ResNet50 embeddings)
- Classifying using precomputed wardrobe embeddings
- Running pgvector similarity search against wardrobe_items in PostgreSQL
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from backend.app.database.connection import get_db  # <-- required for pgvector search


# 1. IMAGE PREPROCESSING FOR RESNET50

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 2. LOAD RESNET50 EMBEDDING MODEL

resnet = models.resnet50(pretrained=True)
resnet.eval()

embedding_model = nn.Sequential(*list(resnet.children())[:-1])


def compute_embedding(image: Image.Image) -> np.ndarray:
    """
    Compute a 2048-D ResNet50 embedding for a PIL image.
    """
    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        vec = embedding_model(tensor).squeeze().numpy()

    return vec  # shape (2048,)


# 3. LOAD PRECOMPUTED EMBEDDINGS (Numpy Files)

WARDROBE_EMB = np.load("ComputerVisionFiles/fashion_mnist_resnet50_embeddings.npy")
WARDROBE_LABELS = np.load("ComputerVisionFiles/fashion_mnist_labels.npy")

assert WARDROBE_EMB.shape[0] == len(WARDROBE_LABELS), \
    "Mismatch: number of embeddings != number of labels"


# 4. COSINE SIMILARITY (NumPy)
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between one vector and a matrix of vectors.
    """
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(b_norm, a_norm)  # returns similarity scores array


def find_nearest_neighbor(embedding: np.ndarray):
    """
    Classify an embedding by nearest precomputed wardrobe vector.
    Returns:
        - label (str/number)
        - confidence (similarity score)
        - index (nearest vector index)
    """

    sims = cosine_similarity(embedding, WARDROBE_EMB)
    idx = int(np.argmax(sims))
    return WARDROBE_LABELS[idx], float(sims[idx]), idx


# 5. MAIN API: CLASSIFY IMAGE USING PRECOMPUTED EMBEDDINGS
def classify_image(image: Image.Image):
    """
    Convert image → embedding → nearest precomputed label.
    """
    emb = compute_embedding(image)
    label, confidence, idx = find_nearest_neighbor(emb)

    return {
        "embedding": emb,
        "label": str(label),
        "confidence": confidence,
        "nearest_index": idx
    }


# 6. PGVECTOR SIMILARITY SEARCH (async)

async def find_similar_items(vector, conn, limit=10):
    vector = np.asarray(vector, dtype=np.float32)

    rows = await conn.fetch(
        """
        SELECT item_id, embedding <-> $1 AS distance
        FROM embeddings
        ORDER BY embedding <-> $1
        LIMIT $2
        """,
        vector,
        limit
    )
    return rows