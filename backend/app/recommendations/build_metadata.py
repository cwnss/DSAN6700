import os
import pickle
from pathlib import Path
import numpy as np

# Ensure the project root is on the path
ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.append(str(ROOT))

from RecommendationFiles.recommendation_engine import FashionRecommendationEngine

# Load Recommendation Engine

ENGINE_PATH = "RecommendationFiles/recommendation_engine.pkl"

print(f"Loading engine from: {ENGINE_PATH}")
engine = FashionRecommendationEngine.load(ENGINE_PATH)

embeddings = engine.reduced_embeddings
n_items = embeddings.shape[0]

print(f"Engine loaded with {n_items} items.")

# Build simple metadata
# Categories: top, bottom, shoes (cycle)
# Occasions & Seasons from your UI

CATEGORIES = ["top", "bottom", "shoes"]
OCCASIONS = ["Casual", "Dinner", "Formal", "Party", "Everyday"]
SEASONS = ["Spring", "Summer", "Fall", "Winter"]

metadata = {}

for idx in range(n_items):

    metadata[idx] = {
        "category": CATEGORIES[idx % len(CATEGORIES)],
        "occasion": [OCCASIONS[idx % len(OCCASIONS)]],
        "season": [SEASONS[idx % len(SEASONS)]]
    }

print("Metadata generated for all items.")

# Save metadata file

OUTPUT_PATH = "backend/app/recommendations/item_metadata.pkl"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(metadata, f)

print(f"Metadata saved to: {OUTPUT_PATH}")
