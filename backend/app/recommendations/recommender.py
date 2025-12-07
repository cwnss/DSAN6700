import numpy as np
import pickle
from typing import List, Dict
from RecommendationFiles.example_usage import FashionRecommendationEngine


class OutfitRecommender:

    def __init__(
        self,
        engine_pkl_path: str = "RecommendationFiles/recommendation_engine.pkl",
        metadata_path: str = "backend/app/recommendations/item_metadata.pkl"
    ):
        # Load FAISS engine (already built in your example file)
        self.engine = FashionRecommendationEngine.load(engine_pkl_path)

        # Load metadata containing:
        # item_id → class_label, season_tags, occasion_tags
        with open(metadata_path, "rb") as f:
            self.meta = pickle.load(f)

    # ---------------------------------------------------------
    # Build outfit recommendations using FAISS + metadata
    # ---------------------------------------------------------
    def recommend_outfits(self, occasion: str, season: str, k: int = 10):
        outfits = []

        # Filter items that match occasion + season tags
        valid_indices = [
            idx for idx, item in self.meta.items()
            if occasion.lower() in item["occasion"]
            and season.lower() in item["season"]
        ]

        if not valid_indices:
            return []

        # Compute centroid embedding of valid items
        embeddings = self.engine.embeddings[valid_indices]
        query_vector = np.mean(embeddings, axis=0)

        # Query FAISS ANN
        recommendations = self.engine.recommend(
            query_embedding=query_vector,
            k=k,
        )

        # Build outfit suggestions by grouping categories
        tops = []
        bottoms = []
        shoes = []

        for rec in recommendations:
            item_meta = self.meta[rec["index"]]
            category = item_meta["category"]

            if category == "top":
                tops.append(rec["index"])
            elif category == "bottom":
                bottoms.append(rec["index"])
            elif category == "shoes":
                shoes.append(rec["index"])

        # Combine into outfits
        for t in tops[:5]:
            for b in bottoms[:5]:
                for s in shoes[:5]:
                    outfits.append({
                        "items": [t, b, s],
                        "score": 1.0  # FAISS does not provide multi-item scoring here
                    })

        return outfits[:k]
