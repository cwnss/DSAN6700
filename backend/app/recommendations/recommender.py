import time
import numpy as np
import logging
from typing import List, Dict, Set
from RecommendationFiles.recommendation_engine import FashionRecommendationEngine
from backend.app.metrics import ML_INFERENCE_TIME

logger = logging.getLogger(__name__)


class OutfitRecommender:
    def __init__(
        self,
        engine_pkl_path: str = "RecommendationFiles/recommendation_engine.pkl",
    ):
        # Load FAISS index + embeddings from the pre-trained recommendation engine
        self.engine = FashionRecommendationEngine.load(engine_pkl_path)
        logger.info(f"Loaded recommendation engine from {engine_pkl_path}")

    async def recommend_outfits(self, occasion: str, season: str, db, k: int = 10):
        """
        Build outfit recommendations using the ML recommendation engine.
        Uses embeddings to find visually compatible items.
        """
        start_time = time.time()
        # Get all wardrobe items WITH their embeddings
        wardrobe = await db.fetch(
            """
            SELECT w.item_id, w.category, w.metadata, e.embedding
            FROM wardrobe_items w
            INNER JOIN embeddings e ON e.item_id = w.item_id
            """
        )
        
        logger.info(f"[RECOMMENDER] Found {len(wardrobe) if wardrobe else 0} items with embeddings")

        if not wardrobe:
            logger.warning("[RECOMMENDER] No wardrobe items with embeddings found!")
            ML_INFERENCE_TIME.labels(operation="recommendation").observe(time.time() - start_time)
            return []

        def extract_list(meta, key):
            if not meta:
                return []
            if isinstance(meta, str):
                try:
                    import json
                    meta = json.loads(meta)
                except:
                    return []
            v = meta.get(key, [])
            if isinstance(v, str):
                return [v.lower().strip()]
            return [x.lower().strip() for x in v] if v else []

        def is_any_occasion(occasions: List[str]) -> bool:
            """Check if item can be used for any occasion"""
            for occ in occasions:
                if "any" in occ and "occasion" in occ:
                    return True
            return False
        
        def is_all_season(seasons: List[str]) -> bool:
            """Check if item can be used for any season"""
            for s in seasons:
                # Match "all-season", "all season", "all seasons", "any season"
                if ("all" in s or "any" in s) and "season" in s:
                    return True
            return False

        # Filter items by occasion and season
        filtered = []
        for row in wardrobe:
            meta = row["metadata"]
            item_occasions = extract_list(meta, "occasions")
            item_seasons = extract_list(meta, "season")
            
            # Log what we're checking
            logger.debug(f"[RECOMMENDER] Item {row['item_id']}: occasions={item_occasions}, seasons={item_seasons}")

            # Item matches if:
            # 1. Selected occasion matches item's occasions, OR
            # 2. Item is marked "any occasion", OR
            # 3. Item has no occasions set (flexible)
            occasion_match = (
                occasion.lower() in item_occasions or 
                is_any_occasion(item_occasions) or
                not item_occasions
            )
            
            # Same logic for seasons
            season_match = (
                season.lower() in item_seasons or 
                is_all_season(item_seasons) or
                not item_seasons
            )

            if occasion_match and season_match:
                filtered.append(row)
                logger.debug(f"[RECOMMENDER] ✓ Item {row['item_id']} passed filter")
            else:
                logger.debug(f"[RECOMMENDER] ✗ Item {row['item_id']} filtered out (occasion_match={occasion_match}, season_match={season_match})")

        logger.info(f"[RECOMMENDER] After occasion/season filter: {len(filtered)} items")
        
        # If no items match filter, use all items
        if not filtered:
            logger.info("[RECOMMENDER] No items match filter, using all items")
            filtered = list(wardrobe)

        # Separate items by category
        tops, bottoms, shoes, dresses, accessories, outerwear = [], [], [], [], [], []
        item_embeddings = {}
        
        # Similarity thresholds - only add if item ACTUALLY matches
        # Lower than before but still filters bad matches
        MIN_SIMILARITY_SHOES = 0.30
        MIN_SIMILARITY_OUTERWEAR = 0.30
        MIN_SIMILARITY_ACCESSORIES = 0.30
        
        for row in filtered:
            item_id = row["item_id"]
            cat = (row["category"] or "").lower()
            embedding = row["embedding"]
            
            if embedding is not None:
                item_embeddings[item_id] = {
                    "embedding": np.array(embedding),
                    "category": cat,
                    "row": row
                }
                
                if cat in ["top", "tops"]:
                    tops.append(item_id)
                elif cat in ["bottom", "bottoms"]:
                    bottoms.append(item_id)
                elif cat in ["shoe", "shoes"]:
                    shoes.append(item_id)
                elif cat in ["dress", "dresses"]:
                    dresses.append(item_id)
                elif cat in ["outerwear"]:
                    outerwear.append(item_id)
                elif cat in ["accessory", "accessories"]:
                    accessories.append(item_id)

        logger.info(f"[RECOMMENDER] Categories - tops:{len(tops)}, bottoms:{len(bottoms)}, shoes:{len(shoes)}, dresses:{len(dresses)}, outerwear:{len(outerwear)}, accessories:{len(accessories)}")
        
        # Debug: Show which items are in each category
        for item_id in bottoms:
            meta = item_embeddings[item_id]["row"]["metadata"]
            if isinstance(meta, str):
                import json
                try:
                    meta = json.loads(meta)
                except:
                    meta = {}
            subcategory = meta.get("subcategory", "unknown") if meta else "unknown"
            logger.info(f"[RECOMMENDER] Bottom item {item_id}: {subcategory}")

        if not tops and not bottoms and not dresses:
            logger.warning("[RECOMMENDER] No tops, bottoms, or dresses found!")
            ML_INFERENCE_TIME.labels(operation="recommendation").observe(time.time() - start_time)
            return []

        outfits = []
        
        # Track used items for VARIETY - don't repeat same items across outfits
        used_bottoms: Set[int] = set()
        used_shoes: Set[int] = set()
        used_outerwear: Set[int] = set()
        used_accessories: Set[int] = set()
        
        # Log available bottoms for debugging
        logger.info(f"[RECOMMENDER] Available bottoms: {bottoms}")
        
        def find_best_match(candidate_ids: List[int], outfit_emb: np.ndarray, 
                           used_set: Set[int], threshold: float) -> tuple:
            """Find best matching item that hasn't been used yet"""
            best_id = None
            best_score = 0
            
            # First try to find unused items
            for item_id in candidate_ids:
                if item_id in used_set:
                    continue
                item_emb = item_embeddings[item_id]["embedding"]
                similarity = np.dot(outfit_emb, item_emb) / (
                    np.linalg.norm(outfit_emb) * np.linalg.norm(item_emb) + 1e-8
                )
                if similarity > best_score and similarity >= threshold:
                    best_score = similarity
                    best_id = item_id
            
            # If all items used but we have items, allow reuse for variety in later outfits
            if best_id is None and len(used_set) >= len(candidate_ids) and candidate_ids:
                for item_id in candidate_ids:
                    item_emb = item_embeddings[item_id]["embedding"]
                    similarity = np.dot(outfit_emb, item_emb) / (
                        np.linalg.norm(outfit_emb) * np.linalg.norm(item_emb) + 1e-8
                    )
                    if similarity > best_score and similarity >= threshold:
                        best_score = similarity
                        best_id = item_id
            
            return best_id, best_score
        
        # Use the recommendation engine to find compatible items
        if tops and bottoms:
            for top_id in tops[:5]:
                top_emb = item_embeddings[top_id]["embedding"]
                
                try:
                    recommendations = self.engine.recommend(
                        query_embedding=top_emb,
                        k=20
                    )
                    
                    # Find best matching bottom WITH VARIETY - prefer unused bottoms
                    best_bottom, best_score = find_best_match(
                        bottoms, top_emb, used_bottoms, threshold=0.0  # No threshold for bottoms, always pick one
                    )
                    
                    # If no unused bottom found, allow reuse
                    if best_bottom is None and bottoms:
                        best_bottom = bottoms[0]
                        bottom_emb = item_embeddings[best_bottom]["embedding"]
                        best_score = np.dot(top_emb, bottom_emb) / (
                            np.linalg.norm(top_emb) * np.linalg.norm(bottom_emb) + 1e-8
                        )
                    
                    logger.info(f"[RECOMMENDER] Top {top_id} paired with bottom {best_bottom} (score: {best_score:.2f}, used_bottoms: {used_bottoms})")
                    
                    if best_bottom:
                        outfit_items = [top_id, best_bottom]
                        outfit_emb = (top_emb + item_embeddings[best_bottom]["embedding"]) / 2
                        total_score = best_score
                        num_matches = 2
                        used_bottoms.add(best_bottom)  # Track for variety
                        
                        # Add shoes if available AND matches well AND not already overused
                        if shoes:
                            shoe_id, shoe_score = find_best_match(
                                shoes, outfit_emb, used_shoes, MIN_SIMILARITY_SHOES
                            )
                            if shoe_id:
                                outfit_items.append(shoe_id)
                                outfit_emb = (outfit_emb + item_embeddings[shoe_id]["embedding"]) / 2
                                total_score += shoe_score
                                num_matches += 1
                                used_shoes.add(shoe_id)
                                logger.info(f"[RECOMMENDER] ✓ Added shoe {shoe_id} (score: {shoe_score:.2f})")
                            else:
                                logger.info(f"[RECOMMENDER] ✗ No shoe matched threshold {MIN_SIMILARITY_SHOES}")
                        
                        # Add outerwear if available AND matches well
                        if outerwear:
                            outer_id, outer_score = find_best_match(
                                outerwear, outfit_emb, used_outerwear, MIN_SIMILARITY_OUTERWEAR
                            )
                            if outer_id:
                                outfit_items.append(outer_id)
                                outfit_emb = (outfit_emb + item_embeddings[outer_id]["embedding"]) / 2
                                total_score += outer_score
                                num_matches += 1
                                used_outerwear.add(outer_id)
                                logger.info(f"[RECOMMENDER] ✓ Added outerwear {outer_id} (score: {outer_score:.2f})")
                            else:
                                logger.info(f"[RECOMMENDER] ✗ No outerwear matched threshold {MIN_SIMILARITY_OUTERWEAR}")
                        
                        # Add accessories if available AND matches well
                        if accessories:
                            acc_id, acc_score = find_best_match(
                                accessories, outfit_emb, used_accessories, MIN_SIMILARITY_ACCESSORIES
                            )
                            if acc_id:
                                outfit_items.append(acc_id)
                                total_score += acc_score
                                num_matches += 1
                                used_accessories.add(acc_id)
                                logger.info(f"[RECOMMENDER] ✓ Added accessory {acc_id} (score: {acc_score:.2f})")
                            else:
                                logger.info(f"[RECOMMENDER] ✗ No accessory matched threshold {MIN_SIMILARITY_ACCESSORIES}")
                        
                        avg_score = total_score / num_matches
                        
                        logger.info(f"[RECOMMENDER] Outfit: {outfit_items} - avg score: {avg_score:.2f}")
                        
                        outfits.append({
                            "items": outfit_items,
                            "score": float(avg_score),
                            "num_items": len(outfit_items),
                            "ml_powered": True
                        })
                        
                except Exception as e:
                    logger.error(f"[RECOMMENDER] Engine error: {e}")
                    outfit_items = [top_id, bottoms[0] if bottoms else None]
                    outfit_items = [x for x in outfit_items if x]
                    if outfit_items:
                        outfits.append({"items": outfit_items, "score": 0.5})
        
        # Handle dresses (they don't need bottoms)
        elif dresses:
            for dress_id in dresses[:5]:
                dress_emb = item_embeddings[dress_id]["embedding"]
                outfit_items = [dress_id]
                outfit_emb = dress_emb.copy()
                total_score = 1.0
                num_matches = 1
                
                # Add shoes if matches
                if shoes:
                    shoe_id, shoe_score = find_best_match(
                        shoes, outfit_emb, used_shoes, MIN_SIMILARITY_SHOES
                    )
                    if shoe_id:
                        outfit_items.append(shoe_id)
                        outfit_emb = (outfit_emb + item_embeddings[shoe_id]["embedding"]) / 2
                        total_score += shoe_score
                        num_matches += 1
                        used_shoes.add(shoe_id)
                
                # Add outerwear if matches
                if outerwear:
                    outer_id, outer_score = find_best_match(
                        outerwear, outfit_emb, used_outerwear, MIN_SIMILARITY_OUTERWEAR
                    )
                    if outer_id:
                        outfit_items.append(outer_id)
                        outfit_emb = (outfit_emb + item_embeddings[outer_id]["embedding"]) / 2
                        total_score += outer_score
                        num_matches += 1
                        used_outerwear.add(outer_id)
                
                # Add accessories if matches
                if accessories:
                    acc_id, acc_score = find_best_match(
                        accessories, outfit_emb, used_accessories, MIN_SIMILARITY_ACCESSORIES
                    )
                    if acc_id:
                        outfit_items.append(acc_id)
                        total_score += acc_score
                        num_matches += 1
                        used_accessories.add(acc_id)
                
                avg_score = total_score / num_matches
                
                logger.info(f"[RECOMMENDER] Dress outfit: {outfit_items} - avg score: {avg_score:.2f}")
                
                outfits.append({
                    "items": outfit_items,
                    "score": float(avg_score),
                    "num_items": len(outfit_items),
                    "ml_powered": True
                })
        
        # Fallback: simple category matching if ML didn't produce results
        if not outfits:
            logger.info("[RECOMMENDER] Using fallback category matching")
            tops_to_use = tops[:5] if tops else []
            bottoms_to_use = bottoms[:5] if bottoms else []
            
            for t in tops_to_use:
                for b in bottoms_to_use:
                    outfits.append({
                        "items": [t, b],
                        "score": 0.5
                    })
                    if len(outfits) >= k:
                        break
                if len(outfits) >= k:
                    break

        # Sort by score and return top k
        outfits.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"[RECOMMENDER] Returning {min(len(outfits), k)} outfits")

        ML_INFERENCE_TIME.labels(operation="recommendation").observe(time.time() - start_time)
        return outfits[:k]
