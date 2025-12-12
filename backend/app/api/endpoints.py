from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4
import json
import io

from PIL import Image

from backend.app.services.s3_service import upload_file_to_s3, get_presigned_url
from backend.app.database.connection import get_db
from backend.app.config import settings

import logging
import time
import traceback
import io
from fastapi import HTTPException

logger = logging.getLogger("wardrobe")
logger.setLevel(logging.INFO)


from backend.app.services.embedding_service import (
    classify_image,
    compute_embedding,
    find_similar_items
)

from backend.app.recommendations.recommender import OutfitRecommender

router = APIRouter()
recommender = OutfitRecommender()


# Utility: Read image once
async def read_image_bytes(file: UploadFile) -> bytes:
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return contents


def load_pil_image(contents: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")


# Request Models
class OutfitRequest(BaseModel):
    occasion: str
    season: str


class SaveOutfitRequest(BaseModel):
    items: List[int]
    occasion: str
    season: str
    name: str = ""


class UpdateWardrobeItemRequest(BaseModel):
    category: str
    subcategory: str = ""
    season: List[str] = []
    brand: str = ""
    colors: List[str] = []
    occasions: List[str] = []
    notes: str = ""

class RateOutfitRequest(BaseModel):
    rating: int
    notes: Optional[str] = ""


# Upload Wardrobe Item
@router.post("/wardrobe/upload")
async def upload_wardrobe_item(
    category: str,
    file: UploadFile = File(...),
    db=Depends(get_db)  # db is a real asyncpg Connection
):
    request_id = str(uuid4())
    logger.info(f"[{request_id}] Upload started: category={category}, filename={file.filename}")

    try:
        # Read UploadFile contents
        t0 = time.time()
        contents = await file.read()
        logger.info(f"[{request_id}] Step1: read file ({len(contents)} bytes) in {time.time() - t0:.3f}s")

        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Upload to S3
        t1 = time.time()
        s3_key = f"wardrobe/{uuid4()}/{file.filename}"

        file_stream = io.BytesIO(contents)
        file_stream.seek(0)

        image_url = await upload_file_to_s3(
            file=file_stream,
            bucket=settings.S3_BUCKET_IMAGES,
            key=s3_key
        )
        logger.info(f"[{request_id}] Step2: S3 upload completed in {time.time() - t1:.3f}s")

        # Insert wardrobe row (db is a CONNECTION now)
        t2 = time.time()
        item_id = await db.fetchval(
            """
            INSERT INTO wardrobe_items (image_url, category)
            VALUES ($1, $2)
            RETURNING item_id
            """,
            image_url, category
        )
        logger.info(f"[{request_id}] Step3: DB insert wardrobe → item_id={item_id} ({time.time() - t2:.3f}s)")

        # Compute embedding
        t3 = time.time()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        vector = compute_embedding(image).tolist()
        logger.info(f"[{request_id}] Step4: embedding computed ({time.time() - t3:.3f}s)")

        # Insert embedding row
        t4 = time.time()
        await db.execute(
            """
            INSERT INTO embeddings (item_id, embedding)
            VALUES ($1, $2)
            """,
            item_id, vector
        )
        logger.info(f"[{request_id}] Step5: DB insert embedding ({time.time() - t4:.3f}s)")

        total = time.time() - t0
        logger.info(f"[{request_id}] ✔ Upload completed in {total:.3f}s")

        return {
            "status": "success",
            "item_id": item_id,
            "image_url": image_url,
        }

    except Exception as e:
        logger.error(f"[{request_id}] Upload failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Wardrobe upload failed: {str(e)}")


# Get All Wardrobe Items
@router.get("/wardrobe/items")
async def get_wardrobe_items(db=Depends(get_db)):
    # Fetch all wardrobe items from database
    # db is already a connection from get_db() dependency
    items = await db.fetch(
        """
        SELECT item_id, image_url, category, metadata
        FROM wardrobe_items
        ORDER BY item_id DESC
        """
    )
    
    # Convert to list of dicts
    response = []
    for row in items:
        # Handle metadata - it might be None, dict, or JSON string
        metadata = row.get("metadata")
        if metadata is None:
            metadata = {}
        elif isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        elif not isinstance(metadata, dict):
            metadata = {}
        
        # Convert S3 URI to presigned URL for frontend display
        image_url = row["image_url"]
        if image_url and image_url.startswith("s3://"):
            try:
                image_url = get_presigned_url(image_url)
                logger.info(f"Converted S3 URI to presigned URL for item {row['item_id']}")
            except Exception as e:
                logger.error(f"Failed to generate presigned URL for item {row['item_id']}: {e}")
                # Keep original S3 URI if conversion fails
        
        response.append({
            "item_id": row["item_id"],
            "image_url": image_url,
            "category": row["category"],
            "subcategory": metadata.get("subcategory"),
            "brand": metadata.get("brand"),
            "colors": metadata.get("colors", []),
            "occasions": metadata.get("occasions", []),
            "season": metadata.get("season"),
            "notes": metadata.get("notes"),
        })
    
    return {"items": response}


# Predict + Similar Items
@router.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    db=Depends(get_db)
):
    try:
        contents = await read_image_bytes(file)
        image = load_pil_image(contents)

        classified = classify_image(image)

        if "embedding" not in classified:
            raise HTTPException(
                status_code=500,
                detail="Model did not return an embedding."
            )

        embedding = classified["embedding"]

        similar = await find_similar_items(
            embedding,
            conn=db,
            limit=5
        )

        return {
            "filename": file.filename,
            "predicted_label": classified.get("label"),
            "confidence": classified.get("confidence"),
            "nearest_index": classified.get("nearest_index"),
            "similar_items": [
                {
                    "item_id": row["item_id"],
                    "distance": float(row["distance"]),
                } for row in similar
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# Generate Outfits
@router.post("/outfits/generate")
async def generate_outfits(req: OutfitRequest, db=Depends(get_db)):
    outfits = await recommender.recommend_outfits(req.occasion, req.season, db)
    return {
        "occasion": req.occasion,
        "season": req.season,
        "count": len(outfits),
        "outfits": outfits
    }


# Save Outfit
@router.post("/outfits/save")
async def save_outfit(req: SaveOutfitRequest, db=Depends(get_db)):
    try:
        outfit_id = str(uuid4())

        await db.execute(
            """
            INSERT INTO saved_outfits (outfit_id, items, occasion, season, name)
            VALUES ($1, $2, $3, $4, $5)
            """,
            outfit_id,
            req.items,
            req.occasion,
            req.season,
            req.name,
        )

        return {"status": "success", "outfit_id": outfit_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save outfit failed: {str(e)}")


# Get Saved Outfits
@router.get("/outfits/saved")
async def get_saved_outfits(db=Depends(get_db)):

    outfits = await db.fetch(
        """
        SELECT outfit_id, items, occasion, season, name, created_at
        FROM saved_outfits
        ORDER BY created_at DESC
        """
    )

    wardrobe = await db.fetch(
        """
        SELECT item_id, image_url, category, metadata
        FROM wardrobe_items
        """
    )

    # Build lookup and convert S3 URIs to presigned URLs
    wardrobe_lookup = {}
    for wrow in wardrobe:
        item_data = dict(wrow)
        # Convert S3 URI to presigned URL
        image_url = item_data.get("image_url")
        if image_url and image_url.startswith("s3://"):
            try:
                item_data["image_url"] = get_presigned_url(image_url)
            except Exception as e:
                logger.error(f"Failed to generate presigned URL: {e}")
        wardrobe_lookup[wrow["item_id"]] = item_data

    response = []
    for row in outfits:

        enriched_items = [
            wardrobe_lookup.get(
                item_id,
                {"item_id": item_id, "image_url": None, "category": None, "metadata": {}}
            )
            for item_id in row["items"]
        ]

        created_at = row["created_at"]
        created_at = created_at.isoformat() if created_at else None

        response.append({
            "outfit_id": row["outfit_id"],
            "name": row["name"] or "",
            "occasion": row["occasion"],
            "season": row["season"],
            "created_at": created_at,
            "items": enriched_items
        })

    return {"saved_outfits": response}


# Delete Saved Outfit
@router.delete("/outfits/{outfit_id}")
async def delete_outfit(outfit_id: str, db=Depends(get_db)):
    try:
        await db.execute("DELETE FROM saved_outfits WHERE outfit_id = $1", outfit_id)
        return {"status": "success", "deleted_outfit_id": outfit_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete outfit failed: {str(e)}")


# Update Saved Outfit
class UpdateOutfitRequest(BaseModel):
    name: Optional[str] = None
    occasion: Optional[str] = None
    season: Optional[str] = None
    items: Optional[List[int]] = None

@router.patch("/outfits/{outfit_id}")
async def update_outfit(outfit_id: str, req: UpdateOutfitRequest, db=Depends(get_db)):
    try:
        # Build dynamic update query based on provided fields
        updates = []
        values = []
        param_num = 1
        
        if req.name is not None:
            updates.append(f"name = ${param_num}")
            values.append(req.name)
            param_num += 1
        
        if req.occasion is not None:
            updates.append(f"occasion = ${param_num}")
            values.append(req.occasion)
            param_num += 1
        
        if req.season is not None:
            updates.append(f"season = ${param_num}")
            values.append(req.season)
            param_num += 1
        
        if req.items is not None:
            updates.append(f"items = ${param_num}")
            values.append(req.items)
            param_num += 1
        
        if not updates:
            return {"status": "success", "message": "No updates provided", "outfit_id": outfit_id}
        
        # Add outfit_id as the last parameter
        values.append(outfit_id)
        
        query = f"UPDATE saved_outfits SET {', '.join(updates)} WHERE outfit_id = ${param_num}"
        await db.execute(query, *values)
        
        return {"status": "success", "updated_outfit_id": outfit_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update outfit failed: {str(e)}")


# Delete Wardrobe Item
@router.delete("/wardrobe/item/{item_id}")
async def delete_wardrobe_item(item_id: int, db=Depends(get_db)):
    try:
        # Delete embedding first (foreign key)
        await db.execute("DELETE FROM embeddings WHERE item_id = $1", item_id)
        # Delete wardrobe item
        result = await db.execute("DELETE FROM wardrobe_items WHERE item_id = $1", item_id)
        return {"status": "success", "deleted_item_id": item_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


# Update Wardrobe Item
@router.patch("/wardrobe/item/{item_id}")
async def update_wardrobe_item(item_id: int, req: UpdateWardrobeItemRequest, db=Depends(get_db)):
    try:
        # Build metadata JSON from request fields
        metadata = {
            "subcategory": req.subcategory,
            "season": req.season,
            "brand": req.brand,
            "colors": req.colors,
            "occasions": req.occasions,
            "notes": req.notes,
        }

        # Update category and metadata
        await db.execute(
            """
            UPDATE wardrobe_items
            SET category = $1, metadata = $2
            WHERE item_id = $3
            """,
            req.category,
            json.dumps(metadata),
            item_id,
        )

        return {"status": "success", "item_id": item_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


# Clear All Wardrobe Items (for testing)
@router.delete("/wardrobe/clear-all")
async def clear_all_wardrobe_items(db=Depends(get_db)):
    try:
        # Delete all embeddings first (foreign key constraint)
        await db.execute("DELETE FROM embeddings")
        # Delete all wardrobe items
        await db.execute("DELETE FROM wardrobe_items")
        return {"status": "success", "message": "All wardrobe items cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")


# Upload Document
@router.post("/documents/upload", status_code=201)
async def upload_document(file: UploadFile = File(...), db=Depends(get_db)):
    try:
        contents = await read_image_bytes(file)  # reading contents is required for S3

        document_id = str(uuid4())
        s3_key = f"documents/{document_id}/{file.filename}"

        s3_uri = await upload_file_to_s3(
            file=io.BytesIO(contents),
            bucket=settings.S3_BUCKET_DOCUMENTS,
            key=s3_key
        )

        metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
            "s3_uri": s3_uri
        }

        await db.execute(
            """
            INSERT INTO documents (document_id, s3_uri, metadata)
            VALUES ($1, $2, $3::jsonb)
            """,
            document_id,
            s3_uri,
            json.dumps(metadata)
        )

        return {"status": "success", "document_id": document_id, "s3_uri": s3_uri}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/outfits/{outfit_id}/rate")
async def rate_outfit(outfit_id: str, req: RateOutfitRequest, db=Depends(get_db)):
    """
    Add a rating (1–5) + notes to a saved outfit.
    """
    # Validate rating
    if req.rating < 1 or req.rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5.")

    rating_id = str(uuid4())

    try:
        await db.execute(
            """
            INSERT INTO outfit_ratings (rating_id, outfit_id, rating, notes)
            VALUES ($1, $2, $3, $4)
            """,
            rating_id,
            outfit_id,
            req.rating,
            req.notes,
        )

        return {
            "status": "success",
            "rating_id": rating_id,
            "outfit_id": outfit_id,
            "rating": req.rating,
            "notes": req.notes,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add rating: {str(e)}"
        )

@router.get("/outfits/{outfit_id}/ratings")
async def get_outfit_ratings(outfit_id: str, db=Depends(get_db)):
    try:
        rows = await db.fetch(
            """
            SELECT rating_id, rating, notes, created_at
            FROM outfit_ratings
            WHERE outfit_id = $1
            ORDER BY created_at DESC
            """,
            outfit_id
        )

        return {
            "outfit_id": outfit_id,
            "ratings": [
                {
                    "rating_id": r["rating_id"],
                    "rating": r["rating"],
                    "notes": r["notes"],
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None
                }
                for r in rows
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch outfit ratings: {str(e)}"
        )
