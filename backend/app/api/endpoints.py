from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4
import json
import io

from PIL import Image
import numpy as np

from backend.app.services.s3_service import upload_file_to_s3
from backend.app.database.connection import get_db
from backend.app.config import settings

from backend.app.services.embedding_service import (
    classify_image,
    compute_embedding,
    find_similar_items
)

from backend.app.recommendations.recommender import OutfitRecommender

router = APIRouter()
recommender = OutfitRecommender()


# Utility: Load a PIL Image
def load_image_file(uploaded_file: UploadFile):
    uploaded_file.file.seek(0)
    contents = uploaded_file.file.read()
    return Image.open(io.BytesIO(contents)).convert("RGB")


# Request Models
class OutfitRequest(BaseModel):
    occasion: str
    season: str

class SaveOutfitRequest(BaseModel):
    items: List[int]
    occasion: str
    season: str


# Upload Wardrobe Item
@router.post("/wardrobe/upload")
async def upload_wardrobe_item(
    category: str,
    file: UploadFile = File(...), 
    conn = Depends(get_db)
):
    """
    1. Upload clothing item image to S3  
    2. Insert wardrobe item row  
    3. Compute embedding  
    4. Insert embedding into pgvector table  
    """

    try:
        # Upload image → S3
        s3_key = f"wardrobe/{uuid4()}/{file.filename}"
        image_url = await upload_file_to_s3(
            file=file,
            bucket=settings.S3_BUCKET_IMAGES,
            key=s3_key
        )

        # Insert wardrobe item row
        db = await get_db()
        async with db.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO wardrobe_items (image_url, category)
                VALUES ($1, $2)
                RETURNING item_id
                """,
                image_url, category
            )
            item_id = row["item_id"]

        # Compute embedding
        image = load_image_file(file)
        vector = compute_embedding(image).tolist()

        # Insert embedding
        async with db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO embeddings (item_id, embedding)
                VALUES ($1, $2)
                """,
                item_id,
                vector
            )

        return {
            "status": "success",
            "item_id": item_id,
            "image_url": image_url,
            "message": "Wardrobe item saved + embedding stored."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Predict + Similar Items
@router.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    conn = Depends(get_db)
):
    try:
        image = load_image_file(file)
        classified = classify_image(image)

        similar = await find_similar_items(
            classified["embedding"],
            conn=conn,
            limit=5
        )

        return {
            "filename": file.filename,
            "predicted_label": classified["label"],
            "confidence": classified["confidence"],
            "nearest_index": classified["nearest_index"],
            "similar_items": [
                {
                    "item_id": row["item_id"],
                    "distance": float(row["distance"])
                }
                for row in similar
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# Generate Outfit (FAISS/logic)
@router.post("/outfits/generate")
async def generate_outfits(req: OutfitRequest):
    outfits = recommender.recommend_outfits(req.occasion, req.season)
    return {
        "occasion": req.occasion,
        "season": req.season,
        "count": len(outfits),
        "outfits": outfits
    }


# Save Outfit
@router.post("/outfits/save")
async def save_outfit(req: SaveOutfitRequest):
    try:
        outfit_id = str(uuid4())
        db = await get_db()

        async with db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO saved_outfits (outfit_id, items, occasion, season)
                VALUES ($1, $2, $3, $4)
                """,
                outfit_id,
                req.items,
                req.occasion,
                req.season
            )

        return {"status": "success", "outfit_id": outfit_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save outfit failed: {str(e)}")


# List Saved Outfits
@router.get("/outfits/saved")
async def get_saved_outfits():
    db = await get_db()

    async with db.acquire() as conn:
        outfits = await conn.fetch(
            """
            SELECT outfit_id, items, occasion, season, created_at
            FROM saved_outfits
            ORDER BY created_at DESC
            """
        )

        wardrobe = await conn.fetch(
            """
            SELECT item_id, image_url, category, metadata
            FROM wardrobe_items
            """
        )

    wardrobe_lookup = {row["item_id"]: dict(row) for row in wardrobe}

    response = []
    for row in outfits:
        enriched_items = [
            wardrobe_lookup.get(
                item_id,
                {"item_id": item_id, "image_url": None, "category": None, "metadata": {}}
            )
            for item_id in row["items"]
        ]

        response.append({
            "outfit_id": row["outfit_id"],
            "occasion": row["occasion"],
            "season": row["season"],
            "created_at": row["created_at"].isoformat(),
            "items": enriched_items
        })

    return {"saved_outfits": response}


# Upload Document
@router.post("/documents/upload", status_code=201)
async def upload_document(file: UploadFile = File(...)):
    try:
        document_id = str(uuid4())
        s3_key = f"documents/{document_id}/{file.filename}"

        s3_uri = await upload_file_to_s3(
            file=file,
            bucket=settings.S3_BUCKET_DOCUMENTS,
            key=s3_key
        )

        metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
            "s3_uri": s3_uri
        }

        db = await get_db()
        async with db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO documents (document_id, s3_uri, metadata)
                VALUES ($1, $2, $3::jsonb)
                """,
                document_id,
                s3_uri,
                json.dumps(metadata)
            )

        return {
            "status": "success",
            "document_id": document_id,
            "s3_uri": s3_uri
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
