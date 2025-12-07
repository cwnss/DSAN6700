from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel
from typing import List
from uuid import uuid4
import json
import io

from PIL import Image

from backend.app.services.s3_service import upload_file_to_s3
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


# -----------------------------
# Utility: Read image once
# -----------------------------
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


# -----------------------------
# Request Models
# -----------------------------
class OutfitRequest(BaseModel):
    occasion: str
    season: str


class SaveOutfitRequest(BaseModel):
    items: List[int]
    occasion: str
    season: str


# -----------------------------
# Upload Wardrobe Item
# -----------------------------
@router.post("/wardrobe/upload")
async def upload_wardrobe_item(
    category: str,
    file: UploadFile = File(...),
    db=Depends(get_db)  # db is a real asyncpg Connection
):
    request_id = str(uuid4())
    logger.info(f"[{request_id}] Upload started: category={category}, filename={file.filename}")

    try:
        # --------------------------------------------------------
        # Read UploadFile contents
        # --------------------------------------------------------
        t0 = time.time()
        contents = await file.read()
        logger.info(f"[{request_id}] Step1: read file ({len(contents)} bytes) in {time.time() - t0:.3f}s")

        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # --------------------------------------------------------
        # Upload to S3
        # --------------------------------------------------------
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

        # --------------------------------------------------------
        # Insert wardrobe row (db is a CONNECTION now)
        # --------------------------------------------------------
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

        # --------------------------------------------------------
        # Compute embedding
        # --------------------------------------------------------
        t3 = time.time()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        vector = compute_embedding(image).tolist()
        logger.info(f"[{request_id}] Step4: embedding computed ({time.time() - t3:.3f}s)")

        # --------------------------------------------------------
        # Insert embedding row
        # --------------------------------------------------------
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



# -----------------------------
# Predict + Similar Items
# -----------------------------
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

        async with db.acquire() as conn:
            similar = await find_similar_items(
                embedding,
                conn=conn,
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


# -----------------------------
# Generate Outfits
# -----------------------------
@router.post("/outfits/generate")
async def generate_outfits(req: OutfitRequest):
    outfits = recommender.recommend_outfits(req.occasion, req.season)
    return {
        "occasion": req.occasion,
        "season": req.season,
        "count": len(outfits),
        "outfits": outfits
    }


# -----------------------------
# Save Outfit
# -----------------------------
@router.post("/outfits/save")
async def save_outfit(req: SaveOutfitRequest, db=Depends(get_db)):
    try:
        outfit_id = str(uuid4())

        async with db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO saved_outfits (outfit_id, items, occasion, season)
                VALUES ($1, $2, $3, $4)
                """,
                outfit_id,
                req.items,
                req.occasion,
                req.season,
            )

        return {"status": "success", "outfit_id": outfit_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save outfit failed: {str(e)}")


# -----------------------------
# Get Saved Outfits
# -----------------------------
@router.get("/outfits/saved")
async def get_saved_outfits(db=Depends(get_db)):

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

        created_at = row["created_at"]
        created_at = created_at.isoformat() if created_at else None

        response.append({
            "outfit_id": row["outfit_id"],
            "occasion": row["occasion"],
            "season": row["season"],
            "created_at": created_at,
            "items": enriched_items
        })

    return {"saved_outfits": response}


# -----------------------------
# Upload Document
# -----------------------------
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

        return {"status": "success", "document_id": document_id, "s3_uri": s3_uri}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
