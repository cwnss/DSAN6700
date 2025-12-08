# ----------------------------------------------------
# 1. Base Image
# ----------------------------------------------------
    FROM python:3.10-slim-bookworm

    # Prevent Python from writing .pyc files
    ENV PYTHONDONTWRITEBYTECODE=1
    ENV PYTHONUNBUFFERED=1
    ENV PYTHONPATH=/
    
    # ----------------------------------------------------
    # 2. System Dependencies
    # ----------------------------------------------------
    RUN apt-get update && apt-get install -y \
        libgl1-mesa-glx \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*
    
    # ----------------------------------------------------
    # 3. Set Working Directory
    # ----------------------------------------------------
    WORKDIR /

    # ----------------------------------------------------
    # 4. Install Python Dependencies
    # ----------------------------------------------------
    COPY backend/requirements.txt /backend/
    RUN pip install --no-cache-dir -r /backend/requirements.txt

    # ----------------------------------------------------
    # 5. Copy Source Code
    # ----------------------------------------------------
    COPY backend/ /backend/
    COPY ComputerVisionFiles/ /ComputerVisionFiles/
    COPY RecommendationFiles/ /RecommendationFiles/
    
    # ----------------------------------------------------
    # 6. Expose Port
    # ----------------------------------------------------
    EXPOSE 8000
    
    # ----------------------------------------------------
    # 7. Start FastAPI App
    # ----------------------------------------------------
    CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
    