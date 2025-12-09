from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from backend.app.api.endpoints import router as api_router

app = FastAPI()

# Set up Prometheus metrics
Instrumentator().instrument(app).expose(app)

# include your API router
app.include_router(api_router)

@app.get("/")
def root():
    return {"status": "ok", "message": "Backend is running"}
