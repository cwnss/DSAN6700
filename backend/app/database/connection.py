import asyncpg
from backend.app.config import settings
from pgvector.asyncpg import register_vector
import logging

logger = logging.getLogger("db")
logger.setLevel(logging.INFO)

_db_pool = None


async def connect_to_db():
    """
    Called ONCE at application startup.
    Creates and warms the asyncpg pool.
    """
    global _db_pool

    if _db_pool is not None:
        return _db_pool

    logger.info("Initializing PostgreSQL connection pool…")

    try:
        _db_pool = await asyncpg.create_pool(
            dsn=settings.DATABASE_URL,
            min_size=1,
            max_size=5,
            timeout=10,
            init=_init_connection
        )
        logger.info("PostgreSQL pool initialized successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize DB pool: {e}")
        raise

    return _db_pool


async def _init_connection(conn):
    """Registers pgvector for every fresh connection."""
    await register_vector(conn)


async def get_db():
    """
    FastAPI dependency.
    Uses an already-initialized pool.
    NEVER creates a pool on request.
    """
    if _db_pool is None:
        await connect_to_db()   # safety net, runs instantly after startup

    async with _db_pool.acquire() as conn:
        await register_vector(conn)  # safe no-op
        yield conn
