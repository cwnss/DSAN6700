import asyncpg
from backend.app.config import settings
from pgvector.asyncpg import register_vector

_db_pool = None


async def _init_pool():
    """
    Create the asyncpg pool and register pgvector on all connections.
    """
    global _db_pool

    if _db_pool is None:
        _db_pool = await asyncpg.create_pool(
            dsn=settings.DATABASE_URL,
            min_size=1,
            max_size=5,
            init=_init_connection  # register pgvector per connection
        )

    return _db_pool


async def _init_connection(conn):
    """
    Called for every new connection in the pool.
    Ensures pgvector is registered so asyncpg accepts numpy arrays / lists.
    """
    await register_vector(conn)


async def get_db():
    """
    Dependency for FastAPI.
    Ensures the pool exists, retrieves a connection,
    and releases it afterward.
    """
    pool = await _init_pool()
    async with pool.acquire() as conn:
        # register again (harmless, but ensures correctness)
        await register_vector(conn)
        yield conn
