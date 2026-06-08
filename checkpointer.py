import os
import sys
import asyncio
from typing import Optional, Generator, AsyncGenerator
from contextlib import contextmanager, asynccontextmanager
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool, AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# Set up event loop policy on Windows for psycopg compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Global pool references
_sync_pool: Optional[ConnectionPool] = None
_async_pool: Optional[AsyncConnectionPool] = None


def get_sync_pool() -> ConnectionPool:
    """Get or initialize the synchronous connection pool."""
    global _sync_pool
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set in .env")
    if _sync_pool is None:
        # Create pool with recommended parameters
        _sync_pool = ConnectionPool(
            conninfo=DATABASE_URL,
            max_size=10,
            kwargs={"autocommit": True, "row_factory": dict_row}
        )
    return _sync_pool


def get_async_pool() -> AsyncConnectionPool:
    """Get or initialize the asynchronous connection pool."""
    global _async_pool
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set in .env")
    if _async_pool is None:
        # Create pool with recommended parameters
        _async_pool = AsyncConnectionPool(
            conninfo=DATABASE_URL,
            max_size=10,
            open=False,
            kwargs={"autocommit": True, "row_factory": dict_row}
        )
    return _async_pool


def close_sync_pool():
    """Close the synchronous connection pool if initialized."""
    global _sync_pool
    if _sync_pool is not None:
        _sync_pool.close()
        _sync_pool = None


async def close_async_pool():
    """Close the asynchronous connection pool if initialized."""
    global _async_pool
    if _async_pool is not None:
        await _async_pool.close()
        _async_pool = None


@contextmanager
def get_sync_checkpointer() -> Generator[PostgresSaver, None, None]:
    """Yield a synchronous PostgresSaver checkpointer using the connection pool."""
    pool = get_sync_pool()
    yield PostgresSaver(pool)


@asynccontextmanager
async def get_async_checkpointer() -> AsyncGenerator[AsyncPostgresSaver, None]:
    """Yield an asynchronous AsyncPostgresSaver checkpointer using the async connection pool."""
    pool = get_async_pool()
    await pool.open()
    yield AsyncPostgresSaver(pool)
