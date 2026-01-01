"""Database checkpointing and graph compilation utilities.

This module provides functions for managing PostgreSQL connection pooling,
graph compilation, and checkpoint management for the LangGraph agent.
"""

from typing import Optional
from urllib.parse import quote_plus

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.core.common.config import Environment, settings
from app.core.common.logging import logger

# Module-level singleton for connection pool
_connection_pool: Optional[AsyncConnectionPool] = None


async def get_connection_pool() -> Optional[AsyncConnectionPool]:
    """Get or create PostgreSQL connection pool with environment-specific settings.

    Returns:
        Optional[AsyncConnectionPool]: Connection pool instance, or None if creation fails in production.
    """
    global _connection_pool

    if _connection_pool is None:
        try:
            # Configure pool size based on environment
            max_size = settings.POSTGRES_POOL_SIZE

            connection_url = (
                "postgresql://"
                f"{quote_plus(settings.POSTGRES_USER)}:{quote_plus(settings.POSTGRES_PASSWORD)}"
                f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
            )

            _connection_pool = AsyncConnectionPool(
                connection_url,
                open=False,
                max_size=max_size,
                kwargs={
                    "autocommit": True,
                    "connect_timeout": 5,
                    "prepare_threshold": None,
                },
            )
            await _connection_pool.open()
            logger.info("connection_pool_created", max_size=max_size, environment=settings.ENVIRONMENT.value)
        except Exception as e:
            logger.error("connection_pool_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
            # In production, we might want to degrade gracefully
            if settings.ENVIRONMENT == Environment.PRODUCTION:
                logger.warning("continuing_without_connection_pool", environment=settings.ENVIRONMENT.value)
                return None
            raise e
    return _connection_pool


async def get_checkpointer():
    # Get connection pool (may be None in production if DB unavailable)
    connection_pool = await get_connection_pool()
    if connection_pool:
        checkpointer = AsyncPostgresSaver(connection_pool)
        await checkpointer.setup()
    else:
        # In production, proceed without checkpointer if needed
        checkpointer = None
        if settings.ENVIRONMENT != Environment.PRODUCTION:
            raise Exception("Connection pool initialization failed")
    return checkpointer


async def clear_checkpoints(session_id: str) -> None:
    """Clear all checkpoints for a session from database.

    Args:
        session_id: The session ID to clear checkpoints for.

    Raises:
        Exception: If there's an error clearing the checkpoints.
    """
    try:
        # Make sure the pool is initialized in the current event loop
        conn_pool = await get_connection_pool()

        # Use a new connection for this specific operation
        async with conn_pool.connection() as conn:
            for table in settings.CHECKPOINT_TABLES:
                try:
                    await conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (session_id,))
                    logger.info(f"Cleared {table} for session {session_id}")
                except Exception as e:
                    logger.error(f"Error clearing {table}", error=str(e))
                    raise

    except Exception as e:
        logger.error("Failed to clear chat history", error=str(e))
        raise
