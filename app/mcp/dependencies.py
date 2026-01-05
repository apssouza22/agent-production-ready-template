from contextlib import AsyncExitStack
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.sse import sse_client

from app.core.config import settings
from app.core.logging import logger
from app.mcp.models import Resource


@asynccontextmanager
async def mcp_sse_client(
    mcp_host: str = "localhost:7001",
) -> AsyncGenerator[ClientSession]:
  """
  Creates and initializes an MCP client session over SSE.

  Establishes an SSE connection to the MCP server and yields an initialized
  `ClientSession` for communication.

  Yields:
      ClientSession: An initialized MCP client session.
  """
  async with sse_client(
      f"http://{mcp_host}/sse"
  ) as (
      read_stream,
      write_stream,
  ):
    async with ClientSession(read_stream, write_stream) as session:
      await session.initialize()
      yield session


class MCPSessionManager:
  """Manages MCP client sessions for the application lifetime."""

  def __init__(self):
    self._exit_stack: Optional[AsyncExitStack] = None
    self._resource: Optional[Resource] = None
    self._initialized: bool = False

  async def initialize(self) -> Resource:
    """Initialize MCP sessions and load tools."""
    if self._initialized:
      return self._resource

    self._exit_stack = AsyncExitStack()
    await self._exit_stack.__aenter__()

    tools = []
    sessions = []

    for hostname in settings.MCP_HOSTNAMES:
      try:
        session = await self._exit_stack.enter_async_context(
            mcp_sse_client(hostname)
        )
        session_tools = await load_mcp_tools(session)
        tools.extend(session_tools)
        sessions.append(session)
        logger.info("connected_to_mcp_server", hostname=hostname, tool_count=len(session_tools))
      except Exception as e:
        logger.error("failed_to_connect_to_mcp_server", hostname=hostname, error=str(e))

    self._resource = Resource(tools=tools, sessions=sessions)
    self._initialized = True
    return self._resource

  async def reconnect(self) -> bool:
    """Attempt to reconnect to MCP servers."""
    logger.info("reconnecting_to_mcp_servers")
    try:
      await self.cleanup()
      await self.initialize()
      return True
    except Exception as e:
      logger.error("mcp_reconnection_failed", error=str(e))
      return False

  async def cleanup(self):
    """Close all MCP sessions."""
    if self._exit_stack is not None:
      try:
        await self._exit_stack.__aexit__(None, None, None)
        logger.info("mcp_sessions_closed_successfully")
      except Exception as e:
        logger.error("mcp_sessions_cleanup_failed", error=str(e))
      finally:
        self._exit_stack = None
        self._resource = None
        self._initialized = False

  def get_resource(self) -> Resource:
    """Get the current MCP resource."""
    if not self._initialized or self._resource is None:
      raise RuntimeError("MCP session manager not initialized")
    return self._resource


# Module-level singleton
_mcp_session_manager: Optional[MCPSessionManager] = None


def get_mcp_session_manager() -> MCPSessionManager:
  """Get the global MCP session manager."""
  global _mcp_session_manager
  if _mcp_session_manager is None:
    _mcp_session_manager = MCPSessionManager()
  return _mcp_session_manager
