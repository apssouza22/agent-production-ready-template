from contextlib import AsyncExitStack
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.sse import sse_client, logger

from app.core.config import settings
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



@asynccontextmanager
async def get_mcp_dependencies() -> AsyncGenerator[Resource]:
  tools = []
  sessions = []
  async with AsyncExitStack() as stack:
    for hostname in settings.MCP_HOSTNAMES:
      logger.info(f"Connecting to MCP server at {hostname}")
      session = await stack.enter_async_context(
          mcp_sse_client(hostname)
      )
      tools += await load_mcp_tools(session)
      logger.info(f"Loaded {len(tools)} tools from {hostname}")
      sessions.append(session)
    yield Resource(
        tools=tools,
        sessions=sessions,
    )
