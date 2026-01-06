from mcp.server.fastmcp import FastMCP

from app.core.config import settings

mcpServer = FastMCP(
    "MCP Server",
    port=settings.MCP_SERVER_PORT,
)


@mcpServer.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcpServer.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"


if __name__ == "__main__":
    mcpServer.run(transport="sse")
