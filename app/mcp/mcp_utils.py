"""Utilities for MCP tool handling."""

from typing import Any, Callable, Optional

from langchain_core.messages import ToolMessage

from app.core.logging import logger
from app.mcp.dependencies import get_mcp_session_manager


async def handle_mcp_tool_call(
    tool_fn: Callable,
    tool_call: dict[str, Any],
    tool_name: str,
    max_retries: int = 1,
    on_reconnect: Optional[Callable] = None,
) -> ToolMessage:
    """Handle MCP tool calls with reconnection logic.

    Args:
        tool_fn: The tool function to invoke.
        tool_call: Dictionary containing tool call information with keys:
            - "args": Arguments to pass to the tool
            - "id": Tool call ID
        tool_name: Name of the tool being called.
        max_retries: Maximum number of reconnection attempts. Defaults to 1.
        on_reconnect: Optional callback to execute after successful reconnection
            (e.g., to reload tools).

    Returns:
        ToolMessage: A ToolMessage containing either the tool result or error information.
    """
    for attempt in range(max_retries + 1):
        try:
            tool_result = await tool_fn.ainvoke(tool_call["args"])
            return ToolMessage(
                content=tool_result,
                name=tool_name,
                tool_call_id=tool_call["id"],
            )

        except Exception as tool_error:
            error_type = type(tool_error).__name__

            # Check if it's a ClosedResourceError and we can retry
            if "ClosedResourceError" in error_type and attempt < max_retries:
                logger.warning(
                    "mcp_connection_closed_retrying",
                    tool_name=tool_name,
                    attempt=attempt + 1,
                    error=str(tool_error),
                )

                # Attempt to reconnect MCP sessions
                try:
                    mcp_manager = get_mcp_session_manager()
                    reconnected = await mcp_manager.reconnect()
                    if reconnected:
                        logger.info("mcp_reconnected_retrying_tool")
                        # Call the reconnection callback if provided
                        if on_reconnect:
                            await on_reconnect()
                        continue  # Retry the tool call
                except Exception as reconnect_error:
                    logger.error("mcp_reconnection_failed", error=str(reconnect_error))

            # Either not a ClosedResourceError, out of retries, or reconnection failed
            logger.error(
                "mcp_tool_call_failed",
                error=str(tool_error),
                error_type=error_type,
                tool_name=tool_name,
                tool_call_id=tool_call["id"],
                attempt=attempt + 1,
            )

            error_msg = f"[ERROR] Tool '{tool_name}' failed: {str(tool_error)}"
            if "ClosedResourceError" in error_type:
                error_msg += " (MCP connection issue. Attempted reconnection.)"

            return ToolMessage(
                content=error_msg,
                name=tool_name,
                tool_call_id=tool_call["id"],
            )
