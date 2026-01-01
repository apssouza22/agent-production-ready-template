"""This file contains the LangGraph Agent/workflow and interactions with the LLM."""

import asyncio
from typing import (
    AsyncGenerator,
    Optional,
)

from asgiref.sync import sync_to_async
from langchain_core.messages import convert_to_openai_messages
from langfuse.langchain import CallbackHandler
from langgraph.graph.state import (
    Command,
    CompiledStateGraph,
)
from langgraph.types import (
    RunnableConfig,
    StateSnapshot,
)

from app.core.agentic.tools import tools
from app.core.common.config import settings
from app.core.common.logging import logger
from app.core.langgraph.checkpointer import clear_checkpoints, create_compiled_graph
from app.core.langgraph.graph_utils import create_chat_node, create_tool_call_node, process_messages
from app.core.langgraph.memory import get_relevant_memory, update_memory
from app.core.llm.llm import llm_service
from app.core.llm.llm_utils import dump_messages
from app.schemas import Message


class LangGraphAgent:
    """Manages the LangGraph Agent/workflow and interactions with the LLM.

    This class handles the creation and management of the LangGraph workflow,
    including LLM interactions, database connections, and response processing.
    """

    def __init__(self):
        """Initialize the LangGraph Agent with necessary components."""
        # Use the LLM service with tools bound
        self.llm_service = llm_service
        self.llm_service.bind_tools(tools)
        self.tools_by_name = {tool.name: tool for tool in tools}
        self._graph: Optional[CompiledStateGraph] = None
        logger.info(
            "langgraph_agent_initialized",
            model=settings.DEFAULT_LLM_MODEL,
            environment=settings.ENVIRONMENT.value,
        )

    async def _ensure_graph(self) -> CompiledStateGraph:
        """Lazy initialization of the graph.

        Returns:
            CompiledStateGraph: The compiled graph instance.
        """
        if self._graph is None:
            chat_node = create_chat_node(self.llm_service, self.tools_by_name)
            tool_call_node = create_tool_call_node(self.tools_by_name)
            self._graph = await create_compiled_graph(chat_node, tool_call_node)
        return self._graph

    async def get_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
    ) -> list[dict]:
        """Get a response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for Langfuse tracking.
            user_id (Optional[str]): The user ID for Langfuse tracking.

        Returns:
            list[dict]: The response from the LLM.
        """
        graph = await self._ensure_graph()
        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": [CallbackHandler()],
            "metadata": {
                "user_id": user_id,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "debug": settings.DEBUG,
            },
        }
        relevant_memory = (
            await get_relevant_memory(user_id, messages[-1].content)
        ) or "No relevant memory found."
        try:
            response = await graph.ainvoke(
                input={"messages": dump_messages(messages), "long_term_memory": relevant_memory},
                config=config,
            )
            # Run memory update in background without blocking the response
            asyncio.create_task(
                update_memory(
                    user_id, convert_to_openai_messages(response["messages"]), config["metadata"]
                )
            )
            return process_messages(response["messages"])
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")

    async def get_stream_response(
        self, messages: list[Message], session_id: str, user_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Get a stream response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for the conversation.
            user_id (Optional[str]): The user ID for the conversation.

        Yields:
            str: Tokens of the LLM response.
        """
        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": [
                CallbackHandler(
                    environment=settings.ENVIRONMENT.value, debug=False, user_id=user_id, session_id=session_id
                )
            ],
            "metadata": {
                "user_id": user_id,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "debug": settings.DEBUG,
            },
        }
        graph = await self._ensure_graph()

        relevant_memory = (
            await get_relevant_memory(user_id, messages[-1].content)
        ) or "No relevant memory found."

        try:
            async for token, _ in graph.astream(
                {"messages": dump_messages(messages), "long_term_memory": relevant_memory},
                config,
                stream_mode="messages",
            ):
                try:
                    yield token.content
                except Exception as token_error:
                    logger.error("Error processing token", error=str(token_error), session_id=session_id)
                    # Continue with next token even if current one fails
                    continue

            # After streaming completes, get final state and update memory in background
            state: StateSnapshot = await sync_to_async(graph.get_state)(config=config)
            if state.values and "messages" in state.values:
                asyncio.create_task(
                    update_memory(
                        user_id, convert_to_openai_messages(state.values["messages"]), config["metadata"]
                    )
                )
        except Exception as stream_error:
            logger.error("Error in stream processing", error=str(stream_error), session_id=session_id)
            raise stream_error

    async def get_chat_history(self, session_id: str) -> list[Message]:
        """Get the chat history for a given thread ID.

        Args:
            session_id (str): The session ID for the conversation.

        Returns:
            list[Message]: The chat history.
        """
        graph = await self._ensure_graph()

        state: StateSnapshot = await sync_to_async(graph.get_state)(
            config={"configurable": {"thread_id": session_id}}
        )
        return process_messages(state.values["messages"]) if state.values else []

    async def clear_chat_history(self, session_id: str) -> None:
        """Clear all chat history for a given thread ID.

        Args:
            session_id: The ID of the session to clear history for.

        Raises:
            Exception: If there's an error clearing the chat history.
        """
        await clear_checkpoints(session_id)
