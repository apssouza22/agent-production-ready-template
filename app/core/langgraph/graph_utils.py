"""Graph node functions and message processing utilities.

This module provides factory functions for creating graph nodes and utilities
for processing messages in the LangGraph workflow.
"""

from typing import Callable

from langchain_core.messages import BaseMessage, ToolMessage, convert_to_openai_messages
from langgraph.graph import END
from langgraph.graph.state import Command
from langgraph.types import RunnableConfig

from app.core.agentic.prompts import load_system_prompt
from app.core.common.config import settings
from app.core.common.logging import logger
from app.core.common.metrics import llm_inference_duration_seconds
from app.core.llm.llm_utils import dump_messages, prepare_messages, process_llm_response
from app.schemas import GraphState, Message


def create_chat_node(llm_service, tools_by_name: dict) -> Callable:
    """Create the chat node function with LLM service dependency.

    Args:
        llm_service: The LLM service instance.
        tools_by_name: Dictionary mapping tool names to tool instances.

    Returns:
        Callable: The chat node function.
    """

    async def chat_node(state: GraphState, config: RunnableConfig) -> Command:
        """Process the chat state and generate a response.

        Args:
            state: The current state of the conversation.
            config: The configuration for the node execution.

        Returns:
            Command: Command object with updated state and next node to execute.
        """
        # Get the current LLM instance for metrics
        current_llm = llm_service.get_llm()
        model_name = (
            current_llm.model_name
            if current_llm and hasattr(current_llm, "model_name")
            else settings.DEFAULT_LLM_MODEL
        )

        SYSTEM_PROMPT = load_system_prompt(long_term_memory=state.long_term_memory)

        # Prepare messages with system prompt
        messages = prepare_messages(state.messages, current_llm, SYSTEM_PROMPT)

        try:
            # Use LLM service with automatic retries and circular fallback
            with llm_inference_duration_seconds.labels(model=model_name).time():
                response_message = await llm_service.call(dump_messages(messages))

            # Process response to handle structured content blocks
            response_message = process_llm_response(response_message)

            logger.info(
                "llm_response_generated",
                session_id=config["configurable"]["thread_id"],
                model=model_name,
                environment=settings.ENVIRONMENT.value,
            )

            # Determine next node based on whether there are tool calls
            if response_message.tool_calls:
                goto = "tool_call"
            else:
                goto = END

            return Command(update={"messages": [response_message]}, goto=goto)
        except Exception as e:
            logger.error(
                "llm_call_failed_all_models",
                session_id=config["configurable"]["thread_id"],
                error=str(e),
                environment=settings.ENVIRONMENT.value,
            )
            raise Exception(f"failed to get llm response after trying all models: {str(e)}")

    return chat_node


def create_tool_call_node(tools_by_name: dict) -> Callable:
    """Create the tool call node function with tools dependency.

    Args:
        tools_by_name: Dictionary mapping tool names to tool instances.

    Returns:
        Callable: The tool call node function.
    """

    async def tool_call_node(state: GraphState) -> Command:
        """Process tool calls from the last message.

        Args:
            state: The current agent state containing messages and tool calls.

        Returns:
            Command: Command object with updated messages and routing back to chat.
        """
        outputs = []
        for tool_call in state.messages[-1].tool_calls:
            tool_result = await tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=tool_result,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return Command(update={"messages": outputs}, goto="chat")

    return tool_call_node


def process_messages(messages: list[BaseMessage]) -> list[Message]:
    """Convert BaseMessages to API Message format.

    Args:
        messages: List of BaseMessage objects from the graph.

    Returns:
        list[Message]: List of Message objects for the API response.
    """
    openai_style_messages = convert_to_openai_messages(messages)
    # keep just assistant and user messages
    return [
        Message(role=message["role"], content=str(message["content"]))
        for message in openai_style_messages
        if message["role"] in ["assistant", "user"] and message["content"]
    ]
