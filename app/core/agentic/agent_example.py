from typing import Optional, Any, AsyncGenerator

from asgiref.sync import sync_to_async
from langchain_core.messages import ToolMessage
from langchain_core.messages import convert_to_openai_messages
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.types import RunnableConfig, StateSnapshot, Command

from app.core.agentic.prompts import load_system_prompt
from app.core.common.config import settings
from app.core.common.logging import logger
from app.core.common.metrics import llm_inference_duration_seconds
from app.core.common.graph_utils import process_messages
from app.core.memory.memory import get_relevant_memory, bg_update_memory
from app.core.llm.llm_utils import dump_messages, prepare_messages, process_llm_response
from app.schemas import GraphState
from app.schemas import Message


class AgentExample:
  """Example agent to demonstrate the agentic framework."""
  _graph: Optional[CompiledStateGraph] = None

  def __init__(self, name, llm_service, tools: list, checkpointer: AsyncPostgresSaver):
    self.checkpointer = checkpointer
    self.name = name

    self.llm_service = llm_service
    self.llm_service.bind_tools(tools)
    self.tools_by_name = {tool.name: tool for tool in tools}
    self.config = {
      "callbacks": [CallbackHandler()],
      "metadata": {
        "environment": settings.ENVIRONMENT.value,
        "debug": settings.DEBUG,
      },
    }

  async def agent_invoke(
      self,
      messages: list[Message],
      session_id: str,
      user_id: Optional[int] = None,
  ) -> list[Message] | list[Any]:
    graph = await self._ensure_graph()
    config = await self._create_config(session_id, user_id)

    relevant_memory = (
                        await get_relevant_memory(user_id, messages[-1].content)
                      ) or "No relevant memory found."

    try:
      response = await graph.ainvoke(
          input={"messages": dump_messages(messages), "long_term_memory": relevant_memory},
          config=config,
      )
      bg_update_memory(user_id, convert_to_openai_messages(response["messages"]), config["metadata"])

      return process_messages(response["messages"])
    except Exception as e:
      logger.error(f"Error getting response: {str(e)}")
      return []

  async def _create_config(self, session_id, user_id):
    config = self.config.copy()
    config["configurable"] = {"thread_id": session_id}
    config["metadata"]["user_id"] = user_id
    config["metadata"]["session_id"] = session_id
    return config

  async def agent_invoke_async(
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

      graph = await self._ensure_graph()
      config = await self._create_config(session_id, user_id)
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
          bg_update_memory(user_id, convert_to_openai_messages(state.values["messages"]), config["metadata"])

      except Exception as stream_error:
        logger.error("Error in stream processing", error=str(stream_error), session_id=session_id)
        raise stream_error

  async def _ensure_graph(self) -> CompiledStateGraph:
    """Lazy initialization of the graph.

    Returns:
        CompiledStateGraph: The compiled graph instance.
    """
    if self._graph is not None:
      return self._graph

    return await self._create_compiled_graph()

  async def _chat_node(self, state: GraphState, config: RunnableConfig) -> Command:
    """Process the chat state and generate a response.

    Args:
        state: The current state of the conversation.
        config: The configuration for the node execution.

    Returns:
        Command: Command object with updated state and next node to execute.
    """
    # Get the current LLM instance for metrics
    current_llm = self.llm_service.get_llm()
    model_name = (
      current_llm.model_name
      if current_llm and hasattr(current_llm, "model_name")
      else settings.DEFAULT_LLM_MODEL
    )

    system_prompt = load_system_prompt(long_term_memory=state.long_term_memory)

    # Prepare messages with system prompt
    messages = prepare_messages(state.messages, current_llm, system_prompt)

    try:
      # Use LLM service with automatic retries and circular fallback
      with llm_inference_duration_seconds.labels(model=model_name).time():
        response_message = await self.llm_service.call(dump_messages(messages))

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

  async def _tool_call_node(self, state: GraphState) -> Command:
    """Process tool calls from the last message.

    Args:
        state: The current agent state containing messages and tool calls.

    Returns:
        Command: Command object with updated messages and routing back to chat.
    """
    outputs = []
    for tool_call in state.messages[-1].tool_calls:
      tool_result = await self.tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
      outputs.append(
          ToolMessage(
              content=tool_result,
              name=tool_call["name"],
              tool_call_id=tool_call["id"],
          )
      )
    return Command(update={"messages": outputs}, goto="chat")

  async def _create_compiled_graph(self) -> Optional[CompiledStateGraph]:
    """Create and compile the LangGraph workflow with nodes and checkpointer.

    Args:
        chat_node_fn: The chat node function.
        tool_call_node_fn: The tool call node function.

    Returns:
        Optional[CompiledStateGraph]: The compiled graph instance, or None if creation fails in production.
    """
    try:
      graph_builder = StateGraph(GraphState)
      graph_builder.add_node("chat", self._chat_node, ends=["tool_call", END])
      graph_builder.add_node("tool_call", self._tool_call_node, ends=["chat"])
      graph_builder.set_entry_point("chat")
      graph_builder.set_finish_point("chat")

      compiled_graph = graph_builder.compile(
          checkpointer=self.checkpointer, name=f"{self.name}"
      )

      logger.info(
          "graph_created",
          graph_name=f"{self.name}",
          environment=settings.ENVIRONMENT.value,
          has_checkpointer=self.checkpointer is not None,
      )
      return compiled_graph
    except Exception as e:
      logger.error("graph_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
      raise e


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

