from langchain_core.tools import StructuredTool
from mcp import ClientSession
from pydantic import BaseModel


class ToolRequest(BaseModel):
    tool_name: str
    a: int
    b: int


class Resource(BaseModel):
    tools: list[StructuredTool]
    sessions: list[ClientSession]

    class Config:
        arbitrary_types_allowed = True
