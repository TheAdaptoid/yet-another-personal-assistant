
from yapa.logging import get_logger

from .base import Tool

logger = get_logger(__name__)


class ToolRegistry:
    """Registry for managing tools."""

    def __init__(self, initial_tools: list[Tool] | None = None):
        """Initialize the ToolRegistry with an optional list of initial tools."""
        self._tools: dict[str, Tool] = {}

        if initial_tools:
            for tool in initial_tools:
                self.register(tool)

    def register(self, tool: Tool):
        """Register a new tool."""
        self._tools[tool.name] = tool

    def unregister(self, tool_name: str):
        """Unregister a tool by its name."""
        if tool_name in self._tools:
            del self._tools[tool_name]

    def get_tool(self, tool_name: str) -> Tool | None:
        """Retrieve a tool by its name."""
        return self._tools.get(tool_name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())
