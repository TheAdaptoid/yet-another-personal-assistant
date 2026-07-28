"""Tests for ToolRegistry."""

from unittest.mock import MagicMock

from yapa.tools.core import default_tools
from yapa.tools.registry import ToolRegistry


class TestToolRegistry:
    def test_initializes_with_default_tools(self):
        registry = ToolRegistry(default_tools())
        assert registry.get_tool("calculator") is not None
        assert registry.get_tool("read_file") is not None
        assert registry.get_tool("write_file") is not None
        assert registry.get_tool("grep") is not None
        assert registry.get_tool("list_dir") is not None
        assert registry.get_tool("bash") is not None

    def test_list_tools(self):
        registry = ToolRegistry(default_tools())
        tools = registry.list_tools()
        assert len(tools) >= 6
        names = {t.name for t in tools}
        assert "calculator" in names

    def test_get_unknown_tool(self):
        registry = ToolRegistry()
        assert registry.get_tool("nonexistent") is None

    def test_register_and_unregister(self):
        registry = ToolRegistry()
        tool = MagicMock()
        tool.name = "test_tool"
        registry.register(tool)
        assert registry.get_tool("test_tool") is tool
        registry.unregister("test_tool")
        assert registry.get_tool("test_tool") is None
