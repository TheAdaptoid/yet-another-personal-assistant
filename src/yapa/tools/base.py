"""Base abstractions for executable tools."""

from abc import ABC, abstractmethod
from typing import Any

type JsonValue = (
    int | float | str | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


class Tool(ABC):
    """Abstract base class for tools that can be executed with arbitrary arguments."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        needs_approval: bool = False,
    ):
        """Initialize the tool with a name, description, and parameters."""

        self.name = name
        self.description = description
        self.parameters = parameters
        self.needs_approval = needs_approval

    @abstractmethod
    async def execute(self, **kwargs: Any) -> JsonValue:
        """
        Execute the tool with the given arguments.

        Args:
            **kwargs: Arbitrary keyword arguments to be passed to the tool.

        Returns:
            JsonValue: The result of the tool execution.
        """
