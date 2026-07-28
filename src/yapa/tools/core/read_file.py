"""Read file tool."""

from pathlib import Path

from yapa.tools.base import JsonValue, Tool


class ReadFile(Tool):
    """Read file contents with optional offset and limit."""

    def __init__(self):
        """Configure the read_file tool with path, offset, and limit parameters."""
        super().__init__(
            name="read_file",
            description=(
                "Read the contents of a file. "
                "Can optionally specify offset (1-indexed line) "
                "and limit (max lines)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-indexed)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read",
                    },
                },
                "required": ["path"],
            },
            needs_approval=False,
        )

    async def execute(
        self,
        path: str = "",
        offset: int | None = None,
        limit: int | None = None,
        **kwargs: object,
    ) -> JsonValue:
        """Read file content with optional range."""
        try:
            p = Path(path).resolve()
            lines = p.read_text().splitlines(keepends=True)
            start = (offset - 1) if offset else 0
            end = start + limit if limit else None
            return "".join(lines[start:end])
        except Exception as e:
            return f"Error: {e}"


read_file = ReadFile()
