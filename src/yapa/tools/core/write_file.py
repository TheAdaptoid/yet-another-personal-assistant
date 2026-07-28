"""Write file tool."""

from pathlib import Path

from yapa.tools.base import JsonValue, Tool


class WriteFile(Tool):
    """Write content to a file at the specified path."""

    def __init__(self):
        """Configure the write_file tool with path and content parameters."""
        super().__init__(
            name="write_file",
            description=(
                "Write content to a file at the specified path. "
                "The parent directory must already exist. "
                "Overwrites existing files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
            needs_approval=True,
        )

    async def execute(
        self, path: str = "", content: str = "", **kwargs: object
    ) -> JsonValue:
        """Write content to file, error if parent does not exist."""
        try:
            p = Path(path).resolve()
            if not p.parent.exists():
                return "Error: parent directory does not exist"
            p.write_text(content)
            return "ok"
        except Exception as e:
            return f"Error: {e}"


write_file = WriteFile()
