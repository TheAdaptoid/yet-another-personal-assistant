"""Edit file tool — replaces first occurrence of old_string with new_string."""

from pathlib import Path

from yapa.tools.base import JsonValue, Tool


class EditFile(Tool):
    """Replace text in a file without rewriting the entire file."""

    def __init__(self):
        """Configure the edit_file tool with path and string parameters."""
        super().__init__(
            name="edit_file",
            description=(
                "Replace the first occurrence of old_string with new_string "
                "in a file. Use for surgical edits without rewriting "
                "entire files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to edit",
                    },
                    "old_string": {
                        "type": "string",
                        "description": (
                            "The exact string to replace (first occurrence only)"
                        ),
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The string to replace it with",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
            needs_approval=True,
        )

    async def execute(
        self,
        path: str = "",
        old_string: str = "",
        new_string: str = "",
        **kwargs: object,
    ) -> JsonValue:
        """Replace the first occurrence of old_string with new_string."""
        try:
            p = Path(path).resolve()
            text = p.read_text()
            if old_string not in text:
                return f"Error: could not find '{old_string}' in {path}"
            new_text = text.replace(old_string, new_string, 1)
            p.write_text(new_text)
            return "ok"
        except Exception as e:
            return f"Error: {e}"


edit_file = EditFile()
