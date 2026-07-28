"""Grep tool — search files for a pattern."""

from pathlib import Path

from yapa.tools.base import JsonValue, Tool


class Grep(Tool):
    """Search file contents for a regex pattern."""

    def __init__(self):
        """Configure the grep tool with pattern and path parameters."""
        super().__init__(
            name="grep",
            description=(
                "Search for a pattern in files within a directory. "
                "Returns matching file paths with line numbers and content."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the directory to search",
                    },
                    "include": {
                        "type": "string",
                        "description": (
                            "Optional glob pattern to filter files (e.g. '*.py')"
                        ),
                    },
                },
                "required": ["pattern", "path"],
            },
            needs_approval=False,
        )

    async def execute(
        self,
        pattern: str = "",
        path: str = "",
        include: str | None = None,
        **kwargs: object,
    ) -> JsonValue:
        """Search files matching pattern and return matching lines."""
        import re

        try:
            root = Path(path).resolve()
            if not root.is_dir():
                return f"Error: {path} is not a directory"
            results: list[str] = []
            glob_pattern = f"**/{include}" if include else "**/*"
            for f in sorted(root.glob(glob_pattern)):
                if not f.is_file():
                    continue
                try:
                    text = f.read_text(errors="replace")
                    for i, line in enumerate(text.splitlines(), 1):
                        if re.search(pattern, line):
                            rel = f.relative_to(root)
                            results.append(f"{rel}:{i}:{line}")
                except Exception:
                    continue
            return "\n".join(results)
        except Exception as e:
            return f"Error: {e}"


grep = Grep()
