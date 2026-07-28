"""List directory tool."""

from pathlib import Path

from yapa.tools.base import JsonValue, Tool


class ListDir(Tool):
    def __init__(self):
        super().__init__(
            name="list_dir",
            description="List files and directories in a path. Shows names, types (file/dir), and sizes.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the directory",
                    },
                },
                "required": ["path"],
            },
            needs_approval=False,
        )

    async def execute(self, path: str = "", **kwargs: object) -> JsonValue:
        try:
            p = Path(path).resolve()
            if not p.is_dir():
                return f"Error: {path} is not a directory"
            entries: list[str] = []
            for entry in sorted(p.iterdir()):
                if entry.is_dir():
                    entries.append(f"{entry.name}/")
                elif entry.is_file():
                    size = entry.stat().st_size
                    entries.append(f"{entry.name} ({size} bytes)")
                else:
                    entries.append(entry.name)
            return "\n".join(entries)
        except Exception as e:
            return f"Error: {e}"


list_dir = ListDir()