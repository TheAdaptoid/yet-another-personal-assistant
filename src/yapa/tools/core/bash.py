"""Bash tool — executes shell commands."""

import asyncio

from yapa.tools.base import JsonValue, Tool


class Bash(Tool):
    def __init__(self):
        super().__init__(
            name="bash",
            description="Execute a shell command and return its output. Use for running scripts, compiling code, or any command-line operation.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    },
                },
                "required": ["command"],
            },
            needs_approval=True,
        )

    async def execute(self, command: str = "", **kwargs: object) -> JsonValue:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=60.0
            )
            output = stdout.decode()
            if stderr:
                output += "\nstderr:\n" + stderr.decode()
            if proc.returncode != 0:
                output += f"\n(exit code {proc.returncode})"
            return output.strip()
        except asyncio.TimeoutError:
            return "Error: command timed out after 60 seconds"
        except Exception as e:
            return f"Error: {e}"


bash = Bash()