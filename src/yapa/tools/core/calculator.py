"""Calculator tool — evaluates mathematical expressions."""

from yapa.tools.base import JsonValue, Tool


class Calculator(Tool):
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Evaluate a mathematical expression. Supports +, -, *, /, **, //, % and parentheses.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                    },
                },
                "required": ["expression"],
            },
            needs_approval=False,
        )

    async def execute(self, expression: str = "", **kwargs: object) -> JsonValue:
        allowed = {"__builtins__": {}}
        return eval(expression, allowed, {})


calculator = Calculator()