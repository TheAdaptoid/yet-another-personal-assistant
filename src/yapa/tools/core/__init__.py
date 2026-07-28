"""Concrete tool implementations."""

from yapa.tools.base import Tool


def default_tools() -> list[Tool]:
    from .calculator import calculator
    from .grep import grep
    from .list_dir import list_dir
    from .read_file import read_file
    return [calculator, read_file, grep, list_dir]