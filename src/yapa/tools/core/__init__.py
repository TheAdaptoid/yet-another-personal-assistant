"""Concrete tool implementations."""

from yapa.tools.base import Tool


def default_tools() -> list[Tool]:
    """Return a list of all built-in tool instances."""
    from .bash import bash
    from .calculator import calculator
    from .edit_file import edit_file
    from .grep import grep
    from .list_dir import list_dir
    from .read_file import read_file
    from .write_file import write_file

    return [calculator, read_file, write_file, grep, list_dir, bash, edit_file]
