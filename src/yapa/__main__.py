"""YAPA Main Module."""

from yapa.cli import cli
from yapa.database import init_db


def main() -> None:
    """Entry point for the YAPA CLI."""
    init_db()
    cli()


if __name__ == "__main__":
    main()
