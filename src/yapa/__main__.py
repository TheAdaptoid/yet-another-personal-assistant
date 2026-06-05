"""YAPA Main Module."""

from yapa.cli import cli
from yapa.database import init_db

if __name__ == "__main__":
    init_db()
    cli()
