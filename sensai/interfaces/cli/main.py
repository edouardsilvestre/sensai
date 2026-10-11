"""Command-line interface: renders agent events in the terminal."""

import argparse
import sys
from pathlib import Path

from sensai.core.agent import Agent
from sensai.core.config import ConfigError, load_config
from sensai.core.llm.ollama import LLMError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sensai", description="Sensai CLI chatbot")
    parser.add_argument("-e", "--env-file", dest="env_path", type=Path, help="Path to a .env file (default: ./.env)")
    return parser.parse_args(argv)


def error(message: str) -> None:
    print(f"[error] {message}", file=sys.stderr)



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.env_file)
    except ConfigError as e:
        error(str(e))
        return 1

    agent = Agent(config)
    try:
        agent.client.ensure_model(config.model)
    except LLMError as e:
        error(str(e))
        return 1



if __name__ == "__main__":
    sys.exit(main())
