"""Command-line interface: renders agent events in the terminal."""

import argparse
import sys
from pathlib import Path

from sensai.core.agent import Agent, EmptyInputError
from sensai.core.config import ConfigError, load_config
from sensai.core.events import DoneEvent, TokenEvent
from sensai.core.llm.ollama import LLMError

PROMPT = "\n> "
EXIT_COMMANDS = {"/exit", "/quit"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sensai", description="Sensai CLI chatbot")
    parser.add_argument("-e", "--env-file", dest="env_path", type=Path, help="Path to a .env file (default: ./.env)")
    return parser.parse_args(argv)


def error(message: str) -> None:
    print(f"[error] {message}", file=sys.stderr)


def chat_turn(agent: Agent, line: str) -> None:
    """wrapper of agent run"""
    try:
        for event in agent.run(line):
            if isinstance(event, TokenEvent):
                print(event.text, end="", flush=True)
            elif isinstance(event, DoneEvent):
                print()
    except EmptyInputError:
        error("Empty message, please type something.")
    except LLMError as e:
        print()
        error(str(e))
    except KeyboardInterrupt:
        print("\n[interrupted]")


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

    print(f"Sensai — model: {config.model}. Type /exit or press Ctrl+D to quit.")
    while True:
        try:
            line = input(PROMPT)
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line.strip() in EXIT_COMMANDS:
            return 0
        chat_turn(agent, line)


if __name__ == "__main__":
    sys.exit(main())
