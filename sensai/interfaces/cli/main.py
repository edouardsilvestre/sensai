import argparse
import sys
from pathlib import Path

from sensai.core.config import ConfigError, load_config
from sensai.interfaces.cli.chat import run_chat
from sensai.interfaces.cli.output import error
from sensai.interfaces.cli.runs.attack import run_attack
from sensai.interfaces.cli.runs.evaluation import run_eval_command, run_retrieval_command
from sensai.interfaces.cli.runs.index import run_ingest, run_status
from sensai.interfaces.cli.runs.scheduler import run_schedule
from sensai.interfaces.cli.runs.stats import run_stats

DEFAULT_WEB_PORT = 8000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sensai", description="Sensai CLI chatbot")
    parser.add_argument("-e", "--env-file", dest="env_path", type=Path, help="Path to a .env file (default: ./.env)")
    parser.add_argument(
        "-r", "--resume", nargs="?", const="last", metavar="ID",
        help="continue a saved conversation (the last one without ID)",
    )
    parser.add_argument("-p", "--persona", help="persona to start with (overrides SENSAI_PERSONA)")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    commands.add_parser("chat", help="chat with the model (default)")
    ingest_parser = commands.add_parser("ingest", help="index documents for search")
    ingest_parser.add_argument("paths", nargs="+", type=Path, metavar="PATH", help="files or folders to index")
    ingest_parser.add_argument(
        "--rebuild", action="store_true", help="empty the index first (needed after changing the embedding model or updating Sensai's index format)"
    )
    commands.add_parser("status", help="show what the document index contains")
    attack_parser = commands.add_parser("attack", help="run adversarial attack suite against Sensai")
    attack_parser.add_argument(
        "cases", nargs="?", type=Path, default=None, help="path to custom attack suite JSON file"
    )
    attack_parser.add_argument(
        "--category", choices=["jailbreak", "injection", "malformed"], help="filter attacks by category"
    )
    attack_parser.add_argument("--out", type=Path, default=None, help="path to save the attack report JSON file")
    attack_parser.add_argument(
        "--compare", type=Path, default=None, help="path to earlier attack report JSON to compare with"
    )
    stats_parser = commands.add_parser("stats", help="show the metrics of the messages logged")
    stats_parser.add_argument("--days", type=int, default=7, metavar="N", help="the last N days (default: 7)")
    eval_parser = commands.add_parser("eval", help="answer a set of questions and score the answers")
    eval_parser.add_argument("cases", type=Path, metavar="FILE", help='the questions: {"cases": [{"question": ...}]}')
    eval_parser.add_argument("--out", type=Path, metavar="FILE", help="write the answers and their scores there")
    eval_parser.add_argument("--compare", type=Path, metavar="FILE", help="compare with results written before")
    eval_parser.add_argument(
        "--retrieval", action="store_true",
        help="score the document search alone, step by step, against the sources of each question (no chat model)",
    )
    schedule_parser = commands.add_parser("schedule", help="run the scheduled tasks when they are due")
    schedule_parser.add_argument("--once", action="store_true", help="run what is due and exit, for cron or the like")
    web_parser = commands.add_parser("web", help="chat in a browser page")
    web_parser.add_argument("--port", type=int, default=DEFAULT_WEB_PORT, help=f"default: {DEFAULT_WEB_PORT}")
    publish_parser = commands.add_parser("publish", help="serve the pages published with /share to whoever has their link")
    publish_parser.add_argument("--host", default="0.0.0.0", help="the address to listen on (default: all of them)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.env_path)
    except ConfigError as e:
        error(str(e))
        return 1
    if args.persona:
        config.persona = args.persona

    if args.command == "ingest":
        return run_ingest(config, args.paths, args.rebuild)
    if args.command == "status":
        return run_status(config)
    if args.command == "attack":
        return run_attack(config, args)
    if args.command == "stats":
        return run_stats(config, args.days)
    if args.command == "eval" and args.retrieval:
        if args.out or args.compare:
            error("--retrieval prints its scores: it cannot be used with --out or --compare")
            return 1
        return run_retrieval_command(config, args.cases)
    if args.command == "eval":
        return run_eval_command(config, args.cases, args.out, args.compare)
    if args.command == "schedule":
        return run_schedule(config, args.once)
    if args.command == "web":
        # Here: only the page needs the web server and its libraries.
        from sensai.interfaces.web.server import run_web

        return run_web(config, args.port)
    if args.command == "publish":
        if not config.features["sharing"]:
            error("The sharing feature is off: SENSAI_FEATURE_SHARING=true to serve the public pages.")
            return 1
        from sensai.interfaces.web.public import run_public

        return run_public(config, args.host)
    return run_chat(config, args.resume)


if __name__ == "__main__":
    sys.exit(main())
