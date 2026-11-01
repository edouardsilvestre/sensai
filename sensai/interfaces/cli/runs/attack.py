import argparse
from pathlib import Path

from sensai.core.adversarial.models import AdversarialError, AttackResult, load_suite
from sensai.core.adversarial.runner import (
    AttackSummary,
    load_attack_report,
    run_attack_suite,
    save_attack_report,
    summarize_attacks,
)
from sensai.core.agent import Agent
from sensai.core.config import Config
from sensai.core.guardrails.rules import GuardrailError
from sensai.core.llm.ollama import LLMError, OllamaClient
from sensai.interfaces.cli.output import error, plural, warning
from sensai.interfaces.cli.startup import load_guardrails


def print_attack_summary(summary: AttackSummary) -> None:
    print(f"\n{'Category':<15}{'Total':<8}{'Defended':<10}{'Breached':<10}{'Rate':<8}")
    print("-" * 51)
    for cat, cat_sum in summary.by_category.items():
        rate_str = f"{cat_sum.defense_rate * 100:.1f}%"
        print(f"{cat:<15}{cat_sum.total:<8}{cat_sum.defended:<10}{cat_sum.breached:<10}{rate_str:<8}")
    print("-" * 51)
    overall_rate = f"{summary.defense_rate * 100:.1f}%"
    print(f"{'Overall':<15}{summary.total:<8}{summary.defended:<10}{summary.breached:<10}{overall_rate:<8}")


def print_attack_comparison(summary: AttackSummary, previous: dict, compare_path: Path) -> None:
    prev_summary = previous.get("summary", {})
    prev_rate = prev_summary.get("defense_rate", 0.0)
    diff_rate = (summary.defense_rate - prev_rate) * 100
    sign = "+" if diff_rate >= 0 else ""
    print(f"\nComparison with {compare_path}:")
    print(f"  Previous defense rate : {prev_rate * 100:.1f}%")
    print(f"  Current defense rate  : {summary.defense_rate * 100:.1f}% ({sign}{diff_rate:.1f}%)")


def run_attack(config: Config, args: argparse.Namespace) -> int:
    try:
        cases = load_suite(args.cases)
    except AdversarialError as e:
        error(str(e))
        return 1

    if args.category:
        cases = [c for c in cases if c.category == args.category]
        if not cases:
            error(f"No attack cases found for category '{args.category}'")
            return 1

    previous_report = None
    if args.compare:
        try:
            previous_report, _ = load_attack_report(args.compare)
        except AdversarialError as e:
            error(str(e))
            return 1

    client = OllamaClient(config.host, config.ollama_api_key)
    try:
        client.ensure_model(config.model)
    except LLMError as e:
        error(str(e))
        return 1

    try:
        guardrails = load_guardrails(config, client)
    except GuardrailError as e:
        error(str(e))
        return 1

    def make_agent() -> Agent:
        return Agent(config, client=client, guardrails=guardrails)

    print(f"Running {plural(len(cases), 'adversarial attack')} against {config.model}...\n")

    def on_case(number: int, result: AttackResult) -> None:
        status_tag = f"[{result.status.upper()}]"
        pad = 11 - len(status_tag)
        print(f"[{number:2d}/{len(cases)}] {status_tag}{' ' * pad} ({result.case.category}) {result.case.name}")
        if result.status == "breached":
            print(f"         Reason: {result.reason}")

    results = run_attack_suite(make_agent, cases, on_case=on_case)
    summary = summarize_attacks(results)

    print_attack_summary(summary)

    if previous_report:
        print_attack_comparison(summary, previous_report, args.compare)

    if args.out:
        try:
            save_attack_report(args.out, config, results)
            print(f"\nAttack report saved to {args.out}")
        except AdversarialError as e:
            warning(str(e))

    return 0
