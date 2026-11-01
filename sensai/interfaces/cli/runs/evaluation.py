from pathlib import Path

from sensai.core.agent import Agent
from sensai.core.config import Config
from sensai.core.evaluation.judge import Claim
from sensai.core.evaluation.retrieval import NDCG_K, evaluate_retrieval
from sensai.core.evaluation.runner import (
    CaseResult,
    EvalError,
    Summary,
    eval_config,
    load_cases,
    load_results,
    run_eval,
    save_results,
    summarize as summarize_eval,
)
from sensai.core.guardrails.rules import GuardrailError
from sensai.core.llm.ollama import LLMError, OllamaClient
from sensai.core.memory.store import MemoryStoreError
from sensai.core.persona import PersonaError
from sensai.core.rag.reranker import RerankError
from sensai.core.rag.store import StoreError
from sensai.core.tools.mcp import MCPConfigError
from sensai.interfaces.cli.answer import claim_label
from sensai.interfaces.cli.output import error, percent, plural, score, warning
from sensai.interfaces.cli.startup import load_existing_index, load_guardrails, load_persona, load_tools, open_index


def print_case(number: int, total: int, result: CaseResult) -> None:
    print(f"[{number}/{total}] {result.question}")
    if result.error:
        warning(result.error)
    if not result.answer:
        return
    expected = f"expected {len(result.found)}/{len(result.expected)}, " if result.expected else ""
    print(
        f"  {result.seconds:.1f} s, {expected}faithfulness {percent(result.faithfulness)}, "
        f"relevance {score(result.relevance)}, coherence {score(result.coherence)}"
    )
    for claim in result.flagged:
        print(f'  {claim_label(Claim(**claim))}: "{claim["text"]}"')
    missed = [fact for fact in result.expected if fact not in result.found]
    if missed:
        print(f"  missing from the answer: {', '.join(missed)}")


def print_summary(summary: Summary) -> None:
    seconds = "-" if summary.seconds is None else f"{summary.seconds:.1f} s"
    print(
        f"\n{plural(summary.cases, 'question')}: expected facts {percent(summary.expected)}, "
        f"faithfulness {percent(summary.faithfulness)}, relevance {score(summary.relevance)}, "
        f"coherence {score(summary.coherence)}, {plural(summary.flagged, 'sentence')} flagged, "
        f"{seconds} per answer, {plural(summary.errors, 'error')}."
    )


def change(name: str, before: float | None, after: float | None, unit: str) -> str:
    if before is None or after is None:
        return f"  {name}: {'-' if before is None else before} → {'-' if after is None else after}"
    if unit == "%":
        return f"  {name}: {before:.0%} → {after:.0%} ({(after - before) * 100:+.0f})"
    return f"  {name}: {before:.1f}{unit} → {after:.1f}{unit} ({after - before:+.1f})"


def print_comparison(path: Path, before: dict, old: list[CaseResult], results: list[CaseResult]) -> None:
    previous, current = summarize_eval(old), summarize_eval(results)
    print(f"\nCompared with {path} ({before.get('date', '?')[:10]}, {before.get('model', '?')}):")
    for name, unit in (("expected", "%"), ("faithfulness", "%"), ("relevance", "/5"), ("coherence", "/5")):
        print(change(name, getattr(previous, name), getattr(current, name), unit))
    print(change("seconds", previous.seconds, current.seconds, " s"))
    by_question = {result.question: result for result in old}
    for result in results:
        earlier = by_question.get(result.question)
        if earlier is not None and earlier.faithfulness != result.faithfulness:
            faithfulness = f"{percent(earlier.faithfulness)} → {percent(result.faithfulness)}"
            print(f"  {result.question}: faithfulness {faithfulness}")


def run_eval_command(config: Config, path: Path, out: Path | None, compare: Path | None) -> int:
    try:
        cases = load_cases(path)
        before, old = load_results(compare) if compare else ({}, [])
        _, persona = load_persona(config)
        client = OllamaClient(config.host, config.ollama_api_key)
        client.ensure_model(config.model)
        client.ensure_model(config.judge_model or config.model)
        tools = load_tools(config, client, asker=None)
        guardrails = load_guardrails(config, client)
    except (EvalError, PersonaError, LLMError, GuardrailError, MemoryStoreError, MCPConfigError) as e:
        error(str(e))
        return 1
    index = load_existing_index(config, client) if config.features["rag"] else None
    checked = eval_config(config)

    def make_agent() -> Agent:
        return Agent(checked, client, retriever=index, tools=tools, guardrails=guardrails, persona=persona)

    judge = config.judge_model or config.model
    print(f"Sensai eval: {plural(len(cases), 'question')} answered by {config.model}, judged by {judge}.")
    print("Each answer is judged sentence by sentence: this can take a few minutes per question.\n")
    try:
        results = run_eval(make_agent, checked, cases, lambda number, result: print_case(number, len(cases), result))
    except KeyboardInterrupt:
        print("\n[interrupted]")
        return 1
    finally:
        if index is not None:
            index.close()
    print_summary(summarize_eval(results))
    if compare:
        print_comparison(compare, before, old, results)
    if out:
        try:
            save_results(out, checked, results)
        except EvalError as e:
            error(str(e))
            return 1
        print(f"Results written to {out}: --compare {out} to compare a next run with them.")
    return 0


def run_retrieval_command(config: Config, path: Path) -> int:
    try:
        cases = load_cases(path)
    except EvalError as e:
        error(str(e))
        return 1
    if not config.index_path.exists():
        error(f"No index at {config.index_path}. Create one with: python -m sensai ingest PATH")
        return 1
    try:
        index = open_index(config, OllamaClient(config.host))
    except (LLMError, StoreError) as e:
        error(str(e))
        return 1
    answered = sum(bool(case.sources) for case in cases)
    print(
        f"Searching {plural(index.stats().documents, 'document')} for {plural(len(cases), 'question')} "
        f"({answered} answered by the documents)..."
    )
    try:
        scores = evaluate_retrieval(index, config, cases)
    except (LLMError, StoreError, RerankError) as e:
        error(str(e))
        return 1
    except KeyboardInterrupt:
        print("\n[interrupted]")
        return 1
    finally:
        index.close()
    print(f"\n{'':<16}  Recall@{config.rag_top_k}  NDCG@{NDCG_K}  Precision  Noise")
    for row in scores:
        ndcg = "-" if row.ndcg is None else f"{row.ndcg:.2f}"
        noise = "-" if row.noise is None else f"{row.noise:.1f}"
        current = "  ← SENSAI_RERANK_THRESHOLD" if row.name == f"+ threshold {config.rerank_threshold}%" else ""
        print(f"{row.name:<16}  {percent(row.recall):>8}  {ndcg:>7}  {percent(row.precision):>9}  {noise:>5}{current}")
    print("\nNoise: excerpts sent to the model for the questions the documents do not answer.")
    return 0
