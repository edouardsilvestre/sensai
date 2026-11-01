# EV1 Automated Evaluation & Hallucination Detection: user stories

## Catch made-up claims before publishing

As an investigative journalist checking Sensai's answers against leaked records, I want each answer to be checked sentence by sentence against the retrieved excerpts and tool results, so that invented amounts, dates or names are flagged before I quote them.

Acceptance criteria:
- With `/feature fact_check on` (or `SENSAI_FEATURE_FACT_CHECK=true`), each plain answer is verified right after being streamed, without delaying the answer itself.
- Amounts, dates and proper names are extracted by code and compared against the excerpts first: an amount or a date absent from the excerpts is flagged immediately as `not in the excerpts`, without calling the model.
- Remaining sentences are judged individually by `SENSAI_JUDGE_MODEL` (defaulting to the chat model, with `qwen3:1.7b` recommended) with a strict JSON schema (`reason`, `verdict`).
- The check prints a summary with the count of supported sentences, faithfulness percentage, relevance and coherence scores (1 to 5), followed by non-supported claims categorized clearly (`not in the excerpts`, `contradicted`, `to verify`, or `unchecked`).
- Answers from `/json`, `/review`, answers stopped by guardrails, cached answers, or empty answers are never checked.

## Benchmark and compare Sensai versions on a test set

As an investigative editor evaluating model upgrades, prompt tweaks, or new RAG parameters, I want to run Sensai on a standardized list of questions and compare the scores against previous runs, so that I know whether a change genuinely improves faithfulness and relevance.

Acceptance criteria:
- `python -m sensai eval questions.json` loads questions from a JSON file with `{"cases": [{"question": "...", "expected": [...]}]}` and runs each with a fresh, isolated agent (cache, living document, examples, and interactive approvals disabled).
- Optional `expected` facts in each test case are verified in the answer using normalized matching (handling different representations of amounts and dates).
- `--out results.json` exports all answers, individual sentence claims, expected facts matched, run duration, configuration, and aggregate summaries to a JSON file.
- `--compare previous.json` compares the current run against a previous result file, displaying delta changes (+/-) for expected facts percentage, faithfulness, relevance, coherence, latency per question, and individual question shifts.
- An invalid or unreadable questions file reports a clear error message and exits with status 1.

## Keep evaluations bounded and understand false alarms

As a journalist running Sensai on a standard laptop CPU without a GPU, I want checks to complete within a predictable time without hanging, so that an over-reflective model never blocks my investigation.

Acceptance criteria:
- `SENSAI_JUDGE_MAX_TOKENS` (default 1500, range 200-8000) caps the tokens the judge model may output per sentence, thinking included.
- If the judge exceeds the token limit or fails to produce valid JSON, the sentence receives the `unchecked` verdict with the error reason instead of hanging, and the remaining sentences are checked.
- Faithfulness is calculated strictly on checked sentences, and relevance/coherence scores remain optional (reported as `-` when unavailable).
- During live chat, `Ctrl+C` cleanly skips the ongoing check without losing the generated answer.
