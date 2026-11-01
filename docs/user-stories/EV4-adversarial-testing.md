# EV4 Adversarial Testing: user stories

## Audit resilience against adversarial attacks

As a newsroom security officer or investigative lead, I want to run automated adversarial attack suites against Sensai,
so that I know whether the assistant withstands jailbreak attempts, prompt injections and malformed inputs before
journalists rely on it with confidential sources.

Acceptance criteria:
- `python -m sensai attack` loads a benchmark suite of adversarial attacks across categories (`jailbreak`,
  `injection`, `malformed`) and executes them against the agent.
- A jailbreak attempt trying to bypass safety guidelines or system constraints is defended (either refused by
  guardrails or properly declined by the model).
- A prompt injection attempting to leak confidential instructions or override system personas is defended without
  revealing internal instructions or protected identities.
- A malformed input (massive token padding, deeply nested structures, control characters or delimiter abuse) is safely
  handled without unhandled exceptions or crashes.
- Running `python -m sensai attack --category jailbreak` executes only attacks belonging to the specified category.
- When passing a custom suite file `python -m sensai attack /path/to/attacks.json`, the custom cases are loaded and
  evaluated instead of the built-in benchmark.

## Track defense rates and export audit reports

As a newsroom technical auditor, I want detailed defense rate metrics and machine-readable audit reports, so that I can
quantify our security posture and archive evaluation results for compliance reviews.

Acceptance criteria:
- At the end of an attack suite run, the CLI displays a summary table with columns for category, total cases,
  defended cases, breached cases, and defense percentage rate, followed by an overall row.
- The `--out <path>` flag exports the full audit report as a structured JSON file containing timestamps, evaluated
  model name, configuration parameters, summary metrics, and per-case results (status, breach reasons, response).
- Any breached case prints its identifier, category, and failure reason in the CLI output so auditors can inspect
  the vulnerability vector immediately.

## Detect regressions and compare defense improvements

As an engineer refining Sensai's guardrails and system prompts, I want to compare attack results against a previously
saved baseline audit report, so that I can immediately verify whether defense improvements worked or if any security
regression occurred.

Acceptance criteria:
- The `--compare <previous_report.json>` flag loads an earlier audit report and computes comparative metrics.
- The CLI displays the previous defense rate, the current defense rate, and the signed percentage difference
  (e.g., `+15.0%` or `-5.0%`).
- Missing or invalid comparison files display a clear warning or error message without crashing the evaluation runner.
