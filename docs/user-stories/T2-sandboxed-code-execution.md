# T2 Sandboxed Code Execution: user stories

## Get exact figures instead of estimates

As an investigative journalist, I want Sensai to compute totals, percentages and conversions by running code, so that the figures in my article are exact and not guessed by the model.

Acceptance criteria:
- With `SENSAI_CODE_EXEC=true`, asking "Combien font 17 % de 48 250 € ? Calcule-le." makes the model call `run_python`, and the CLI prints the code before it runs.
- The CLI then prints a short result: `ok`, the error (`exit 1: ZeroDivisionError`) or `stopped after 10s`.
- The answer uses the printed output of the program.

## Let the model fix its own code

As a journalist, I want the model to correct its code when it fails, so that I get a result without debugging Python myself.

Acceptance criteria:
- When the code raises an exception, the model receives the traceback and calls `run_python` again with fixed code.
- The model may retry up to `SENSAI_MAX_TOOL_ROUNDS` rounds (5 by default), then it has to answer.
- A program that runs longer than `SENSAI_CODE_TIMEOUT` seconds is stopped, and the model is told so.
- Very long outputs are cut, keeping their end so that the error stays visible.

## Run generated code without risk for the investigation

As a journalist protecting a source, I want the generated code to run isolated from my machine, so that a wrong or malicious program cannot read my files or send data out.

Acceptance criteria:
- Code execution is off unless `SENSAI_CODE_EXEC=true`; `/tools off` disables it for the rest of the session.
- The code runs in a new Docker container that is removed afterwards, without network, with limited memory, CPU and processes, a read-only disk and no host folder mounted.
- If Docker is not installed or not running, Sensai warns at startup and keeps working without code execution; if the image is missing, it downloads it first.
