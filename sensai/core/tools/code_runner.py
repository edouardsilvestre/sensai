import subprocess
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from sensai.core.tools.registry import ToolError, ToolResult

# Characters kept from each output. The end is kept: it holds the last results and the traceback.
MAX_OUTPUT_CHARS = 2000

# `docker run` exits with 125 when the container could not start (daemon down, unknown image...).
DOCKER_FAILED = 125

_FIX_IT = "The program failed: read the traceback, fix the code and call run_python again."


@dataclass
class Execution:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


def docker_command(name: str, image: str) -> list[str]:
    """The code is read from stdin by `python -`, so nothing from the host is mounted in the container."""
    return [
        "docker", "run", "--rm", "-i", "--name", name,
        "--network", "none", "--memory", "256m", "--cpus", "1", "--pids-limit", "64",
        "--read-only", "--tmpfs", "/tmp:size=64m", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--user", "65534:65534",
        image, "python", "-",
    ]


def run_in_docker(code: str, image: str, timeout: int) -> Execution:
    name = f"sensai-{uuid.uuid4().hex[:12]}"
    try:
        done = subprocess.run(docker_command(name, image), input=code, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise ToolError("run_python: Docker is not installed") from None
    except subprocess.TimeoutExpired:
        # Stopping the docker client does not stop the container.
        subprocess.run(["docker", "kill", name], capture_output=True)
        return Execution("", "", exit_code=-1, timed_out=True)
    if done.returncode == DOCKER_FAILED:
        raise ToolError(f"run_python: the sandbox could not start: {done.stderr.strip()}")
    return Execution(done.stdout, done.stderr, done.returncode)


def has_image(image: str) -> bool:
    try:
        return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0
    except FileNotFoundError:
        raise ToolError("Docker is not installed") from None


def pull_image(image: str) -> None:
    pulled = subprocess.run(["docker", "pull", image], capture_output=True, text=True)
    if pulled.returncode != 0:
        raise ToolError(f"Cannot get the Docker image '{image}': {pulled.stderr.strip()}")


def _tail(text: str) -> str:
    text = text.rstrip()
    return text if len(text) <= MAX_OUTPUT_CHARS else "..." + text[-MAX_OUTPUT_CHARS:]


def _summary(run: Execution, timeout: int) -> str:
    if run.timed_out:
        return f"stopped after {timeout}s"
    if run.exit_code == 0:
        return "ok"
    last_line = run.stderr.strip().splitlines()[-1:]
    return f"exit {run.exit_code}" + (f": {last_line[0].split(':')[0]}" if last_line else "")


class CodeRunner:
    """Holds the settings of the `run_python` tool; the bound method is what the model calls."""

    def __init__(self, image: str, timeout: int, execute: Callable[[str, str, int], Execution] = run_in_docker):
        self.image = image
        self.timeout = timeout
        self.execute = execute

    def prepare(self) -> None:
        """Checks that Docker runs, and downloads the image now if it is missing: the first run would time out."""
        if not has_image(self.image):
            pull_image(self.image)

    def run_python(self, code: str) -> ToolResult:
        """
        Run a Python 3 program in a sandbox and see what it prints. Use it to compute, analyse data
        or check an idea instead of guessing. The sandbox has no network and no access to the user's files.
        The program cannot read input(): write the values in the code.

        Args:
          code: The full Python program to run, which prints what you need to see

        Returns:
          The exit code, the output and the errors (traceback) of the program
        """
        code = str(code)
        if not code.strip():
            raise ToolError("run_python: the code is empty")
        run = self.execute(code, self.image, self.timeout)
        lines = [f'<execution exit_code="{run.exit_code}">']
        if run.stdout:
            lines += ["<stdout>", _tail(run.stdout), "</stdout>"]
        if run.stderr:
            lines += ["<stderr>", _tail(run.stderr), "</stderr>"]
        lines.append("</execution>")
        if run.timed_out:
            lines.append(f"The program was stopped after {self.timeout} seconds: make it faster or do less work.")
        elif run.exit_code != 0:
            lines.append(_FIX_IT)
        return ToolResult("\n".join(lines), summary=_summary(run, self.timeout))
