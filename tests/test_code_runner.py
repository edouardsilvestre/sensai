import subprocess

import pytest

from sensai.core.tools import code_runner
from sensai.core.tools.code_runner import MAX_OUTPUT_CHARS, CodeRunner, Execution, has_image, pull_image, run_in_docker
from sensai.core.tools.registry import ToolError

TRACEBACK = 'Traceback (most recent call last):\n  File "<stdin>", line 1, in <module>\nZeroDivisionError: division by zero\n'


class FakeExecute:
    def __init__(self, execution: Execution):
        self.execution = execution
        self.calls: list[tuple[str, str, int]] = []

    def __call__(self, code: str, image: str, timeout: int) -> Execution:
        self.calls.append((code, image, timeout))
        return self.execution


def runner(execution: Execution) -> tuple[CodeRunner, FakeExecute]:
    execute = FakeExecute(execution)
    return CodeRunner("python:3.12-slim", 10, execute), execute


def test_the_docstring_describes_the_tool_to_the_model():
    from ollama._utils import convert_function_to_tool

    function = convert_function_to_tool(runner(Execution("", "", 0))[0].run_python).function
    assert function.name == "run_python" and "Run a Python 3 program" in function.description
    assert function.parameters.required == ["code"]


def test_successful_run_returns_the_output():
    tool, execute = runner(Execution("42\n", "", 0))
    result = tool.run_python("print(6 * 7)")
    assert execute.calls == [("print(6 * 7)", "python:3.12-slim", 10)]
    assert '<execution exit_code="0">' in result.content and "42" in result.content
    assert "fix the code" not in result.content
    assert result.summary == "ok"


def test_a_crash_is_a_result_with_the_traceback():
    tool, _ = runner(Execution("", TRACEBACK, 1))
    result = tool.run_python("1 / 0")
    assert "ZeroDivisionError: division by zero" in result.content
    assert "fix the code and call run_python again" in result.content
    assert result.summary == "exit 1: ZeroDivisionError"


def test_timeout_is_reported():
    tool, _ = runner(Execution("", "", -1, timed_out=True))
    result = tool.run_python("while True: pass")
    assert "stopped after 10 seconds" in result.content
    assert result.summary == "stopped after 10s"


def test_long_outputs_keep_their_end():
    tool, _ = runner(Execution("x" * MAX_OUTPUT_CHARS * 2, "y" * MAX_OUTPUT_CHARS + TRACEBACK, 1))
    content = tool.run_python("...").content
    assert len(content) < 2 * MAX_OUTPUT_CHARS + 500
    assert "ZeroDivisionError" in content


def test_empty_code_is_rejected():
    tool, execute = runner(Execution("", "", 0))
    with pytest.raises(ToolError):
        tool.run_python("  \n")
    assert execute.calls == []


class FakeRun:
    def __init__(self, result=None, error: Exception | None = None):
        self.result, self.error = result, error
        self.commands: list[list[str]] = []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        if self.error and command[1] in ("run", "image"):
            raise self.error
        return self.result


@pytest.fixture
def fake_run(monkeypatch):
    def install(**kwargs) -> FakeRun:
        fake = FakeRun(**kwargs)
        monkeypatch.setattr(code_runner.subprocess, "run", fake)
        return fake

    return install


def test_docker_runs_the_code_isolated(fake_run):
    fake = fake_run(result=subprocess.CompletedProcess([], 0, "42\n", ""))
    assert run_in_docker("print(42)", "python:3.12-slim", 10) == Execution("42\n", "", 0)
    command = fake.commands[0]
    for option in ("--rm", "--read-only", "--network", "none", "--memory", "--pids-limit", "--cap-drop"):
        assert option in command
    assert command[-3:] == ["python:3.12-slim", "python", "-"]


def test_docker_timeout_kills_the_container(fake_run):
    fake = fake_run(error=subprocess.TimeoutExpired("docker", 10))
    assert run_in_docker("while True: pass", "python:3.12-slim", 10).timed_out
    name = fake.commands[0][fake.commands[0].index("--name") + 1]
    assert fake.commands[1] == ["docker", "kill", name]


def test_missing_docker_is_a_tool_error(fake_run):
    fake_run(error=FileNotFoundError("docker"))
    with pytest.raises(ToolError, match="Docker is not installed"):
        run_in_docker("print(1)", "python:3.12-slim", 10)


def test_sandbox_start_failure_is_a_tool_error(fake_run):
    fake_run(result=subprocess.CompletedProcess([], 125, "", "Cannot connect to the Docker daemon\n"))
    with pytest.raises(ToolError, match="could not start: Cannot connect"):
        run_in_docker("print(1)", "python:3.12-slim", 10)


def test_has_image_asks_docker(fake_run):
    fake = fake_run(result=subprocess.CompletedProcess([], 1, b"", b""))
    assert not has_image("python:3.12-slim")
    assert fake.commands == [["docker", "image", "inspect", "python:3.12-slim"]]


def test_has_image_without_docker_is_a_tool_error(fake_run):
    fake_run(error=FileNotFoundError("docker"))
    with pytest.raises(ToolError, match="Docker is not installed"):
        has_image("python:3.12-slim")


def test_failed_pull_is_a_tool_error(fake_run):
    fake_run(result=subprocess.CompletedProcess([], 1, "", "manifest unknown\n"))
    with pytest.raises(ToolError, match="Cannot get the Docker image 'nope': manifest unknown"):
        pull_image("nope")


def test_prepare_downloads_a_missing_image(monkeypatch):
    pulled = []
    monkeypatch.setattr(code_runner, "has_image", lambda image: False)
    monkeypatch.setattr(code_runner, "pull_image", pulled.append)
    CodeRunner("python:3.12-slim", 10).prepare()
    assert pulled == ["python:3.12-slim"]


def test_prepare_keeps_a_present_image(monkeypatch):
    monkeypatch.setattr(code_runner, "has_image", lambda image: True)
    monkeypatch.setattr(code_runner, "pull_image", lambda image: pytest.fail("pulled"))
    CodeRunner("python:3.12-slim", 10).prepare()


def docker_is_ready() -> bool:
    try:
        return has_image("python:3.12-slim")
    except ToolError:
        return False


real_docker = pytest.mark.skipif(not docker_is_ready(), reason="needs Docker and the python:3.12-slim image")


@real_docker
def test_real_sandbox_runs_code_and_returns_tracebacks():
    tool = CodeRunner("python:3.12-slim", 20)
    assert "42" in tool.run_python("print(6 * 7)").content
    assert tool.run_python("1 / 0").summary == "exit 1: ZeroDivisionError"


@real_docker
def test_real_sandbox_has_no_network_and_a_read_only_disk():
    tool = CodeRunner("python:3.12-slim", 20)
    network = "import socket; socket.create_connection(('1.1.1.1', 80), timeout=2)"
    assert tool.run_python(network).summary == "exit 1: OSError"
    assert tool.run_python("open('/etc/x', 'w')").summary == "exit 1: OSError"
