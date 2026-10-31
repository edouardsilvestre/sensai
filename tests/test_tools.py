from ollama._types import Tool
from ollama._utils import convert_function_to_tool

from sensai.core.llm.ollama import ToolCall
from sensai.core.tools.catalog import build_registry
from sensai.core.tools.registry import SchemaTool, ToolError, ToolRegistry, ToolResult, requires_approval


def echo(text: str, times: int = 1) -> str:
    """
    Repeat the text.

    Args:
      text: The text to repeat
      times: How many times

    Returns:
      The repeated text
    """
    if not text:
        raise ToolError("text is empty")
    return text * times


@requires_approval("the files are deleted for good")
def delete_file(path: str) -> str:
    """
    Delete a file.

    Args:
      path: The file to delete

    Returns:
      What was deleted
    """
    return f"deleted {path}"


def test_registry_lists_its_tools_by_function_name():
    registry = ToolRegistry([echo])
    assert registry.names() == ["echo"]
    assert registry.tools() == [echo]
    assert registry and not ToolRegistry()


def test_execute_runs_the_tool():
    assert ToolRegistry([echo]).execute(ToolCall("echo", {"text": "ab", "times": 2})) == ToolResult("abab")


def test_a_tool_can_return_a_summary_for_the_user():
    def count(text: str) -> ToolResult:
        """
        Count the words.

        Args:
          text: The text

        Returns:
          The number of words
        """
        return ToolResult(str(len(text.split())), summary="counted")

    assert ToolRegistry([count]).execute(ToolCall("count", {"text": "a b"})) == ToolResult("2", summary="counted")


def test_unknown_tool_is_reported_to_the_model():
    result = ToolRegistry([echo]).execute(ToolCall("delete_files", {}))
    assert result.error and "Unknown tool 'delete_files'" in result.content and "echo" in result.content


def test_missing_and_unexpected_arguments_are_reported():
    registry = ToolRegistry([echo])
    missing = registry.execute(ToolCall("echo", {}))
    unexpected = registry.execute(ToolCall("echo", {"text": "a", "colour": "red"}))
    assert missing.error and "'text'" in missing.content
    assert unexpected.error and "'colour'" in unexpected.content


def test_tool_errors_are_reported_instead_of_raised():
    assert ToolRegistry([echo]).execute(ToolCall("echo", {"text": ""})) == ToolResult("Error: text is empty", error=True)


class Setup:
    def __init__(self, error: str = ""):
        self.error = error
        self.prepared = False

    def prepare(self) -> None:
        self.prepared = True
        if self.error:
            raise ToolError(self.error)

    def check(self) -> str:
        """
        Check.

        Returns:
          ok
        """
        return "ok"


def test_prepare_runs_the_setup_of_each_tool():
    setup = Setup()
    registry = ToolRegistry([echo, setup.check])
    assert registry.prepare() == []
    assert setup.prepared and registry.names() == ["echo", "check"]


def test_a_tool_whose_setup_fails_is_removed():
    registry = ToolRegistry([echo, Setup("Docker is not installed").check])
    assert registry.prepare() == ["check disabled: Docker is not installed"]
    assert registry.names() == ["echo"]


def test_only_the_memory_is_enabled_by_default(config, client):
    assert build_registry(config, client).names() == [
        "memory_search", "memory_add_fact", "memory_update_fact",
        "memory_delete_fact", "memory_link", "memory_unlink",
    ]
    assert build_registry(config, client).approval_reason("memory_add_fact") is None


def test_web_search_is_enabled_by_its_setting(config, client):
    config.features["web_search"] = True
    config.features["memory"] = False
    assert build_registry(config, client).names() == ["web_search"]


def test_tools_feature_disables_every_tool(config, client):
    config.features["web_search"] = True
    config.features["tools"] = False
    assert not build_registry(config, client)


def test_a_tool_can_require_the_users_approval():
    registry = ToolRegistry([echo, delete_file])
    assert registry.approval_reason("delete_file") == "the files are deleted for good"
    assert registry.approval_reason("echo") is None
    assert registry.approval_reason("format_disk") is None


def test_an_approved_tool_still_runs_and_keeps_its_schema():
    result = ToolRegistry([delete_file]).execute(ToolCall("delete_file", {"path": "a.txt"}))
    assert result == ToolResult("deleted a.txt")
    function = convert_function_to_tool(delete_file).function
    assert function.name == "delete_file"
    assert function.description == "Delete a file."
    assert list(function.parameters.properties) == ["path"]


def test_web_search_requires_approval(config, client):
    config.features["web_search"] = True
    registry = build_registry(config, client)
    assert registry.approval_reason("web_search") == "the query is sent to an external search engine"
    assert convert_function_to_tool(registry.tools()[0]).function.name == "web_search"
def test_code_execution_is_enabled_by_its_setting(config, client):
    config.features["code_exec"] = True
    config.features["memory"] = False
    assert build_registry(config, client).names() == ["run_python"]


PARAMETERS = {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}


def add(arguments: dict) -> str:
    if "a" not in arguments or "b" not in arguments:
        raise ToolError("add: a and b are required")
    return str(arguments["a"] + arguments["b"])


def schema_add(**kwargs) -> SchemaTool:
    return SchemaTool("add", "Add two numbers.\n\nUse it for any addition.", PARAMETERS, add, **kwargs)


def test_a_schema_tool_is_listed_by_its_name_next_to_the_functions():
    registry = ToolRegistry([echo, schema_add()])
    assert registry.names() == ["echo", "add"]


def test_ollama_gets_the_schema_of_a_schema_tool():
    tools = ToolRegistry([echo, schema_add()]).tools()
    assert tools[0] == echo
    function = Tool.model_validate(tools[1]).function
    assert (function.name, function.description) == ("add", "Add two numbers.\n\nUse it for any addition.")
    assert function.parameters.required == ["a", "b"]


def test_a_schema_tool_gets_the_arguments_as_they_are():
    registry = ToolRegistry([schema_add()])
    assert registry.execute(ToolCall("add", {"a": 2, "b": 3})) == ToolResult("5")
    # No check against a Python signature: the tool checks its arguments itself.
    assert registry.execute(ToolCall("add", {"a": 2})) == ToolResult("Error: add: a and b are required", error=True)


def test_a_schema_tool_can_require_the_users_approval():
    registry = ToolRegistry([schema_add(approval_reason="the numbers are sent to a server")])
    assert registry.approval_reason("add") == "the numbers are sent to a server"
    assert ToolRegistry([schema_add()]).approval_reason("add") is None


def test_descriptions_keep_the_first_paragraph():
    assert ToolRegistry([echo, schema_add()]).descriptions() == {"echo": "Repeat the text.", "add": "Add two numbers."}


def test_extended_adds_tools_and_keeps_the_registry_as_it_is():
    registry = ToolRegistry([echo], warnings=["calculator is down"])
    extended = registry.extended([schema_add()])
    assert extended.names() == ["echo", "add"] and extended.warnings == ["calculator is down"]
    assert registry.names() == ["echo"]


def untyped(note, pages: list, ratio: float = 0.5, strict: bool = False) -> str:
    return note


def test_the_arguments_schema_comes_from_the_signature_or_the_schema_tool():
    schemas = ToolRegistry([echo, untyped, schema_add()]).arguments_schemas()
    assert schemas["echo"] == {
        "type": "object", "properties": {"text": {"type": "string"}, "times": {"type": "integer"}}, "required": ["text"]
    }
    assert schemas["untyped"]["properties"] == {
        "note": {}, "pages": {"type": "array"}, "ratio": {"type": "number"}, "strict": {"type": "boolean"}
    }
    assert schemas["untyped"]["required"] == ["note", "pages"]
    assert schemas["add"] == PARAMETERS


def test_parameters_lists_the_arguments_of_each_tool():
    assert ToolRegistry([echo, schema_add()]).parameters() == {"echo": ["text", "times"], "add": ["a", "b"]}
