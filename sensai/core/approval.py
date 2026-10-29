from collections.abc import Callable
from dataclasses import dataclass

from sensai.core.llm.ollama import ToolCall
from sensai.core.tools.registry import ToolRegistry, ToolResult

ALL_TOOLS_REASON = "SENSAI_APPROVAL=all asks before every tool call"
NO_APPROVER_REASON = "no user is available to approve it"


@dataclass
class Approval:
    approved: bool
    reason: str = ""


Approver = Callable[[ToolCall, str], Approval]


def approval_reason(mode: str, tools: ToolRegistry, call: ToolCall) -> str | None:
    if mode == "off" or call.name not in tools.names():
        return None
    reason = tools.approval_reason(call.name)
    if mode == "all":
        return reason or ALL_TOOLS_REASON
    return reason


def refusal(approval: Approval) -> ToolResult:
    because = f": {approval.reason}" if approval.reason else ""
    return ToolResult(
        f"The user refused this tool call{because}. Do not call it again with the same arguments. "
        "Tell the user you could not use it, and do not make up what it would have found: "
        "only state what you already know for sure, or ask the user how to proceed.",
        summary="refused",
    )
