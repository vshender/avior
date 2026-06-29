"""Public API for `avior.core`.

Re-exports the core runtime primitives so callers can write
`from avior.core import Agent, UserMessage, ModelSettings, Provider` without
reaching into individual modules.  Exceptions live in `avior.core.exceptions`
and are imported from there explicitly.  Submodules also remain importable
directly (e.g., `from avior.core.messages import StopReason`).
"""

from avior.core import _logging as _logging  # noqa: F401  (logging side effect)
from avior.core.agent import Agent
from avior.core.context import RunContext
from avior.core.messages import (
    AssistantMessage,
    Message,
    Part,
    StopReason,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolMessage,
    ToolResult,
    ToolResultError,
    ToolResultOk,
    ToolResultPart,
    UserMessage,
)
from avior.core.provider import (
    ModelCapabilities,
    ModelSettings,
    Provider,
    ProviderResponse,
)
from avior.core.result import RunResult
from avior.core.runner import Runner
from avior.core.tools import Tool, tool
from avior.core.usage import Usage
from avior.core.warnings import RunWarning, WarningHandler

__all__ = [
    "Agent",
    "AssistantMessage",
    "Message",
    "ModelCapabilities",
    "ModelSettings",
    "Part",
    "Provider",
    "ProviderResponse",
    "RunContext",
    "RunResult",
    "RunWarning",
    "Runner",
    "StopReason",
    "TextPart",
    "ThinkingPart",
    "Tool",
    "ToolCallPart",
    "ToolMessage",
    "ToolResult",
    "ToolResultError",
    "ToolResultOk",
    "ToolResultPart",
    "Usage",
    "UserMessage",
    "WarningHandler",
    "tool",
]
