"""Public API for `avior.core`.

Re-exports the core runtime primitives so callers can write
`from avior.core import Agent, UserMessage, ModelSettings, Provider` without
reaching into individual modules.  Exceptions live in `avior.core.exceptions`
and are imported from there explicitly.  Submodules also remain importable
directly (e.g., `from avior.core.messages import StopReason`).
"""

from avior.core import _logging as _logging  # noqa: F401  (logging side effect)
from avior.core.agent import Agent
from avior.core.messages import (
    AssistantMessage,
    Message,
    Part,
    StopReason,
    SystemMessage,
    TextPart,
    UserMessage,
)
from avior.core.provider import ModelSettings, Provider, ProviderResponse
from avior.core.result import RunResult
from avior.core.runner import Runner
from avior.core.usage import Usage

__all__ = [
    "Agent",
    "AssistantMessage",
    "Message",
    "ModelSettings",
    "Part",
    "Provider",
    "ProviderResponse",
    "RunResult",
    "Runner",
    "StopReason",
    "SystemMessage",
    "TextPart",
    "Usage",
    "UserMessage",
]
