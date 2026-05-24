"""Public API for `avior.core`.

Re-exports the core runtime primitives so callers can write
`from avior.core import Agent, Message, ModelSettings, Provider` without
reaching into individual modules.  Exceptions live in `avior.core.exceptions`
and are imported from there explicitly.  Submodules also remain importable
directly (e.g., `from avior.core.messages import Role`).
"""

from avior.core import _logging as _logging  # noqa: F401  (logging side effect)
from avior.core.agent import Agent
from avior.core.messages import Message, Part, Role, StopReason, TextPart
from avior.core.provider import ModelSettings, Provider
from avior.core.runner import Runner

__all__ = [
    "Agent",
    "Message",
    "ModelSettings",
    "Part",
    "Provider",
    "Role",
    "Runner",
    "StopReason",
    "TextPart",
]
