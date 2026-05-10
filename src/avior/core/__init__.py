"""Public API for `avior.core`.

Re-exports the user-facing primitives so callers can write
`from avior.core import Agent, Message, ModelSettings, Provider` without
reaching into individual modules. Sub-modules also remain importable
directly (e.g., `from avior.core.messages import Role`).
"""

from avior.core.agent import Agent
from avior.core.messages import Message, Part, Role, TextPart
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
    "TextPart",
]
