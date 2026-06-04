"""Agent definition."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from avior.core.exceptions import ConfigurationError
from avior.core.provider import ModelSettings, Provider
from avior.core.tools import Tool


@dataclass(frozen=True, kw_only=True)
class Agent:
    """Agent definition.

    Holds the static configuration that `Runner` uses to drive a conversation.
    """

    instructions: str
    """System instructions for the model, prepended as a `SystemMessage` before
    every model call by `Runner`.
    """

    tools: Sequence[Tool[Any, Any]] = field(default_factory=tuple)
    """Tools the model may call during a run.  Empty by default.

    Snapshotted to a tuple at construction, so the agent does not alias the
    caller's sequence.
    """

    provider: Provider
    """The `Provider` that performs this agent's model calls."""

    model_settings: ModelSettings
    """The model invocation settings used for every model call."""

    max_iter: int = 20
    """Maximum agent-loop iterations (one model call plus the tool calls it
    requests) before `Runner` raises `MaxIterationsExceeded`.  Overridable per
    call on `Runner.run`.
    """

    def __post_init__(self) -> None:
        """Normalize and validate the agent configuration.

        Snapshots `tools` into a tuple so the frozen agent does not alias the
        caller's sequence (and cannot be mutated through it).

        Tool calls identify a tool only by name, so names must be unique within
        an agent; duplicates would make dispatch ambiguous.  Raises
        `ConfigurationError` if two tools share a name.
        """

        # `frozen=True` blocks rebinding the field but not mutating a list it
        # holds; snapshot to a tuple so the config is truly immutable.  This is
        # the standard frozen-dataclass normalization idiom.
        object.__setattr__(self, "tools", tuple(self.tools))

        counts = Counter(tool.name for tool in self.tools)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            raise ConfigurationError(
                f"Duplicate tool names: {', '.join(duplicates)}.  Tool calls "
                "are dispatched by name, so each tool must have a unique name."
            )
