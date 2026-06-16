"""Agent definition."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Generic

from typing_extensions import TypeVar

from avior.core.exceptions import ConfigurationError
from avior.core.provider import ModelSettings
from avior.core.tools import Tool

# `Agent` is *invariant* in `Deps`: the `TypeVar` carries no variance flag, so
# it defaults to invariant - and the usage requires it anyway.  `Deps` appears
# both covariantly (in `deps_type: type[Deps]`) and contravariantly (in
# `tools: Sequence[Tool[Any, Any, Deps]]`, since `Tool` is contravariant in its
# deps); a parameter used in both positions can only be invariant.  Explicit
# `TypeVar` because Python 3.12 lacks `class Agent[Deps = None]` syntax.
Deps = TypeVar("Deps", default=None)


@dataclass(frozen=True, kw_only=True)
class Agent(Generic[Deps]):
    """A declarative definition of agent behavior that a `Runner` drives.

    `Deps` is the type of the dependencies the agent's tools read through their
    `RunContext`.  The agent only declares this type (a contract for
    type-checking); it never holds the dependencies themselves - the caller
    supplies them per run via `Runner.run(..., deps=...)`.  Declare the type by
    passing `deps_type`; left off, `Deps` is inferred from the tools - `object`
    when they need no deps, `None` when there are no tools.
    """

    instructions: str | None = None
    """System instructions for the model.

    `Runner` passes them to the provider as the system prompt on each model
    call.  Blank instructions - `None`, empty, or whitespace-only - run the
    model with no system prompt, since they convey nothing.
    """

    tools: Sequence[Tool[Any, Any, Deps]] = field(default_factory=tuple)
    """Tools the model may call during a run.  Empty by default.

    Typed at the agent's own `Deps`: a tool belongs here only if the agent's
    dependencies satisfy what the tool requires.  Because `Tool` is
    contravariant in its deps type, a tool that requires `Deps` itself, any
    supertype of it, or nothing at all (`object`) fits; a tool requiring some
    unrelated or more specific type does not.

    Snapshotted to a tuple at construction, so the agent does not alias the
    caller's sequence.
    """

    model_settings: ModelSettings
    """The model invocation settings used for every model call."""

    deps_type: type[Deps] | None = None
    """The type of the dependencies the agent's tools expect; you supply the
    actual value separately, on each run.

    Seeds the `Deps` type parameter: absent an explicit annotation,
    `Agent(deps_type=MyDeps)` infers `Agent[MyDeps]`.  Left as `None`, `Deps` is
    inferred from the tools: `object` when none need deps, `None` when there are
    no tools.  When any tool needs deps, pass `deps_type` explicitly rather than
    relying on inference - a required type is not inferred reliably:
    basedpyright infers it, but mypy falls back to `None` for more than one
    tool.

    `deps_type` is also the only place the agent keeps the deps type at runtime,
    where `Runner.run` reads it as a safety net for callers that skip type
    checking.  When `deps_type` is set and `deps` is missing, the run stops at
    once with a clear error, instead of a tool getting `deps=None` and failing
    later, deep in the run, in a way that is hard to trace; the check looks only
    for a missing `deps`, not a wrong type.  Relying on inference leaves this
    net blind even where inference works: the inferred type lives only in the
    type checker while `deps_type` stays `None`, so the check never fires, and
    an untyped caller could run the agent with no `deps`.

    Keep any explicit `Agent[...]` annotation equal to `deps_type`, or omit it.
    `deps_type` does not pin `Deps` against an annotation: `type[...]` is
    covariant, so `Agent[object](deps_type=MyDeps)` type-checks and the
    annotation wins - the agent is `Agent[object]`, not `Agent[MyDeps]`.  It is
    then statically deps-agnostic - `ctx.deps` is `object`, a `MyDeps`-requiring
    tool no longer fits, and `run` needs no `deps` - so `deps_type` is ignored
    by the type system, surviving only as the runtime record (which then
    disagrees with the static type).
    """

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
