"""Type-level checks for tool/agent deps compatibility.

Pins the rule:

    `Tool[..., D_tool]` fits `Agent[D_agent]` iff `D_agent` is assignable to
    `D_tool`.

Positive cases `assert_type` the exact `Agent[...]` the constructor produces; a
regression that lost `RunContext` covariance (hence `Tool` contravariance) would
make them stop type-checking.  `_r` instead uses an assignment to check `Result`
is covariant - a tool producing a subtype fits a slot for the supertype.

Most negative cases assert the variance at the `Tool` level (what the rule
reduces to); one (N4) asserts it at the `Agent` boundary, so a widening of the
`tools` field type is caught even if the `Tool`-level relation stayed intact.

`_mixed` documents where the checkers diverge: without `deps_type`, basedpyright
infers `Agent[_Sub]` across multiple deps-aware tools and accepts it, while mypy
cannot infer `Deps` across more than one tool and rejects the list - hence its
mypy-only suppression.
"""

from typing import Protocol, assert_type

from pydantic import BaseModel

from avior.core.agent import Agent
from avior.core.context import RunContext
from avior.core.provider import ModelSettings
from avior.core.tools import Tool

_MS = ModelSettings(model="test-model")


class _Args(BaseModel):
    """An arguments model for the fixture tools; its contents do not matter."""


class _Base:
    """A nominal base deps type for the subtype-relation cases."""


class _Sub(_Base):
    """A subclass of `_Base`: the narrower deps type."""


class _Unrelated:
    """A deps type unrelated to `_Base`, used by the rejection cases."""


class _HasName(Protocol):
    """A structural (protocol) deps type, matched by shape not inheritance."""

    name: str


class _Impl:
    """A concrete class that satisfies `_HasName` structurally."""

    name: str = "x"


class _Agnostic(Tool[_Args, str]):  # requires no deps -> Tool[_Args, str, object]
    """A tool that reads no deps; it should fit any agent."""

    name = "agnostic"
    description = ""
    args_model = _Args

    async def execute(self, ctx: RunContext[object], args: _Args) -> str:
        return ""


class _NeedsBase(Tool[_Args, str, _Base]):
    """A tool that requires `_Base` deps."""

    name = "needs_base"
    description = ""
    args_model = _Args

    async def execute(self, ctx: RunContext[_Base], args: _Args) -> str:
        return ""


class _NeedsSub(Tool[_Args, str, _Sub]):
    """A tool that requires the narrower `_Sub` deps."""

    name = "needs_sub"
    description = ""
    args_model = _Args

    async def execute(self, ctx: RunContext[_Sub], args: _Args) -> str:
        return ""


class _NeedsProto(Tool[_Args, str, _HasName]):
    """A tool that requires deps satisfying the `_HasName` protocol."""

    name = "needs_proto"
    description = ""
    args_model = _Args

    async def execute(self, ctx: RunContext[_HasName], args: _Args) -> str:
        return ctx.deps.name


class _ProducesSub(Tool[_Args, _Sub]):
    """A tool whose result type is `_Sub`; for the result-covariance check."""

    name = "produces_sub"
    description = ""
    args_model = _Args

    async def execute(self, ctx: RunContext[object], args: _Args) -> _Sub:
        return _Sub()


# (1) A deps-agnostic tool fits any agent.  With no `deps_type`, the agent is
# deps-agnostic: the tool requires nothing, so `Deps` binds to `object`.
assert_type(
    Agent(instructions="", model_settings=_MS, tools=[_Agnostic()]),
    Agent[object],
)
# `deps_type` drives the exact binding even with an agnostic tool present.
assert_type(
    Agent(instructions="", model_settings=_MS, tools=[_Agnostic()], deps_type=_Sub),
    Agent[_Sub],
)

# (2) A `_Base`-requiring tool fits `_Base` deps or any subclass.
assert_type(
    Agent(instructions="", model_settings=_MS, tools=[_NeedsBase()], deps_type=_Base),
    Agent[_Base],
)
assert_type(
    Agent(instructions="", model_settings=_MS, tools=[_NeedsBase()], deps_type=_Sub),
    Agent[_Sub],
)

# (3) A tool requiring a protocol fits an agent whose deps satisfies it.
assert_type(
    Agent(instructions="", model_settings=_MS, tools=[_NeedsProto()], deps_type=_Impl),
    Agent[_Impl],
)

# Mixed: a deps-agnostic and a `_Base`-requiring tool together, in an agent
# whose deps is the subclass `_Sub`.
assert_type(
    Agent(
        instructions="",
        model_settings=_MS,
        tools=[_Agnostic(), _NeedsBase()],
        deps_type=_Sub,
    ),
    Agent[_Sub],
)

# (4) `Result` is covariant: a tool producing `_Sub` fits a slot for `_Base`.
# A positive case - it must type-check; were `Result` invariant the assignment
# would be rejected (a real error here, no suppression to leave unused).
_r: Tool[_Args, _Base] = _ProducesSub()


# Negative cases: each assignment must be REJECTED.  The suppressions are the
# assertion - remove the type relationship and they become unnecessary, which
# fails CI (see the module docstring).

# (N1) A `_Base`-requiring tool does not fit an unrelated deps type: `_Base` is
# neither `_Unrelated` nor a supertype of it.
_n1: Tool[_Args, str, _Unrelated] = _NeedsBase()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

# (N2) Contravariance has a direction: a tool requiring the more specific `_Sub`
# does NOT fit a slot typed for the broader `_Base`.  An agent with `_Base` deps
# cannot supply the `_Sub` this tool reads.
_n2: Tool[_Args, str, _Base] = _NeedsSub()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

# (N3) A deps-requiring tool does not fit a no-deps (`None`) agent: a `None` run
# supplies no `_Base` for the tool to read.
_n3: Tool[_Args, str, None] = _NeedsBase()  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

# (N4) The same rule at the `Agent` boundary, where the tool list lives.  A
# `_Sub`-requiring tool does not fit an `Agent[_Base]`: the agent only promises
# its tools a `_Base`.  This guards the `tools` field type (`Sequence[Tool[Any,
# Any, Deps]]`) itself - if its deps parameter ever widened to `Any`, the
# `Tool`-level cases above would stay green but this one would start type-
# checking, and the unnecessary suppression would fail CI.  Kept on one physical
# line so both checkers' diagnostics land on the suppressed line (mypy flags the
# list item, basedpyright the inferred `Agent[_Sub]` against the annotation).
_n4: Agent[_Base] = Agent(instructions="", model_settings=_MS, tools=[_NeedsSub()])  # type: ignore[list-item]  # pyright: ignore[reportAssignmentType]


# Type-checker divergence (not a contract assertion).  Without `deps_type`,
# basedpyright infers `Agent[_Sub]` for a mixed set of deps-aware tools and
# accepts it; mypy cannot infer `Deps` across more than one tool, falls back to
# the default `None`, and rejects the list - hence the mypy-only suppression
# (basedpyright ignores `# type: ignore`, per `enableTypeIgnoreComments`).  This
# locks the limitation that motivates passing `deps_type` for deps-aware tools
# (see `Agent.deps_type`).
_mixed = Agent(instructions="", model_settings=_MS, tools=[_NeedsBase(), _NeedsSub()])  # type: ignore[list-item]


# `deps_type` does not pin `Deps` against an explicit annotation (not a contract
# assertion).  `type[...]` is covariant, so a concrete `deps_type` is accepted
# under a wider `Agent[...]` annotation, and the annotation wins: `Deps` is
# `object` here, not `_Sub`.  This cannot be forbidden portably while
# `deps_type` accepts a bare `type[Deps]` - Python has no exact-type constraint
# (see `Agent.deps_type`).  At runtime the guard follows `deps_type`, so running
# this agent without `deps` would raise `MissingDependenciesError` despite the
# call type-checking.
assert_type(
    Agent[object](instructions="", model_settings=_MS, deps_type=_Sub),
    Agent[object],
)
