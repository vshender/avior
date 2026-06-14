"""Type-level checks for the `@tool` decorator's inferred `FunctionTool` type.

Pins the overload contract of `tool`:

  - a deps-free function infers `FunctionTool[Result, object]`;
  - a `RunContext[Deps]` first parameter infers `FunctionTool[Result, Deps]`;
  - an `async` function's `Result` is the awaited type, not `Awaitable[...]`;
  - the parameterized `tool(name=..., description=...)` form infers the same
    `FunctionTool` as the bare form;
  - the inferred tool fits an agent's deps by the same contravariance rule as a
    hand-written `Tool` (see `deps.py`).
"""

from typing import Any, assert_type

from avior.core.agent import Agent
from avior.core.context import RunContext
from avior.core.provider import ModelSettings
from avior.core.tools import FunctionTool, Tool, tool

_MS = ModelSettings(model="test-model")


class _Base:
    """A nominal base deps type for the subtype-relation cases."""


class _Sub(_Base):
    """A subclass of `_Base`: the narrower deps type."""


@tool
def _free_sync(city: str) -> str:
    """A deps-free sync tool; reads no `RunContext`."""

    return city


@tool
async def _free_async(city: str) -> int:
    """A deps-free async tool; its `Result` is the awaited type."""

    return len(city)


@tool
def _ctx_sync(ctx: RunContext[_Base], n: int) -> str:
    """A sync tool that reads `_Base` deps through `RunContext`."""

    return str(n)


@tool
async def _ctx_async(ctx: RunContext[_Sub], n: int) -> bool:
    """An async tool that reads the narrower `_Sub` deps through
    `RunContext`.
    """

    return n > 0


assert_type(_free_sync, FunctionTool[str, object])
assert_type(_free_async, FunctionTool[int, object])
assert_type(_ctx_sync, FunctionTool[str, _Base])
assert_type(_ctx_async, FunctionTool[bool, _Sub])


# The parameterized form returns a decorator that repeats the overloads, so
# the ctx/async matrix still resolves on application: the inferred type matches
# the bare form for both the deps-free and the ctx case.
@tool(name="free_named", description="A renamed deps-free tool.")
def _free_named(city: str) -> str:
    """A parameterized deps-free tool."""

    return city


@tool(name="ctx_named", description="A renamed ctx tool.")
async def _ctx_named(ctx: RunContext[_Sub], n: int) -> bool:
    """A parameterized async ctx tool reading `_Sub` deps."""

    return n > 0


assert_type(_free_named, FunctionTool[str, object])
assert_type(_ctx_named, FunctionTool[bool, _Sub])


# The direct-call form `tool(func, name=...)` is typed by the same four
# overloads, now widened with the metadata kwargs, so it infers the result
# rather than falling back to the parameterized (decorator-returning) overload.
def _plain(city: str) -> str:
    """A function wrapped by a direct call carrying metadata."""

    return city


_direct = tool(_plain, name="renamed", description="A renamed tool.")
assert_type(_direct, FunctionTool[str, object])


# A `_Base`-requiring tool fits a `_Base` agent ...
assert_type(
    Agent(instructions="", model_settings=_MS, tools=[_ctx_sync], deps_type=_Base),
    Agent[_Base],
)
# ... or any subclass: a `_Sub` agent still supplies the `_Base` the tool reads.
assert_type(
    Agent(instructions="", model_settings=_MS, tools=[_ctx_sync], deps_type=_Sub),
    Agent[_Sub],
)
# A deps-free tool fits any agent.
assert_type(
    Agent(instructions="", model_settings=_MS, tools=[_free_sync], deps_type=_Base),
    Agent[_Base],
)


# Negative cases: each assignment must be REJECTED (the suppression is the
# assertion).  A `_Sub`-requiring tool does not fit a `_Base` slot: an agent
# supplying only `_Base` cannot satisfy a tool that reads `_Sub`.
_n1: Tool[Any, bool, _Base] = _ctx_async  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
# The same rule at the `Agent` boundary, where the tool list lives.
_n2: Agent[_Base] = Agent(instructions="", model_settings=_MS, tools=[_ctx_async])  # type: ignore[list-item]  # pyright: ignore[reportAssignmentType]
