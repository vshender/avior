"""The run context passed to tools.

The runner gives every tool a `RunContext` when it calls the tool.  The context
is frozen: a tool can read its fields but cannot reassign them.  It holds the
run's dependencies (`deps`) and the identity of the current tool call.

The context exposes no operations of its own - no I/O, no clock, no method that
acts on the run.  Anything live a tool needs, like a database or HTTP client,
comes through `deps`, supplied per run.  So the context itself adds no coupling
to a particular runtime, which lets the same tool run unchanged on a durable
backend that can pause a run and resume it later.
"""

from dataclasses import dataclass
from typing import Generic

from typing_extensions import TypeVar

# `RunContext` is *covariant* in `Deps`: if `Sub <: Base` (i.e. `Sub` is a
# subtype of `Base`) then `RunContext[Sub] <: RunContext[Base]`.  In plain
# terms, a `RunContext[Sub]` can be used wherever a `RunContext[Base]` is
# expected.  A tool written against `RunContext[Base]` therefore also handles
# the context of an agent whose deps is `Base` or any subtype of it.
#
# The deps-agnostic case is the same rule at the top type: a tool that ignores
# deps declares `RunContext[object]`, and since every type is a subtype of
# `object`, it accepts any agent's context.  (`object` is `Tool`'s default deps
# type.)
#
# Covariance is safe here only because the context is frozen: the `deps` field
# can be read but not reassigned, so a tool cannot store a wider value into it.
#
# The variance is declared explicitly; it cannot be left to inference.  If this
# class used the native `class RunContext[Deps]` form, the checker would infer
# the `deps` dataclass field as *invariant* (it does not treat the field as
# read-only, even when frozen).  That invariance would propagate to `Tool` and
# break the reuse: `Tool[..., object]` would then fit only `Agent[object]`.
# Defaults are likewise explicit because Python 3.12 has no `class Foo[T = ...]`
# syntax, so the package uses `typing_extensions.TypeVar` + `Generic[...]`.
#
# The default is `None`, not `object`: it mirrors `Agent`'s deps default and the
# real value of a no-deps run (`ctx.deps is None`).  "Requires nothing"
# (`object`) is `Tool`'s concern, the held value (`None`) is the context's.
Deps = TypeVar("Deps", covariant=True, default=None)


@dataclass(frozen=True, kw_only=True)
class RunContext(Generic[Deps]):
    """The read-only context a tool receives for one tool call.

    `Deps` is the type of the run's dependencies - the live objects (database
    clients, HTTP clients, configuration, caller identity) the tool may read
    through `deps`.  An agent declares the type its tools expect; the caller
    supplies the value per run via `Runner.run(..., deps=...)`.  Omit that
    argument - allowed when the agent needs no dependencies - and `deps` is
    `None`.

    The context is frozen: a tool cannot rebind `deps` or any other field.
    """

    deps: Deps
    """The run's dependencies, of the type the agent declared.

    Live and supplied per run.  Read-only by contract: nothing stops a tool from
    mutating the object's contents, but mutating them is not portable.  A local
    run reuses one `deps` object, so changes persist (and leak back to the
    caller); a durable substrate reconstructs `deps` per run rather than
    restoring it, so changes are lost.  Treating `deps` as read-only keeps a
    tool's behavior the same on every backend.  `deps` is input the run reads,
    not a place to accumulate state.
    """

    tool_name: str
    """The name of the tool being called."""

    tool_call_id: str
    """The provider's id for this tool call, matching the originating
    `ToolCallPart.call_id`.
    """

    run_step: int
    """The 1-based agent-loop iteration this call belongs to.

    All tool calls the model requests in one model turn share the same
    `run_step`.
    """
