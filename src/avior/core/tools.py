"""Tools the LLM can call.

A `Tool` is a named operation the LLM may invoke during a run.  It declares the
shape of its arguments as a Pydantic model and runs them in `execute`.

The LLM is shown the tool's name, description, and the JSON schema of its
arguments model.  When the LLM asks to call the tool, the arguments it sends are
validated and coerced through that same model before `execute` runs - so
`execute` always receives a typed, validated arguments object, never a raw dict.
"""

from abc import ABC, abstractmethod
from typing import Generic

from pydantic import BaseModel
from typing_extensions import TypeVar

from avior.core.context import RunContext

# `Tool` is *invariant* in `Args`: the parameter appears both covariantly
# (`args_model: type[Args]`) and contravariantly (`execute`'s `args`
# parameter), and one used in both positions can only be invariant - which is
# the `TypeVar`'s default, so no flag.
Args = TypeVar("Args", bound=BaseModel)
# `Tool` is *covariant* in `Result`: it appears only as `execute`'s return
# type (an output position), so a tool producing a subtype fits where one
# producing a supertype is expected (`Tool[Args, Sub]` <: `Tool[Args, Base]`).
# Explicit because the package targets Python 3.12, where native syntax would
# infer it.
Result = TypeVar("Result", covariant=True)
# `Tool` is *contravariant* in `Deps`: if `Sub <: Base` (i.e. `Sub` is a
# subtype of `Base`) then `Tool[..., Base] <: Tool[..., Sub]`.  The subtyping is
# reversed because a tool *consumes* its deps (it reads `ctx.deps`, never
# produces them).  In plain terms, a tool that requires only `Base` can be used
# wherever a tool requiring the more specific `Sub` is expected - whatever
# supplies a `Sub` also supplies a `Base`.  At the top type this is the
# deps-agnostic case: a tool requiring `object` (nothing) fits any agent, which
# is why `object` is the default.
#
# It is written as an explicit `TypeVar` only because the package targets
# Python 3.12, which has no `class Tool[..., Deps = object]` default syntax.
# On native syntax `Tool`'s contravariance *would* be inferred automatically
# (`Deps` appears only inside the covariant `RunContext` in a parameter
# position) - unlike `RunContext`, whose covariance could not be inferred there.
Deps = TypeVar("Deps", default=object, contravariant=True)


class Tool(ABC, Generic[Args, Result, Deps]):
    """A named operation the LLM can invoke, with typed arguments.

    Subclass it, set `name`, `description`, and `args_model`, and implement
    `execute`.  `args_model` is the single source of truth for both the schema
    sent to the LLM and the validation/coercion of the arguments it returns.

    The type parameters make a single tool subclass type-safe: `execute` takes
    that subclass's `args_model` instance, returns its own result type, and
    reads dependencies of type `Deps` through `ctx`.  `Deps` defaults to
    `object` - a tool that reads no dependencies requires nothing of the agent's
    deps, so it fits any agent.

    A collection of different tools cannot keep the per-tool parameters,
    though - each tool has its own arguments model and result type, and Python
    has no existential types to say "a tool with some arguments model and some
    result type".  Those two are erased to `Any`, so a mixed collection is
    typed `Tool[Any, Any, Deps]`.  The deps parameter is the exception: it
    stays the one `Deps` the whole collection shares (an agent types its tools
    as `Tool[Any, Any, Deps]`), because that is what lets the agent check each
    tool against its own deps type.  The erased per-tool types are
    re-established at runtime: the runner validates the incoming arguments
    through each tool's `args_model` before calling `execute`.
    """

    name: str
    """The tool's name, as exposed to the LLM."""

    description: str
    """A natural-language description of what the tool does, for the LLM."""

    args_model: type[Args]
    """The Pydantic model describing the tool's arguments.  Two roles:

    - its JSON schema is sent to the LLM as the tool's input schema;
    - arguments the LLM returns are validated and coerced through it before
      reaching `execute`.
    """

    @abstractmethod
    async def execute(self, ctx: RunContext[Deps], args: Args) -> Result:
        """Run the tool with validated `args` and return its result.

        `ctx` is the read-only run context: it carries the run's `deps` and the
        identity of this tool call.  `args` is the validated, coerced arguments
        object - an instance of `args_model`, never a raw dict.
        """
