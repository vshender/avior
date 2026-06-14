"""Tools the LLM can call.

A `Tool` is a named operation the LLM may invoke during a run.  It declares the
shape of its arguments as a Pydantic model and runs them in `execute`.

The LLM is shown the tool's name, description, and the JSON schema of its
arguments model.  When the LLM asks to call the tool, the arguments it sends are
validated and coerced through that same model before `execute` runs - so
`execute` always receives a typed, validated arguments object, never a raw dict.

`Tool` is the low-level primitive: you set `name`, `description`, and
`args_model`, and implement `execute`.  `@tool` is the sugar over it - it takes
an ordinary typed function and derives those same pieces from its signature.  A
function-defined tool fits an agent exactly as a hand-written `Tool` subclass
does.
"""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import (
    Annotated,
    Any,
    Concatenate,
    Generic,
    Protocol,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from pydantic import BaseModel, create_model
from typing_extensions import TypeVar

from avior.core.context import RunContext
from avior.core.exceptions import ConfigurationError

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


# `FunctionTool` and `tool` use native (PEP 695) type parameters.  Spelling type
# variables out explicitly is only necessary when native 3.12 syntax cannot
# express what a generic needs - two cases:
#
# - a variance inference would get wrong: `RunContext` needs `Deps` covariant,
#   but its frozen `deps` field reads as assignable, so inference makes it
#   invariant;
# - a PEP-696 default (e.g. `Deps` defaulting to `object`); native default
#   syntax is 3.13+, and the package targets 3.12.
#
# `FunctionTool` hits neither: no field of it stores `Result` or `Deps`, so
# nothing forces them invariant.  Inference is correct - `Result` covariant from
# `execute`'s return, `Deps` contravariant from its `RunContext[Deps]` parameter
# - and no default is wanted.  Its `Result`/`Deps` parameters shadow the
# module-level type variables of the same name - a benign shadow, since both
# name the same concepts (a tool's result and deps types).


@dataclass(frozen=True)
class FunctionTool[Result, Deps](Tool[Any, Result, Deps]):
    """A `Tool` whose behavior is a Python function, produced by `@tool`.

    It is a frozen dataclass, not a Pydantic model: it holds a live function
    (`func`), and Pydantic models are for serializable data.

    `FunctionTool` is generic over `Result` and `Deps`, but not over the
    arguments model.  Where a hand-written `Tool` subclass names a concrete
    `args_model` - so `Args` has a name, `Tool[WeatherArgs, ...]` - `@tool`
    synthesizes the model from the signature at runtime (via `create_model`), so
    that model has no statically nameable type, only `type[BaseModel]`.
    `Result` survives because the return annotation names it; the synthesized
    model does not, so `Args` drops to `Any`.

    Dropping it costs nothing the dispatch relied on.  The runner validates the
    LLM's raw arguments through `args_model` before `execute` either way, so the
    real model instance is re-established at runtime.  The static `Args` only
    ever helped a direct typed call, and there is no named type here to make one
    against.  `Deps` is preserved - it is what fits the tool to an agent's deps.
    """

    name: str
    """The tool's name, as exposed to the LLM."""

    description: str
    """A natural-language description of what the tool does, for the LLM."""

    args_model: type[BaseModel]
    """The Pydantic model generated from the function's parameters."""

    func: Callable[..., object]
    """The wrapped function.  May be sync or async."""

    takes_ctx: bool
    """Whether `func` takes the `RunContext` first, to be injected at call."""

    positional_params: tuple[str, ...]
    """`func`'s positional-only parameters, in order (passed positionally)."""

    async def execute(self, ctx: RunContext[Deps], args: BaseModel) -> Result:
        """Call the wrapped function with the validated arguments.

        The runner has already validated and coerced the LLM's raw arguments
        through `args_model`, so `args` is a typed model instance.
        Positional-only parameters are passed positionally and the rest by
        keyword; the context is prepended only if the function declared it.
        A sync function runs inline; an async one is awaited.
        """

        fields = {name: getattr(args, name) for name in type(args).model_fields}
        positional = [fields.pop(name) for name in self.positional_params]

        if self.takes_ctx:
            out = self.func(ctx, *positional, **fields)
        else:
            out = self.func(*positional, **fields)

        if inspect.isawaitable(out):
            out = await cast(Awaitable[object], out)

        return cast(Result, out)


class _ToolDecorator(Protocol):
    """The decorator that `tool(name=..., description=...)` returns.

    Its `__call__` repeats the four bare-`tool` overloads, so applying the
    returned decorator to a function infers the same `FunctionTool[Result,
    Deps]` that bare `@tool` would.  The repetition is unavoidable: the
    ctx/async distinction resolves only when the decorator is applied to the
    function, not when `tool(...)` builds the decorator, so the matrix has to
    live on this `__call__` too.

    The function is positional-only (`/`), as in the bare `tool` overloads and
    for the same reason (a decorator's target is passed positionally, so its
    parameter name is not public API); see the note above those overloads.
    """

    @overload
    def __call__[Result, Deps, **P](
        self,
        func: Callable[Concatenate[RunContext[Deps], P], Awaitable[Result]],
        /,
    ) -> FunctionTool[Result, Deps]: ...

    @overload
    def __call__[Result, Deps, **P](
        self,
        func: Callable[Concatenate[RunContext[Deps], P], Result],
        /,
    ) -> FunctionTool[Result, Deps]: ...

    @overload
    def __call__[Result, **P](
        self,
        func: Callable[P, Awaitable[Result]],
        /,
    ) -> FunctionTool[Result, object]: ...

    @overload
    def __call__[Result, **P](
        self,
        func: Callable[P, Result],
        /,
    ) -> FunctionTool[Result, object]: ...


# Sentinel for an omitted `func`, kept distinct from a real `func=None`: `None`
# is not a valid function, so a passed `None` must reach the validation (and be
# rejected), not read as "no function passed" (which is the decorator form).
_MISSING: Any = object()


# The four function overloads are tried top to bottom, so their order is
# load-bearing.  Context-first: a `RunContext[Deps]` first parameter binds
# `Deps` and matches before the context-free forms would.  Async-first: an
# async function binds `Result` to its awaited type (the `X` in `Awaitable[X]`),
# not to the whole `Awaitable[X]` - which is what the bare-value form would bind
# it to.  Each of the four also accepts the optional `name` / `description`
# kwargs, so the direct-call form `tool(func, name=...)` stays typed.  A fifth
# overload, last, handles the parameterized `tool(name=..., description=...)`
# call: it takes no function and returns the decorator above.
#
# In every form `func` is positional-only (`/`).  This follows
# `dataclasses.dataclass(cls, /)`, the stdlib precedent for an
# optional-argument decorator: a decorator's target is passed positionally
# (`@tool`, `tool(f)`), so its parameter name is an internal detail, not public
# API.


# Reads the run context, async: `Deps` from `ctx`, `Result` from the await.
@overload
def tool[Result, Deps, **P](
    func: Callable[Concatenate[RunContext[Deps], P], Awaitable[Result]],
    /,
    *,
    name: str | None = ...,
    description: str | None = ...,
) -> FunctionTool[Result, Deps]: ...


# Reads the run context, sync.
@overload
def tool[Result, Deps, **P](
    func: Callable[Concatenate[RunContext[Deps], P], Result],
    /,
    *,
    name: str | None = ...,
    description: str | None = ...,
) -> FunctionTool[Result, Deps]: ...


# No run context, async: nothing constrains `Deps`, so it is `object`.
@overload
def tool[Result, **P](
    func: Callable[P, Awaitable[Result]],
    /,
    *,
    name: str | None = ...,
    description: str | None = ...,
) -> FunctionTool[Result, object]: ...


# No run context, sync.
@overload
def tool[Result, **P](
    func: Callable[P, Result],
    /,
    *,
    name: str | None = ...,
    description: str | None = ...,
) -> FunctionTool[Result, object]: ...


# Parameterized: called with metadata and no function, so it returns the
# decorator (which carries the four overloads again, via `_ToolDecorator`).
@overload
def tool(
    *,
    name: str | None = ...,
    description: str | None = ...,
) -> _ToolDecorator: ...


def tool(
    func: Any = _MISSING,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> FunctionTool[Any, Any] | _ToolDecorator:
    """Turn a typed function into a `FunctionTool`.

    Use it bare (`@tool`) above a `def` or `async def`, or call it directly
    (`tool(func)`).  The function's name, docstring, and parameters become the
    tool's name, description, and arguments model.  An optional first parameter
    annotated `RunContext[Deps]` is recognized as the run context and kept out
    of the arguments model.

    Pass `name` or `description` to override the values that would otherwise
    come from the function's `__name__` and docstring; either may be given on
    its own.  Both work as a parameterized decorator (`@tool(name=...)`, which
    returns the decorator) and in a direct call (`tool(func, name=...)`).  An
    explicit `description=""` clears the description; an empty `name` is
    rejected, since the LLM addresses a tool by name.

    For a method, pass a bound one - `tool(instance.method)` - whose signature
    has already dropped `self`.  Applying `@tool` to a method in a class body
    wraps the unbound function and is rejected, since `self` is unbound there.
    """

    def make(func: Callable[..., object], /) -> FunctionTool[Any, Any]:
        # Validate the function first: it rejects a non-function (e.g. a passed
        # `None`) with a clear error before `func.__name__` is read below.
        args_model, takes_ctx, positional_params = _build_args_model(func)

        resolved_name = name if name is not None else func.__name__
        if not resolved_name:
            raise ConfigurationError(
                f"@tool needs a non-empty name; got an empty `name` for "
                f"{func.__name__!r}."
            )

        resolved_description = (
            description if description is not None else (inspect.getdoc(func) or "")
        )

        return FunctionTool(
            name=resolved_name,
            description=resolved_description,
            args_model=args_model,
            func=func,
            takes_ctx=takes_ctx,
            positional_params=positional_params,
        )

    # Dispatch on whether a function is in hand:
    #
    # - parameterized form `@tool(...)`: no function yet -> return the decorator
    #   that builds the tool once applied;
    # - bare `@tool` or direct `tool(func[, name=...])`: a function in hand ->
    #   build now;
    # - a passed `None` is not the sentinel, so it flows into `make`, where
    #   `_build_args_model` rejects it (the type system already rejects
    #   `tool(None)`; this is the runtime guard for untyped callers).
    if func is _MISSING:
        return cast(_ToolDecorator, make)
    return make(func)


def _build_args_model(
    func: Callable[..., object],
) -> tuple[type[BaseModel], bool, tuple[str, ...]]:
    """Derive a tool's arguments model from a function signature.

    Returns the generated model, whether the function takes the run context as
    its first parameter, and the names of its positional-only parameters (in
    order, to pass them positionally).  `Annotated[T, Field(...)]`
    metadata (such as a parameter description) is carried onto the model field.
    A parameter with no annotation is typed `Any`; one with a default keeps it,
    so the field is not required.

    Raises `ConfigurationError` for a signature that does not map to a tool: a
    `self`/`cls` first parameter (an unbound method), a `RunContext` parameter
    that is not first or is keyword-only, `*args` / `**kwargs`, an argument name
    starting with `_`, or a `model_`-prefixed name that shadows a `BaseModel`
    attribute (Pydantic's reserved namespace).  The context parameter is matched
    by type, so its own name may start with `_`.

    Other Pydantic model-build errors (an invalid `Field`, an unsupported type)
    are left to propagate as themselves, not reinterpreted as a name conflict.
    """

    if not (inspect.isfunction(func) or inspect.ismethod(func)):
        raise ConfigurationError(
            f"@tool expects a function or method, got a {type(func).__name__!r}; "
            f"`functools.partial` and callable objects are not supported - wrap "
            f"the behavior in a `def`."
        )

    signature = inspect.signature(func)
    parameters = list(signature.parameters.values())

    # An unbound method applied in a class body still has `self`/`cls` first, so
    # it would leak into the schema and never be bound.  A bound method's
    # signature already drops it, so `tool(instance.method)` is unaffected.
    # Checked before resolving annotations so the message stays precise.
    if parameters and parameters[0].name in ("self", "cls"):
        raise ConfigurationError(
            f"Tool function {func.__name__!r} has {parameters[0].name!r} as its "
            f"first parameter; @tool does not bind methods.  Use a standalone "
            f"function, or pass a bound method: `tool(instance.method)`."
        )

    hints = _resolve_param_hints(func)
    takes_ctx = bool(parameters) and _is_run_context(hints.get(parameters[0].name))
    if takes_ctx:
        # The context is injected positionally (`func(ctx, ...)`), so a
        # keyword-only context cannot receive it.
        if parameters[0].kind is inspect.Parameter.KEYWORD_ONLY:
            raise ConfigurationError(
                f"Tool function {func.__name__!r} declares its `RunContext` "
                f"parameter {parameters[0].name!r} as keyword-only; declare it as "
                f"a normal first parameter (remove the `*` before it)."
            )
        parameters = parameters[1:]

    fields: dict[str, Any] = {}
    positional_params: list[str] = []
    for parameter in parameters:
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise ConfigurationError(
                f"Tool function {func.__name__!r} cannot take *args or **kwargs; "
                f"declare each argument as its own typed parameter."
            )

        # Pydantic silently drops leading-underscore names from the model, so
        # the argument would vanish from the schema and break the call.
        if parameter.name.startswith("_"):
            raise ConfigurationError(
                f"Tool function {func.__name__!r} has parameter {parameter.name!r}: "
                f"a leading underscore is excluded from the arguments model, so "
                f"rename it without the underscore."
            )

        # A `model_`-prefixed name that shadows a `BaseModel` attribute is a
        # protected-namespace conflict: Pydantic either rejects it outright
        # (`model_dump`) or lets it shadow a member we rely on (`model_fields`).
        if parameter.name.startswith("model_") and hasattr(BaseModel, parameter.name):
            raise ConfigurationError(
                f"Tool function {func.__name__!r} has parameter {parameter.name!r}, "
                f"which collides with Pydantic's reserved `model_` namespace "
                f"(it shadows `BaseModel.{parameter.name}`); rename it."
            )

        if _is_run_context(hints.get(parameter.name)):
            raise ConfigurationError(
                f"Tool function {func.__name__!r} declares RunContext as parameter "
                f"{parameter.name!r}; it must be the first parameter, if present."
            )

        # A positional-only parameter is still a model field, but cannot be
        # passed by keyword, so record it to pass positionally at call time.
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional_params.append(parameter.name)

        annotation = hints.get(parameter.name, Any)
        default = (
            parameter.default
            if parameter.default is not inspect.Parameter.empty
            else ...
        )
        fields[parameter.name] = (annotation, default)

    model = create_model(f"{func.__name__}_args", **fields)
    return model, takes_ctx, tuple(positional_params)


def _resolve_param_hints(func: Callable[..., object]) -> dict[str, Any]:
    """Resolve the annotations of the function's parameters to runtime types.

    Built from `inspect.signature`, so only the effective parameters are
    resolved: the return type (held separately) and a bound method's already-
    consumed `self`/`cls` are excluded.  Neither is a tool argument, and either
    might be unresolvable (a `TYPE_CHECKING`-only type) without that breaking
    construction.

    `Annotated[T, Field(...)]` metadata on a parameter is preserved, so a
    `Field(description=...)` reaches the LLM-facing schema.
    """

    # A shim carrying only the parameter annotations, so the function's own
    # `__annotations__` is never mutated.
    signature_parameters = inspect.signature(func).parameters
    shim = SimpleNamespace()
    shim.__annotations__ = {
        name: parameter.annotation
        for name, parameter in signature_parameters.items()
        if parameter.annotation is not inspect.Parameter.empty
    }
    try:
        # `include_extras` keeps the `Annotated[...]` wrappers; without it
        # `get_type_hints` strips them, dropping `Field(...)` on a parameter.
        return get_type_hints(shim, globalns=func.__globals__, include_extras=True)
    except (NameError, AttributeError) as exc:
        raise ConfigurationError(
            f"Tool function {func.__name__!r} has an annotation that cannot be "
            f"resolved: {exc}.  The annotated type must be available at runtime: "
            f"importable at module scope, not only in a local scope or under "
            f"`if TYPE_CHECKING:`, and spelled correctly."
        ) from exc


def _is_run_context(annotation: object) -> bool:
    """Tell whether an annotation is `RunContext`, bare or parameterized.

    `Annotated[RunContext[...], ...]` counts: the wrapper is stripped before
    checking, so context metadata does not hide the type.
    """

    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation is RunContext or get_origin(annotation) is RunContext
