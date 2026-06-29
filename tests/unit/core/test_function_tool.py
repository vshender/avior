"""Tests for the `@tool` decorator and the `FunctionTool` it produces.

These cover the schema-derivation contract (what `@tool` reads off a function
signature) and that the produced tool dispatches correctly through the runner.
They do not re-test Pydantic's coercion: `@tool` only builds the model, so the
coercion is Pydantic's behavior, not `@tool`'s to prove.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field, JsonValue

from avior.core.agent import Agent
from avior.core.context import RunContext
from avior.core.exceptions import ConfigurationError
from avior.core.messages import (
    AssistantMessage,
    ToolCallPart,
    ToolMessage,
    ToolResultOk,
)
from avior.core.provider import ModelSettings
from avior.core.runner import Runner
from avior.core.testing import StubProvider
from avior.core.tools import tool


def _tool_call(
    call_id: str,
    tool_name: str,
    args: dict[str, JsonValue],
) -> AssistantMessage:
    """Build an assistant message requesting a single tool call."""

    return AssistantMessage(
        parts=[ToolCallPart(call_id=call_id, tool_name=tool_name, args=args)],
        stop_reason="tool_use",
    )


def test_tool_builds_args_model_from_loose_params() -> None:
    """Each parameter becomes a field; a default makes it not required."""

    # GIVEN a tool from a function with a required and a defaulted parameter
    @tool
    def get_weather(city: str, units: str = "c") -> str:
        """Get the weather."""

        return f"{city}:{units}"

    # WHEN the generated arguments model's JSON schema is taken
    schema = get_weather.args_model.model_json_schema()

    # THEN both are properties and only the undefaulted one is required
    assert set(schema["properties"]) == {"city", "units"}
    assert set(schema["required"]) == {"city"}


def test_tool_takes_param_description_from_annotated_field() -> None:
    """A parameter's `Annotated[..., Field(description=...)]` reaches schema."""

    # GIVEN a tool from a function whose parameter carries a Field description
    @tool
    def get_weather(
        city: Annotated[str, Field(description="The city to look up.")],
    ) -> str:
        """Get the weather."""

        return city

    # WHEN the schema is taken
    schema = get_weather.args_model.model_json_schema()

    # THEN the field carries that description
    assert schema["properties"]["city"]["description"] == "The city to look up."


def test_tool_uses_function_name_and_docstring() -> None:
    """The tool's name is the function name; its description, the docstring."""

    # GIVEN a tool from a function with a docstring
    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""

        return city

    # THEN name and description come from the function
    assert get_weather.name == "get_weather"
    assert get_weather.description == "Get the current weather for a city."


def test_tool_without_docstring_has_empty_description() -> None:
    """No docstring yields an empty description - allowed, not an error."""

    # GIVEN a tool from a function with no docstring
    @tool
    def get_weather(city: str) -> str:
        return city

    # THEN the description is empty; a docstring is recommended, not required
    assert get_weather.description == ""


def test_tool_parameterized_overrides_name_and_description() -> None:
    """`@tool(name=..., description=...)` overrides both function defaults."""

    # GIVEN a tool built with an explicit name and description
    @tool(name="weather", description="Look up the weather.")
    def get_weather(city: str) -> str:
        """Original docstring, overridden below."""

        return city

    # THEN the overrides win over the function name and docstring
    assert get_weather.name == "weather"
    assert get_weather.description == "Look up the weather."


def test_tool_parameterized_name_only_keeps_docstring_description() -> None:
    """Giving only `name` leaves the description from the docstring."""

    # GIVEN a tool built with only an explicit name
    @tool(name="weather")
    def get_weather(city: str) -> str:
        """Get the weather."""

        return city

    # THEN name is overridden; description still comes from the docstring
    assert get_weather.name == "weather"
    assert get_weather.description == "Get the weather."


def test_tool_parameterized_description_only_keeps_function_name() -> None:
    """Giving only `description` leaves the name from the function."""

    # GIVEN a tool built with only an explicit description
    @tool(description="Look up the weather.")
    def get_weather(city: str) -> str:
        """Original docstring, overridden below."""

        return city

    # THEN description is overridden; name still comes from the function
    assert get_weather.name == "get_weather"
    assert get_weather.description == "Look up the weather."


def test_tool_direct_call_with_metadata() -> None:
    """`tool(func, name=..., description=...)` builds the tool in one call."""

    # GIVEN a plain function with default metadata
    def get_weather(city: str) -> str:
        """Original docstring, overridden below."""

        return city

    # WHEN a tool is built from the function with explicit metadata in one
    # direct call
    named = tool(get_weather, name="weather", description="Look up the weather.")

    # THEN the overrides are applied without the decorator form
    assert named.name == "weather"
    assert named.description == "Look up the weather."


def test_tool_parameterized_empty_description_clears_it() -> None:
    """An explicit `description=""` overrides the docstring with empty."""

    # GIVEN a tool built with an explicit empty description
    @tool(description="")
    def get_weather(city: str) -> str:
        """This docstring must not win over the explicit empty description."""

        return city

    # THEN the description is the explicit empty string, not the docstring
    assert get_weather.description == ""


def test_tool_lifts_param_description_from_docstring() -> None:
    """A parameter's docstring `Args` entry fills its schema field."""

    # GIVEN a tool whose docstring documents a parameter
    @tool
    def get_weather(city: str) -> str:
        """Get the weather.

        Args:
            city: The city to look up.
        """

        return city

    # WHEN the schema is taken
    schema = get_weather.args_model.model_json_schema()

    # THEN the field carries the description from the docstring
    assert schema["properties"]["city"]["description"] == "The city to look up."


def test_tool_explicit_field_wins_over_docstring_param() -> None:
    """An explicit `Field(description=...)` beats the docstring `Args` entry."""

    # GIVEN a parameter documented both by a Field and the docstring
    @tool
    def get_weather(
        city: Annotated[str, Field(description="from Field")],
    ) -> str:
        """Get the weather.

        Args:
            city: from docstring.
        """

        return city

    # WHEN the schema is taken
    schema = get_weather.args_model.model_json_schema()

    # THEN the explicit `Field` description wins
    assert schema["properties"]["city"]["description"] == "from Field"


def test_tool_explicit_description_still_lifts_param_docs() -> None:
    """An explicit `description` overrides the tool text but still lets the
    docstring fill parameter descriptions - the two are independent.
    """

    # GIVEN a description override on a tool whose docstring documents a param
    @tool(description="Overridden.")
    def get_weather(city: str) -> str:
        """This summary is overridden.

        Args:
            city: The city to look up.
        """

        return city

    # WHEN its description and arguments schema are read
    schema = get_weather.args_model.model_json_schema()

    # THEN the description is the override
    assert get_weather.description == "Overridden."
    # AND the parameter description is still lifted from the docstring
    assert schema["properties"]["city"]["description"] == "The city to look up."


def test_tool_warns_on_docstring_param_not_in_signature() -> None:
    """A docstring `Args` entry naming no parameter warns - not raises.

    The function may be third-party, so a stale or mistyped docstring must not
    break tool construction; the warning surfaces that the parameter the author
    meant to document is left undocumented.
    """

    # GIVEN a function whose docstring documents a misspelled parameter
    # WHEN it is wrapped as a tool
    # THEN a warning is raised rather than the entry silently dropped
    with pytest.warns(UserWarning, match="not in its signature"):

        @tool
        def get_weather(city: str) -> str:
            """Get the weather.

            Args:
                citY: typo for `city`.
            """

            return city

    # AND the tool is still built - the warning is not an error
    assert get_weather.name == "get_weather"


def test_tool_documenting_ctx_param_does_not_warn(
    recwarn: pytest.WarningsRecorder,
) -> None:
    """Documenting the run-context parameter is allowed, not 'unknown'."""

    # GIVEN a tool built from a function that documents its `RunContext` param
    @tool
    def get_weather(ctx: RunContext[object], city: str) -> str:
        """Get the weather.

        Args:
            ctx: the run context.
            city: the city to look up.
        """

        return city

    # WHEN its arguments schema is read
    schema = get_weather.args_model.model_json_schema()

    # THEN no warning is raised (`ctx` is a real parameter, just not a field),
    assert len(recwarn) == 0
    # AND `ctx` is kept out of the schema (it is the run context, not an arg)
    assert "ctx" not in schema["properties"]


def test_tool_numpy_examples_are_preserved() -> None:
    """NumPy-style examples keep their content (the doctest line survives)."""

    # GIVEN a NumPy-style docstring with a doctest example
    @tool(docstring_format="numpy")
    def do(x: int) -> int:
        """Do a thing.

        Examples
        --------
        >>> do(1)
        2
        """

        return x

    # WHEN the description is read
    description = do.description

    # THEN both the call line and its output reach the description
    assert "Examples:" in description
    assert ">>> do(1)" in description
    assert "\n2\n" in description


def test_tool_description_keeps_sections_except_args() -> None:
    """The description keeps body/Returns/Raises/Examples, drops Args/Notes."""

    # GIVEN a tool with a full multi-section docstring
    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city.

        More detail about the lookup.

        Args:
            city: The city to look up.

        Returns:
            A weather summary.

        Raises:
            ValueError: If the city is unknown.

        Examples:
            >>> get_weather("London")
            'cloudy'

        Notes:
            Dropped, since it is not portable across styles.
        """

        return city

    # WHEN the description is read
    description = get_weather.description

    # THEN the summary and body are kept
    assert description.startswith(
        "Get the weather for a city.\n\nMore detail about the lookup."
    )
    # AND the Returns / Raises / Examples sections are kept
    assert "Returns:\n    A weather summary." in description
    assert "Raises:\n    ValueError: If the city is unknown." in description
    assert "Examples:\n```\n>>> get_weather(\"London\")\n'cloudy'\n```" in description
    # AND the Args section content and freeform Notes are dropped
    assert "The city to look up." not in description
    assert "Notes" not in description


def test_tool_multiline_raises_description_nests_continuation() -> None:
    """A wrapped `Raises` description continues indented under its entry."""

    # GIVEN a tool whose Raises entry has a multi-line description
    @tool
    def get_weather(city: str) -> str:
        """Get the weather.

        Raises:
            ValueError: the city code is unknown
                or not yet supported.
        """

        return city

    # WHEN the description is read
    description = get_weather.description

    # THEN the continuation line nests at 8 spaces under the entry, rather than
    # going flush-left (which would read as a separate, type-less entry)
    assert (
        "Raises:\n"
        "    ValueError: the city code is unknown\n"
        "        or not yet supported." in description
    )


def test_tool_folds_annotated_return_description() -> None:
    """A return with `Field(description=...)` becomes a Returns section."""

    # GIVEN a tool whose return type carries a description
    @tool
    def add(x: int, y: int) -> Annotated[int, Field(description="The sum of x and y.")]:
        """Add two numbers."""

        return x + y

    # WHEN the description is read
    description = add.description

    # THEN the return description is folded into the tool description
    assert description == "Add two numbers.\n\nReturns:\n    The sum of x and y."


def test_tool_annotated_return_wins_over_returns_section() -> None:
    """An annotated return beats a `Returns` section, mirroring params."""

    # GIVEN both an annotated return and a Returns section
    @tool
    def add(x: int, y: int) -> Annotated[int, Field(description="from annotation")]:
        """Add two numbers.

        Returns:
            from docstring.
        """

        return x + y

    # THEN the annotation wins
    assert "from annotation" in add.description
    assert "from docstring" not in add.description


def test_tool_annotated_return_survives_unresolvable_self() -> None:
    """A bound method's annotated return is read even when `self` is
    unresolvable - the return resolves in isolation, like parameters do.
    """

    # GIVEN a bound method with an annotated return whose `self` annotation
    # cannot be resolved at runtime (a forward ref to a missing type - the
    # scenario `@tool` already supports for parameters).  Inject it directly to
    # avoid a source-level forward ref the type checkers would flag.
    class Service:
        def fetch(
            self, n: int
        ) -> Annotated[int, Field(description="the fetched value")]:
            return n

    Service.fetch.__annotations__["self"] = "MissingSelf"

    # WHEN a tool is built from the bound method
    fetch_tool = tool(Service().fetch)

    # THEN the return description still reaches the tool description - the
    # unresolvable `self` does not mask it
    assert "the fetched value" in fetch_tool.description


def test_tool_parses_non_google_style_via_auto() -> None:
    """Auto-detection handles a non-Google style (here, NumPy)."""

    # GIVEN a tool documented in NumPy style
    @tool
    def get_weather(city: str) -> str:
        """Get the weather.

        Parameters
        ----------
        city : str
            The city to look up.
        """

        return city

    # WHEN the schema is taken
    schema = get_weather.args_model.model_json_schema()

    # THEN the NumPy `Parameters` section still fills the field description
    assert schema["properties"]["city"]["description"] == "The city to look up."


def test_tool_empty_annotated_return_suppresses_returns_section() -> None:
    """An explicit empty result description differs from no annotation."""

    # GIVEN an explicitly empty return annotation alongside a Returns section
    @tool
    def add(x: int, y: int) -> Annotated[int, Field(description="")]:
        """Add two numbers.

        Returns:
            this docstring text must be suppressed.
        """

        return x + y

    # THEN the explicit empty wins: no Returns section is emitted at all
    assert add.description == "Add two numbers."


def test_tool_rejects_unknown_docstring_format() -> None:
    """An unknown `docstring_format` is rejected at the API boundary."""

    # GIVEN a function (with and without a docstring, to cover both paths)
    def documented(city: str) -> str:
        """Get the weather."""

        return city

    def undocumented(city: str) -> str:
        return city

    # WHEN a tool is built with an invalid format - the type system rejects it
    # for a typed caller, so this guards the untyped one (the value is typed
    # `Any`)
    bad_format: Any = "bad"

    # THEN it fails with a clear error rather than a raw `KeyError`, whether or
    # not the function has a docstring
    for func in (documented, undocumented):
        with pytest.raises(ConfigurationError, match="unknown docstring_format"):
            tool(func, docstring_format=bad_format)


def test_tool_docstring_format_pins_the_style() -> None:
    """`docstring_format` pins the style for parsing."""

    # GIVEN a Sphinx-style docstring parsed with the style pinned
    @tool(docstring_format="sphinx")
    def get_weather(city: str) -> str:
        """Get the weather.

        :param city: The city to look up.
        """

        return city

    # THEN the Sphinx `:param:` entry fills the field description
    schema = get_weather.args_model.model_json_schema()
    assert schema["properties"]["city"]["description"] == "The city to look up."


def test_tool_rejects_empty_name() -> None:
    """An empty `name` is rejected: the LLM addresses a tool by name."""

    # GIVEN a function
    def get_weather(city: str) -> str:
        """Get the weather."""

        return city

    # WHEN a tool is built with an explicit empty name
    # THEN it fails rather than producing an unaddressable tool
    with pytest.raises(ConfigurationError, match="non-empty name"):
        tool(get_weather, name="")


def test_tool_rejects_none_function() -> None:
    """`tool(None)` is a passed value, not the no-function sentinel.

    The type checker already rejects this call (hence the suppressions); the
    test pins the runtime behavior for untyped callers, and that `None` is
    distinguished from the internal "no function passed" sentinel rather than
    silently returning a decorator.
    """

    # GIVEN `None` passed where a function is expected
    # WHEN `tool` is called with it as the positional function argument
    # THEN it fails with a clear error, not a silent decorator
    with pytest.raises(ConfigurationError, match="function or method"):
        tool(None)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]


def test_tool_omits_run_context_from_args_schema() -> None:
    """A `RunContext` first parameter is detected and kept out of the schema."""

    # GIVEN a tool from a function that takes the run context plus an argument
    @tool
    def get_weather(ctx: RunContext[object], city: str) -> str:
        """Get the weather."""

        return city

    # THEN the tool knows it takes context, and only `city` is in the schema
    assert get_weather.takes_ctx is True
    assert set(get_weather.args_model.model_json_schema()["properties"]) == {"city"}


async def test_tool_dispatches_sync_function_end_to_end() -> None:
    """A sync `@tool` function dispatches through the runner like any tool."""

    # GIVEN a sync tool and a provider that calls it, then replies
    @tool
    def echo(value: str) -> str:
        """Echo the value back."""

        return f"echo:{value}"

    provider = StubProvider.from_responses(
        [_tool_call("c1", "echo", {"value": "hi"}), "done"]
    )
    agent = Agent(
        instructions="be helpful",
        model_settings=ModelSettings(model="test-model"),
        tools=[echo],
    )

    # WHEN the runner is invoked
    result = await Runner(provider=provider).run(agent, "echo hi")

    # THEN the run completes and the tool's result was captured
    assert result.output == "done"
    tool_messages = [m for m in result.messages if isinstance(m, ToolMessage)]
    assert tool_messages[0].parts[0].result == ToolResultOk(content="echo:hi")


async def test_tool_injects_run_context_with_deps_end_to_end() -> None:
    """An async ctx-reading tool gets the run's deps injected at call time."""

    # GIVEN a deps type and an async tool that reads it through the context
    @dataclass
    class Deps:
        token: str

    @tool
    async def read_token(ctx: RunContext[Deps]) -> str:
        """Return the deps token."""

        return f"token={ctx.deps.token}"

    provider = StubProvider.from_responses([_tool_call("c1", "read_token", {}), "done"])
    agent = Agent(
        instructions="be helpful",
        model_settings=ModelSettings(model="test-model"),
        tools=[read_token],
        deps_type=Deps,
    )

    # WHEN the runner is invoked with a deps value
    result = await Runner(provider=provider).run(
        agent, "token?", deps=Deps(token="secret")
    )

    # THEN the tool saw those deps
    tool_messages = [m for m in result.messages if isinstance(m, ToolMessage)]
    assert tool_messages[0].parts[0].result == ToolResultOk(content="token=secret")


def test_tool_rejects_run_context_after_first_parameter() -> None:
    """A `RunContext` parameter that is not first is a configuration error."""

    # GIVEN a function with the context in second position
    def bad(city: str, ctx: RunContext[object]) -> str:
        return city

    # WHEN it is wrapped
    # THEN wrapping fails because the context must come first
    with pytest.raises(ConfigurationError, match="must be the first parameter"):
        tool(bad)


def test_tool_rejects_keyword_only_run_context() -> None:
    """A keyword-only `RunContext` cannot receive the positional context."""

    # GIVEN a function whose context parameter is keyword-only
    def bad(*, ctx: RunContext[object], city: str) -> str:
        return city

    # WHEN it is wrapped
    # THEN wrapping fails because the context is injected positionally
    with pytest.raises(ConfigurationError, match="keyword-only"):
        tool(bad)


def test_tool_rejects_model_namespace_name() -> None:
    """A `model_` name shadowing a `BaseModel` attribute is rejected."""

    # GIVEN a function with a parameter in the reserved `model_` namespace
    def bad(model_config: int) -> int:
        return model_config

    # WHEN it is wrapped
    # THEN wrapping fails with a clear message instead of a raw `TypeError`
    with pytest.raises(ConfigurationError, match="reserved `model_` namespace"):
        tool(bad)


def _var_positional(*args: int) -> None:
    """A `*args` function for the variadic-rejection test.

    Module-level and typed (not a lambda) so it is a well-typed `Callable` that
    `tool(...)` accepts statically; the rejection then happens at runtime.
    """


def _var_keyword(**kwargs: int) -> None:
    """A `**kwargs` twin of `_var_positional` for the same test."""


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(_var_positional, id="var_positional"),
        pytest.param(_var_keyword, id="var_keyword"),
    ],
)
def test_tool_rejects_var_args_and_var_kwargs(bad: Callable[..., object]) -> None:
    """`*args` / `**kwargs` do not map to a tool schema and are rejected."""

    # GIVEN a function using a variadic parameter
    # WHEN it is wrapped
    # THEN wrapping fails with a clear message
    with pytest.raises(ConfigurationError, match="cannot take "):
        tool(bad)


async def test_tool_passes_positional_only_parameter_positionally() -> None:
    """A positional-only parameter is still a field, but called positionally."""

    # GIVEN a tool whose only parameter is positional-only
    @tool
    def shout(text: str, /) -> str:
        """Uppercase the text."""

        return text.upper()

    provider = StubProvider.from_responses(
        [_tool_call("c1", "shout", {"text": "hi"}), "done"]
    )
    agent = Agent(
        instructions="be helpful",
        model_settings=ModelSettings(model="test-model"),
        tools=[shout],
    )

    # WHEN the runner is invoked
    result = await Runner(provider=provider).run(agent, "go")

    # THEN the parameter is in the schema and was passed positionally (no error)
    assert "text" in shout.args_model.model_json_schema()["properties"]
    assert shout.positional_params == ("text",)
    tool_messages = [m for m in result.messages if isinstance(m, ToolMessage)]
    assert tool_messages[0].parts[0].result == ToolResultOk(content="HI")


def test_tool_rejects_leading_underscore_parameter() -> None:
    """A leading-underscore name is excluded from the model, so rejected."""

    # GIVEN a function whose parameter name starts with an underscore
    def bad(_value: str) -> str:
        return _value

    # WHEN it is wrapped
    # THEN wrapping fails rather than silently producing an empty schema
    with pytest.raises(ConfigurationError, match="leading underscore"):
        tool(bad)


def test_tool_allows_underscore_named_run_context() -> None:
    """The context parameter is matched by type, so an `_`-name is fine."""

    # GIVEN a tool whose context parameter has a leading-underscore name
    @tool
    def lookup(_ctx: RunContext[object], city: str) -> str:
        """Look something up."""

        return city

    # THEN the context is recognized (not rejected) and kept out of the schema
    assert lookup.takes_ctx is True
    assert set(lookup.args_model.model_json_schema()["properties"]) == {"city"}


def test_tool_detects_annotated_run_context() -> None:
    """`Annotated[RunContext[...], ...]` is recognized as context, not field."""

    # GIVEN a tool whose context parameter is wrapped in `Annotated`
    @tool
    def lookup(ctx: Annotated[RunContext[object], "ctx"], city: str) -> str:
        """Look something up."""

        return city

    # THEN the `Annotated` wrapper does not hide it: it stays out of the schema
    assert lookup.takes_ctx is True
    assert set(lookup.args_model.model_json_schema()["properties"]) == {"city"}


@pytest.mark.parametrize("annotation", ["DefinitelyMissing", "pytest.Nope"])
def test_tool_rejects_unresolvable_annotation(annotation: str) -> None:
    """An unresolvable parameter annotation becomes a clear error.

    Both an undefined name (`NameError`) and a bad attribute (`AttributeError`,
    e.g. `pytest.Nope`) must be reported clearly, not leak raw.
    """

    # GIVEN a function whose parameter annotation cannot be resolved at runtime
    def f(x: object) -> object:
        return x

    f.__annotations__ = {"x": annotation}

    # WHEN it is wrapped
    # THEN wrapping fails with a clear message instead of a raw error
    with pytest.raises(ConfigurationError, match="cannot be resolved"):
        tool(f)


def test_tool_ignores_unresolvable_return_annotation() -> None:
    """An unresolvable return annotation is ignored: it is unused at runtime."""

    # GIVEN a function whose return annotation cannot be resolved, params fine
    def f(x: int) -> object:
        return x

    f.__annotations__ = {"x": int, "return": "DefinitelyMissing"}

    # WHEN it is wrapped
    # THEN it builds anyway, since only parameters drive the schema
    built = tool(f)
    assert set(built.args_model.model_json_schema()["properties"]) == {"x"}


def test_tool_rejects_non_function_callable() -> None:
    """`@tool` supports functions and methods; other callables are rejected."""

    # GIVEN a callable that is not a function or method (a `functools.partial`)
    def base(a: int) -> int:
        return a

    partial = functools.partial(base, a=1)

    # WHEN it is wrapped
    # THEN wrapping fails with a clear message instead of a raw `TypeError`
    with pytest.raises(ConfigurationError, match="function or method"):
        tool(partial)


def test_tool_rejects_unbound_method_with_self() -> None:
    """`@tool` on a class-body method sees an unbound `self`, so it rejects."""

    # GIVEN a function whose first parameter is `self` (an unbound method)
    def lookup(self: object, city: str) -> str:
        return city

    # WHEN it is wrapped
    # THEN wrapping fails rather than leaking `self` into the schema
    with pytest.raises(ConfigurationError, match="does not bind methods"):
        tool(lookup)


def test_tool_accepts_bound_method() -> None:
    """A bound method works: its signature already drops `self`."""

    # GIVEN a bound method whose signature is `(city)` after binding
    class Service:
        def lookup(self, city: str) -> str:
            return city

    method = Service().lookup

    # WHEN the bound method is wrapped
    built = tool(method)

    # THEN `self` is gone and only `city` is in the schema
    assert set(built.args_model.model_json_schema()["properties"]) == {"city"}


def test_tool_accepts_bound_method_with_unresolvable_self_annotation() -> None:
    """A bound method works even if `self` has an unresolvable annotation.

    `self` is not an effective parameter (the bound signature drops it), so its
    annotation is never resolved - it must not break tool construction.
    """

    # GIVEN a bound method whose `self` is annotated with a name not resolvable
    # at runtime (as it would be under `from __future__ import annotations`)
    class Service:
        def lookup(self, city: str) -> str:
            return city

    Service.lookup.__annotations__ = {"self": "Service", "city": "str", "return": "str"}
    method = Service().lookup

    # WHEN the bound method is wrapped
    built = tool(method)

    # THEN it builds; the unresolved `self` annotation is ignored
    assert set(built.args_model.model_json_schema()["properties"]) == {"city"}


async def test_tool_passes_validated_model_instance_not_dict() -> None:
    """A nested-model argument reaches the function as a model, not a dict.

    `@tool`'s contract is that the function gets the validated, typed values,
    not the raw dict from the model.
    """

    # GIVEN a tool whose argument is itself a Pydantic model
    class Point(BaseModel):
        x: int
        y: int

    @tool
    def plot(point: Point) -> str:
        """Report the runtime type of the received argument."""

        return type(point).__name__

    provider = StubProvider.from_responses(
        [_tool_call("c1", "plot", {"point": {"x": 1, "y": 2}}), "done"]
    )
    agent = Agent(
        instructions="be helpful",
        model_settings=ModelSettings(model="test-model"),
        tools=[plot],
    )

    # WHEN the runner dispatches the call with a nested-dict argument
    result = await Runner(provider=provider).run(agent, "go")

    # THEN the function received a `Point` instance, not the raw dict
    tool_messages = [m for m in result.messages if isinstance(m, ToolMessage)]
    assert tool_messages[0].parts[0].result == ToolResultOk(content="Point")
