"""Tests for `Runner`'s tool-dispatch loop."""

import json
from dataclasses import dataclass
from typing import Protocol

import pytest
from pydantic import BaseModel, JsonValue

from avior.core.agent import Agent
from avior.core.context import RunContext
from avior.core.exceptions import MaxIterationsExceeded, MissingDependenciesError
from avior.core.messages import (
    AssistantMessage,
    ToolCallPart,
    ToolMessage,
    ToolResultError,
    ToolResultOk,
)
from avior.core.provider import ModelSettings
from avior.core.runner import Runner
from avior.core.testing import StubProvider
from avior.core.tools import Tool


class _EchoArgs(BaseModel):
    value: str


class _Echo(Tool[_EchoArgs, str]):
    """Echoes its argument; used to observe dispatch."""

    name = "echo"
    description = "Echo the value back."
    args_model = _EchoArgs

    async def execute(self, ctx: RunContext[object], args: _EchoArgs) -> str:
        return f"echo:{args.value}"


def _agent(*, max_iter: int = 10) -> Agent:
    """Build an `Agent` with a single `_Echo` tool."""

    return Agent(
        instructions="be helpful",
        model_settings=ModelSettings(model="test-model"),
        tools=[_Echo()],
        max_iter=max_iter,
    )


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


async def test_runner_dispatches_tool_call_then_returns_final() -> None:
    """Runs a requested tool, feeds the result back, then returns the reply."""

    # GIVEN a provider that first asks to call `echo`, then replies with text
    provider = StubProvider.from_responses(
        [
            _tool_call("c1", "echo", {"value": "hi"}),
            "All done.",
        ]
    )

    # WHEN the runner is invoked
    result = await Runner(provider=provider).run(_agent(), "please echo hi")

    # THEN the final text is returned
    assert result.output == "All done."

    # AND the tool ran, its result captured as an `ok` ToolMessage
    tool_messages = [m for m in result.messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].parts[0].result == ToolResultOk(content="echo:hi")

    # AND the second model call saw that tool result in its input
    assert any(isinstance(m, ToolMessage) for m in provider.calls[-1].messages)


async def test_runner_serializes_base_model_tool_result_as_json() -> None:
    """A tool returning a `BaseModel` has its result fed back as JSON."""

    class _Weather(BaseModel):
        city: str
        temp_c: int

    class _GetWeather(Tool[_EchoArgs, _Weather]):
        name = "weather"
        description = "Return structured weather."
        args_model = _EchoArgs

        async def execute(self, ctx: RunContext[object], args: _EchoArgs) -> _Weather:
            return _Weather(city=args.value, temp_c=21)

    # GIVEN an agent whose tool returns a `BaseModel`, then a final reply
    provider = StubProvider.from_responses(
        [
            _tool_call("c1", "weather", {"value": "Paris"}),
            "done",
        ]
    )
    agent = Agent(
        instructions="be helpful",
        model_settings=ModelSettings(model="test-model"),
        tools=[_GetWeather()],
    )

    # WHEN the runner is invoked
    result = await Runner(provider=provider).run(agent, "weather in Paris?")

    # THEN the model's JSON dump is what gets fed back to the model
    tool_messages = [m for m in result.messages if isinstance(m, ToolMessage)]
    expected = ToolResultOk(content=_Weather(city="Paris", temp_c=21).model_dump_json())
    assert tool_messages[0].parts[0].result == expected


async def test_runner_serializes_other_tool_result_as_json_dump() -> None:
    """A tool result that is neither `str` nor `BaseModel` is JSON-dumped."""

    class _GetWeather(Tool[_EchoArgs, dict[str, object]]):
        name = "weather"
        description = "Return weather as a dict."
        args_model = _EchoArgs

        async def execute(
            self,
            ctx: RunContext[object],
            args: _EchoArgs,
        ) -> dict[str, object]:
            return {"city": args.value, "temp_c": 21}

    # GIVEN an agent whose tool returns a plain dict, then a final reply
    provider = StubProvider.from_responses(
        [
            _tool_call("c1", "weather", {"value": "Paris"}),
            "done",
        ]
    )
    agent = Agent(
        instructions="be helpful",
        model_settings=ModelSettings(model="test-model"),
        tools=[_GetWeather()],
    )

    # WHEN the runner is invoked
    result = await Runner(provider=provider).run(agent, "weather in Paris?")

    # THEN the dict is fed back as its JSON dump
    tool_messages = [m for m in result.messages if isinstance(m, ToolMessage)]
    expected = ToolResultOk(
        content=json.dumps({"city": "Paris", "temp_c": 21}, default=str)
    )
    assert tool_messages[0].parts[0].result == expected


async def test_runner_raises_max_iterations_when_tools_never_settle() -> None:
    """A model that only ever calls tools trips the `max_iter` guard."""

    # GIVEN a provider that always requests a tool call
    def always_call(_call: object) -> AssistantMessage:
        return _tool_call("c", "echo", {"value": "x"})

    provider = StubProvider(always_call)

    # WHEN the runner is invoked
    # THEN it gives up after `max_iter` iterations
    with pytest.raises(MaxIterationsExceeded):
        await Runner(provider=provider).run(_agent(max_iter=3), "go")


async def test_runner_feeds_error_result_for_unknown_tool() -> None:
    """A call to an unregistered tool becomes an `error` result, not a crash."""

    # GIVEN a provider that calls a tool the agent does not have, then finishes
    provider = StubProvider.from_responses(
        [
            _tool_call("c1", "does_not_exist", {}),
            "ok",
        ]
    )

    # WHEN the runner is invoked
    result = await Runner(provider=provider).run(_agent(), "go")

    # THEN the run completes and the unknown call was reported back as an error
    tool_messages = [m for m in result.messages if isinstance(m, ToolMessage)]
    expected = ToolResultError(content="Unknown tool: 'does_not_exist'.")
    assert tool_messages[0].parts[0].result == expected


async def test_runner_feeds_error_result_for_invalid_args() -> None:
    """Arguments that fail validation become an `error` result, not a crash."""

    # GIVEN a provider that calls `echo` with the required field missing
    provider = StubProvider.from_responses(
        [
            _tool_call("c1", "echo", {}),
            "ok",
        ]
    )

    # WHEN the runner is invoked
    result = await Runner(provider=provider).run(_agent(), "go")

    # THEN the validation failure is reported back as an error result
    tool_messages = [m for m in result.messages if isinstance(m, ToolMessage)]
    result_part = tool_messages[0].parts[0].result
    assert isinstance(result_part, ToolResultError)
    assert result_part.content.startswith("Invalid arguments: ")


@dataclass
class _Deps:
    """Toy dependency object threaded into a tool through its context."""

    token: str


class _Capture(Tool[_EchoArgs, str, _Deps]):
    """Records the `RunContext` it is given so a test can inspect it."""

    name = "capture"
    description = "Capture the run context."
    args_model = _EchoArgs

    def __init__(self) -> None:
        self.seen: RunContext[_Deps] | None = None

    async def execute(self, ctx: RunContext[_Deps], args: _EchoArgs) -> str:
        self.seen = ctx
        return f"token={ctx.deps.token}"


async def test_runner_threads_deps_and_call_identity_into_tool_ctx() -> None:
    """A tool's `RunContext` carries `deps` and the call's identity."""

    # GIVEN a deps-typed agent whose tool records the context it receives
    provider = StubProvider.from_responses(
        [
            _tool_call("call-7", "capture", {"value": "x"}),
            "done",
        ]
    )
    deps = _Deps(token="secret")
    tool = _Capture()
    agent = Agent(
        instructions="be helpful",
        model_settings=ModelSettings(model="test-model"),
        tools=[tool],
        deps_type=_Deps,
    )

    # WHEN the runner is invoked with that deps value
    await Runner(provider=provider).run(agent, "go", deps=deps)

    # THEN the tool saw the same deps object and this call's identity
    assert tool.seen is not None
    assert tool.seen.deps is deps
    assert tool.seen.tool_name == "capture"
    assert tool.seen.tool_call_id == "call-7"
    assert tool.seen.run_step == 1


async def test_runner_requires_deps_when_agent_declares_deps_type() -> None:
    """A deps-typed agent run without `deps` raises before any model call."""

    # GIVEN a deps-typed agent
    provider = StubProvider.from_responses(["unused"])
    agent = Agent(
        instructions="be helpful",
        model_settings=ModelSettings(model="test-model"),
        tools=[_Capture()],
        deps_type=_Deps,
    )

    # WHEN `run` is invoked without `deps` (a type error, exercised at runtime)
    # THEN it raises before the provider is ever called
    with pytest.raises(MissingDependenciesError, match="deps"):
        await Runner(provider=provider).run(agent, "go")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    assert provider.calls == []


async def test_runner_guard_follows_deps_type_under_wider_annotation() -> None:
    """A concrete `deps_type` arms the guard even under a wider `Agent[object]`.

    `type[...]` is covariant, so `Agent[object](deps_type=...)` type-checks and
    the run below is statically allowed; the runtime guard still follows the
    concrete `deps_type` and raises.
    """

    # GIVEN an `Agent[object]` that nonetheless carries a concrete `deps_type`
    provider = StubProvider.from_responses(["unused"])
    agent = Agent[object](
        instructions="be helpful",
        model_settings=ModelSettings(model="test-model"),
        deps_type=_Deps,
    )

    # WHEN run without `deps` (statically allowed for `Agent[object]`)
    # THEN the runtime guard still fires
    with pytest.raises(MissingDependenciesError, match="deps"):
        await Runner(provider=provider).run(agent, "go")
    assert provider.calls == []


async def test_runner_allows_omitting_deps_for_object_deps_type() -> None:
    """`deps_type=object` needs no `deps`: runtime agrees with the overloads."""

    # GIVEN an agent that declares `deps_type=object` (deps-agnostic)
    provider = StubProvider.from_responses(["done"])
    agent = Agent(
        instructions="be helpful",
        model_settings=ModelSettings(model="test-model"),
        deps_type=object,
    )

    # WHEN `run` is invoked without `deps` (the overloads accept this)
    result = await Runner(provider=provider).run(agent, "go")

    # THEN it runs instead of raising (no static/runtime contract mismatch)
    assert result.output == "done"


class _Nullable(Protocol):
    """An empty protocol; `None` satisfies it structurally."""


class _NeedsNullable(Tool[_EchoArgs, str, _Nullable]):
    """A tool whose deps type accepts `None`; echoes the deps it gets."""

    name = "needs_nullable"
    description = "Echo the received deps."
    args_model = _EchoArgs

    async def execute(self, ctx: RunContext[_Nullable], args: _EchoArgs) -> str:
        return f"deps={ctx.deps}"


async def test_runner_accepts_explicit_none_deps_for_nullable_type() -> None:
    """A real `deps=None` is honored for a deps type that accepts it.

    `None` satisfies an empty `Protocol`, so the call type-checks; the runtime
    guard must not mistake the passed `None` for an omitted argument and raise.
    """

    # GIVEN a deps-typed agent whose declared deps type is satisfied by `None`
    provider = StubProvider.from_responses(
        [
            _tool_call("c1", "needs_nullable", {"value": "x"}),
            "done",
        ]
    )
    agent = Agent(
        instructions="be helpful",
        model_settings=ModelSettings(model="test-model"),
        tools=[_NeedsNullable()],
        deps_type=_Nullable,
    )

    # WHEN `run` is invoked with an explicit `deps=None` (a valid value here)
    result = await Runner(provider=provider).run(agent, "go", deps=None)

    # THEN it does not raise; the run completes and the tool received `None`
    assert result.output == "done"
    tool_messages = [m for m in result.messages if isinstance(m, ToolMessage)]
    assert tool_messages[0].parts[0].result == ToolResultOk(content="deps=None")
