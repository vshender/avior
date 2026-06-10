"""Tests for `Runner`'s tool-dispatch loop."""

import json

import pytest
from pydantic import BaseModel

from avior.core.agent import Agent
from avior.core.exceptions import MaxIterationsExceeded
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

    async def execute(self, args: _EchoArgs) -> str:
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
    args: dict[str, object],
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

        async def execute(self, args: _EchoArgs) -> _Weather:
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

        async def execute(self, args: _EchoArgs) -> dict[str, object]:
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
    def always_call(_messages: object, _settings: object) -> AssistantMessage:
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
