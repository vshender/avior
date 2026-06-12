"""Tests for `avior.core.agent`."""

import pytest
from pydantic import BaseModel

from avior.core.agent import Agent
from avior.core.context import RunContext
from avior.core.exceptions import ConfigurationError
from avior.core.provider import ModelSettings
from avior.core.tools import Tool


class _NoArgs(BaseModel):
    pass


class _Ping(Tool[_NoArgs, str]):
    """A tool named `dup`."""

    name = "dup"
    description = "First tool."
    args_model = _NoArgs

    async def execute(self, ctx: RunContext[object], args: _NoArgs) -> str:
        return "ping"


class _Pong(Tool[_NoArgs, str]):
    """A different tool that also claims the name `dup`."""

    name = "dup"
    description = "Second tool sharing the name."
    args_model = _NoArgs

    async def execute(self, ctx: RunContext[object], args: _NoArgs) -> str:
        return "pong"


def test_agent_rejects_duplicate_tool_names() -> None:
    """Two tools sharing a name fail at construction with a config error."""

    # GIVEN two distinct tools that both use the name "dup"
    # WHEN an agent is constructed with both
    # THEN construction raises `ConfigurationError` naming the duplicate
    with pytest.raises(ConfigurationError, match="dup"):
        Agent(
            instructions="be helpful",
            model_settings=ModelSettings(model="test-model"),
            tools=[_Ping(), _Pong()],
        )


def test_agent_snapshots_tools_and_does_not_alias_caller_list() -> None:
    """`tools` is snapshotted to a tuple; the caller's list is not aliased."""

    # GIVEN an agent constructed from a mutable list with one tool
    ping = _Ping()
    tools: list[Tool[_NoArgs, str]] = [ping]
    agent = Agent(
        instructions="be helpful",
        model_settings=ModelSettings(model="test-model"),
        tools=tools,
    )

    # WHEN the caller mutates the original list afterwards
    tools.append(_Pong())

    # THEN the agent kept its own immutable snapshot, unaffected by the mutation
    assert isinstance(agent.tools, tuple)
    assert agent.tools == (ping,)
