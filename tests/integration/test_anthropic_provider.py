"""Integration smoke tests against the real Anthropic API.

Gated by `ANTHROPIC_API_KEY`; skipped when the environment variable is unset.
Not run by the default `make test` / unit-test path - invoked separately via
`make test-integration` and a dedicated GitHub Actions workflow.
"""

import os

import pytest
from pydantic import BaseModel

from avior.core import Agent, ModelSettings, Runner
from avior.core.exceptions import MaxTokensExceededError
from avior.core.messages import AssistantMessage, ToolCallPart
from avior.core.tools import Tool
from avior.providers.anthropic import AnthropicProvider

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


class _MagicNumberArgs(BaseModel):
    """Arguments for the `_MagicNumber` tool."""

    city: str


class _MagicNumber(Tool[_MagicNumberArgs, str]):
    """Returns a city's "magic number" - a value the model cannot know itself.

    Used to force a genuine tool round-trip: the model has to call the tool and
    relay its result, so the result text is proof the call actually happened.
    """

    name = "get_magic_number"
    description = "Get the secret magic number for a city."
    args_model = _MagicNumberArgs

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def execute(self, args: _MagicNumberArgs) -> str:
        self.calls.append(args.city)
        return "4242"


async def test_runner_run_against_anthropic_returns_non_empty_text(
    anthropic_provider: AnthropicProvider,
) -> None:
    """`Runner.run` against real Anthropic returns a non-empty assistant reply.

    End-to-end smoke: avior `Agent` -> `AnthropicProvider` -> `anthropic` SDK ->
    HTTP -> Anthropic Messages API -> decoded `Message`.  Asserts only on the
    transport contract (a non-empty string), not on response content.
    """

    # GIVEN an agent using the real Anthropic provider and a cheap model
    agent = Agent(
        provider=anthropic_provider,
        instructions="Reply with one short word.",
        model_settings=ModelSettings(
            model="claude-haiku-4-5-20251001",
            max_tokens=64,
        ),
    )

    # WHEN we run a trivial prompt
    result = await Runner.run(agent, "Say hello.")

    # THEN we get a non-empty text response
    assert result.output.strip() != ""


async def test_runner_run_raises_max_tokens_exceeded_against_anthropic(
    anthropic_provider: AnthropicProvider,
) -> None:
    """`Runner.run` raises `MaxTokensExceededError` when the token cap is hit.

    Confirms end-to-end mapping: Anthropic returns `stop_reason="max_tokens"`
    -> provider sets canonical `stop_reason="max_tokens"` -> Runner raises.
    """

    # GIVEN an agent with `model_settings.max_tokens` too small to complete
    agent = Agent(
        provider=anthropic_provider,
        instructions="Write a long story.",
        model_settings=ModelSettings(
            model="claude-haiku-4-5-20251001",
            max_tokens=1,
        ),
    )

    # WHEN `Runner.run` is invoked
    # THEN `MaxTokensExceededError` is raised
    with pytest.raises(MaxTokensExceededError):
        await Runner.run(agent, "Tell me a story.")


async def test_runner_run_against_anthropic_calls_a_tool_end_to_end(
    anthropic_provider: AnthropicProvider,
) -> None:
    """`Runner.run` drives a full tool round-trip against real Anthropic.

    Exercises the wire contract that mocks cannot: Anthropic accepts our tool
    definition, returns a `tool_use` block, accepts the echoed `tool_use` plus
    `tool_result` blocks on the continuation request, and produces a final
    answer relaying the tool's result.
    """

    # GIVEN an agent offered a tool whose result the model cannot otherwise know
    tool = _MagicNumber()
    agent = Agent(
        provider=anthropic_provider,
        instructions=(
            "When asked for a city's magic number, you must call the "
            "get_magic_number tool, then state the number it returns."
        ),
        model_settings=ModelSettings(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
        ),
        tools=[tool],
    )

    # WHEN we run a prompt that requires the tool
    result = await Runner.run(agent, "What is the magic number for Paris?")

    # THEN the tool was actually invoked with the parsed argument
    assert tool.calls == ["Paris"]

    # AND the assistant requested the call through the canonical IR
    tool_calls = [
        part
        for message in result.messages
        if isinstance(message, AssistantMessage)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]
    assert any(call.tool_name == "get_magic_number" for call in tool_calls)

    # AND the final answer relays the tool's result
    assert "4242" in result.output
