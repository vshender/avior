"""Integration smoke tests against the real Anthropic API.

Gated by `ANTHROPIC_API_KEY`; skipped when the environment variable is unset.
Not run by the default `make test` / unit-test path - invoked separately via
`make test-integration` and a dedicated GitHub Actions workflow.
"""

import os

import pytest
from pydantic import BaseModel

from avior.core import Agent, ModelSettings, Runner
from avior.core.context import RunContext
from avior.core.exceptions import MaxTokensExceededError
from avior.core.messages import (
    AssistantMessage,
    ThinkingPart,
    ToolCallPart,
    UserMessage,
)
from avior.core.tools import Tool
from avior.providers.anthropic import AnthropicProvider

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)

_MODEL = "claude-haiku-4-5-20251001"

# A model whose thinking runs through Anthropic's adaptive config, unlike the
# budget-mode `_MODEL`.
_ADAPTIVE_MODEL = "claude-sonnet-4-6"


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

    async def execute(self, ctx: RunContext[object], args: _MagicNumberArgs) -> str:
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

    # GIVEN an agent and a runner using the real Anthropic provider
    agent = Agent(
        instructions="Reply with one short word.",
        model_settings=ModelSettings(model=_MODEL, max_tokens=64),
    )

    # WHEN we run a trivial prompt
    result = await Runner(provider=anthropic_provider).run(agent, "Say hello.")

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
        instructions="Write a long story.",
        model_settings=ModelSettings(model=_MODEL, max_tokens=1),
    )

    # WHEN `Runner.run` is invoked
    # THEN `MaxTokensExceededError` is raised
    with pytest.raises(MaxTokensExceededError):
        await Runner(provider=anthropic_provider).run(agent, "Tell me a story.")


async def test_runner_run_against_anthropic_calls_a_tool_end_to_end(
    anthropic_provider: AnthropicProvider,
) -> None:
    """`Runner.run` drives a full tool round-trip against real Anthropic.

    Exercises the wire contract that mocks cannot: Anthropic accepts our tool
    declaration, returns a `tool_use` block, accepts the echoed `tool_use` plus
    `tool_result` blocks on the continuation request, and produces a final
    answer relaying the tool's result.
    """

    # GIVEN an agent offered a tool whose result the model cannot otherwise know
    tool = _MagicNumber()
    agent = Agent(
        instructions=(
            "When asked for a city's magic number, you must call the "
            "get_magic_number tool, then state the number it returns."
        ),
        model_settings=ModelSettings(model=_MODEL, max_tokens=256),
        tools=[tool],
    )

    # WHEN we run a prompt that requires the tool
    result = await Runner(provider=anthropic_provider).run(
        agent, "What is the magic number for Paris?"
    )

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


async def test_complete_returns_thinking_against_anthropic(
    anthropic_provider: AnthropicProvider,
) -> None:
    """`complete` decodes a real thinking block with summarized text.

    Sends a raw `thinking` config (enabled, `display="summarized"`) and a
    reasoning prompt, then checks the assistant turn carries a `ThinkingPart`
    whose text is the model's summarized reasoning.
    """

    # GIVEN settings that request budget thinking with a readable summary
    settings = ModelSettings(
        model=_MODEL,
        provider_options={
            "anthropic": {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 2000,
                    "display": "summarized",
                }
            }
        },
    )

    # WHEN `complete` is awaited on a prompt that needs reasoning
    result = await anthropic_provider.complete(
        [UserMessage.from_text("What is 17 * 23?  Reason step by step.")],
        settings,
    )

    # THEN the assistant turn carries a thinking block with summarized text
    thinking_parts = [p for p in result.message.parts if isinstance(p, ThinkingPart)]
    assert thinking_parts
    assert any(p.content.strip() for p in thinking_parts)


async def test_complete_returns_adaptive_thinking_against_anthropic(
    anthropic_provider: AnthropicProvider,
) -> None:
    """The portable `thinking` level drives adaptive thinking on a modern model.

    A modern model reasons through Anthropic's adaptive config
    (`{"type": "adaptive"}` plus `output_config.effort`).  Asserts the assistant
    turn carries a `ThinkingPart`, proving the portable level maps to the
    adaptive wire form and Anthropic accepts it.  The block's text is omitted by
    default, so only its presence is checked, not its content.
    """

    # GIVEN settings that request thinking through the portable level
    settings = ModelSettings(model=_ADAPTIVE_MODEL, thinking="high")

    # WHEN `complete` is awaited on a prompt that needs reasoning
    result = await anthropic_provider.complete(
        [UserMessage.from_text("What is 17 * 23?  Reason step by step.")],
        settings,
    )

    # THEN the assistant turn carries a thinking block
    thinking_parts = [p for p in result.message.parts if isinstance(p, ThinkingPart)]
    assert thinking_parts


async def test_complete_drops_temperature_with_thinking_against_anthropic(
    anthropic_provider: AnthropicProvider,
) -> None:
    """avior drops a `temperature` that thinking makes invalid, avoiding a 400.

    Anthropic rejects a non-default `temperature` while thinking is active, and
    returns a 400.  With thinking on and a custom temperature, avior drops the
    temperature before sending, so the call succeeds and records the drop as a
    warning.
    """

    # GIVEN settings with thinking on and a temperature Anthropic would reject
    settings = ModelSettings(model=_MODEL, thinking="low", temperature=0.5)

    # WHEN `complete` is awaited
    result = await anthropic_provider.complete(
        [UserMessage.from_text("What is 2 + 2?")],
        settings,
    )

    # THEN the call succeeded (no 400) and a temperature warning was recorded
    temperature_warnings = [
        w for w in result.warnings if w.setting_name == "temperature"
    ]
    assert len(temperature_warnings) == 1


async def test_runner_run_thinking_tool_chain_against_anthropic(
    anthropic_provider: AnthropicProvider,
) -> None:
    """A thinking-enabled tool round-trip completes against real Anthropic.

    With `thinking` on, the assistant turn carries thinking blocks alongside the
    tool call; the continuation must echo them back unchanged or Anthropic
    rejects the turn.  A completed run relaying the tool's result proves the
    multi-step thinking round-trip holds.
    """

    # GIVEN a thinking-enabled agent offered a tool whose result it cannot know
    tool = _MagicNumber()
    agent = Agent(
        instructions=(
            "When asked for a city's magic number, you must call the "
            "get_magic_number tool, then state the number it returns."
        ),
        model_settings=ModelSettings(model=_MODEL, thinking="low"),
        tools=[tool],
    )

    # WHEN we run a prompt that requires the tool
    result = await Runner(provider=anthropic_provider).run(
        agent, "What is the magic number for Paris?"
    )

    # THEN the tool ran and the final answer relays its result, so the thinking
    # blocks round-tripped through the tool loop
    assert tool.calls == ["Paris"]
    assert "4242" in result.output

    # AND the assistant turn carried thinking blocks (else the round-trip was
    # never exercised)
    thinking_parts = [
        part
        for message in result.messages
        if isinstance(message, AssistantMessage)
        for part in message.parts
        if isinstance(part, ThinkingPart)
    ]
    assert thinking_parts
