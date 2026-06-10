"""Integration smoke tests against the real OpenAI Responses API.

Gated by `OPENAI_API_KEY`; skipped when the environment variable is unset.
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
from avior.providers.openai_responses import OpenAIResponsesProvider

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
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


async def test_runner_run_against_openai_returns_non_empty_text(
    openai_responses_provider: OpenAIResponsesProvider,
) -> None:
    """`Runner.run` against real OpenAI Responses returns a non-empty reply.

    End-to-end smoke: avior `Agent` -> `OpenAIResponsesProvider` -> `openai`
    SDK -> HTTP -> OpenAI Responses API -> decoded `Message`.  Asserts only on
    the transport contract (a non-empty string), not on response content.
    """

    # GIVEN an agent and a runner using the real OpenAI Responses provider
    agent = Agent(
        instructions="Reply with one short word.",
        model_settings=ModelSettings(
            model="gpt-4.1-nano",
            max_tokens=256,
        ),
    )

    # WHEN we run a trivial prompt
    result = await Runner(provider=openai_responses_provider).run(agent, "Say hello.")

    # THEN we get a non-empty text response
    assert result.output.strip() != ""


async def test_runner_run_raises_max_tokens_exceeded_against_openai(
    openai_responses_provider: OpenAIResponsesProvider,
) -> None:
    """`Runner.run` raises `MaxTokensExceededError` when the token cap is hit.

    Confirms end-to-end mapping: OpenAI Responses returns `status="incomplete"`
    with `incomplete_details.reason="max_output_tokens"` -> provider sets
    canonical `stop_reason="max_tokens"` -> Runner raises.
    """

    # GIVEN an agent with `model_settings.max_tokens` too small to complete.
    # (OpenAI Responses API enforces `max_output_tokens >= 16`; 16 is enough
    # to trigger truncation for a long-story prompt.)
    agent = Agent(
        instructions="Write a long story.",
        model_settings=ModelSettings(
            model="gpt-4.1-nano",
            max_tokens=16,
        ),
    )

    # WHEN `Runner.run` is invoked
    # THEN `MaxTokensExceededError` is raised
    with pytest.raises(MaxTokensExceededError):
        await Runner(provider=openai_responses_provider).run(agent, "Tell me a story.")


async def test_runner_run_against_openai_calls_a_tool_end_to_end(
    openai_responses_provider: OpenAIResponsesProvider,
) -> None:
    """`Runner.run` drives a full tool round-trip against real OpenAI.

    Exercises the wire contract that mocks cannot: OpenAI accepts our function
    tool definition, returns a `function_call`, accepts the echoed
    `function_call` plus `function_call_output` items on the continuation
    request, and produces a final answer relaying the tool's result.
    """

    # GIVEN an agent offered a tool whose result the model cannot otherwise know
    tool = _MagicNumber()
    agent = Agent(
        instructions=(
            "When asked for a city's magic number, you must call the "
            "get_magic_number tool, then state the number it returns."
        ),
        model_settings=ModelSettings(model="gpt-4.1-nano", max_tokens=256),
        tools=[tool],
    )

    # WHEN we run a prompt that requires the tool
    result = await Runner(provider=openai_responses_provider).run(
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
