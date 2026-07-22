"""Integration smoke tests against the real OpenAI Responses API.

Gated by `OPENAI_API_KEY`; skipped when the environment variable is unset.
Not run by the default `make test` / unit-test path - invoked separately via
`make test-integration` and a dedicated GitHub Actions workflow.
"""

import os
from typing import Literal

import pytest
from pydantic import BaseModel

from avior.core import Agent, ModelSettings, Runner
from avior.core.context import RunContext
from avior.core.exceptions import MaxTokensExceededError
from avior.core.messages import (
    AssistantMessage,
    ThinkingPart,
    ToolCallPart,
    ToolMessage,
    UserMessage,
)
from avior.core.tools import Tool
from avior.providers.openai_responses import OpenAIResponsesProvider

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)

_MODEL = "gpt-4.1-nano"

# A reasoning model, whose output carries reasoning items that must be replayed
# before their tool calls on the continuation request.
_REASONING_MODEL = "o4-mini"

# A model whose reasoning is off by default: an effort level turns reasoning
# on; `reasoning.effort="none"` keeps it off.
_OFF_BY_DEFAULT_MODEL = "gpt-5.1"

# A model whose reasoning is on by default: `reasoning.effort="none"` turns
# reasoning off.
_ON_BY_DEFAULT_MODEL = "gpt-5.5"


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
        model_settings=ModelSettings(model=_MODEL, max_tokens=256),
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
        model_settings=ModelSettings(model=_MODEL, max_tokens=16),
    )

    # WHEN `Runner.run` is invoked
    # THEN `MaxTokensExceededError` is raised
    with pytest.raises(MaxTokensExceededError):
        await Runner(provider=openai_responses_provider).run(agent, "Tell me a story.")


async def test_runner_run_against_openai_calls_a_tool_end_to_end(
    openai_responses_provider: OpenAIResponsesProvider,
) -> None:
    """`Runner.run` drives a full tool round-trip against real OpenAI.

    Exercises the wire contract that mocks cannot: OpenAI accepts our tool
    declaration, returns a `function_call`, accepts the echoed `function_call`
    plus `function_call_output` items on the continuation request, and produces
    a final answer relaying the tool's result.
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


async def test_runner_run_reasoning_tool_chain_against_openai(
    openai_responses_provider: OpenAIResponsesProvider,
) -> None:
    """A reasoning-model tool round-trip completes against real OpenAI.

    A reasoning model emits a reasoning item carrying an `encrypted_content`
    alongside its tool call.  The provider replays that item before the tool
    call on the continuation request, so the model's reasoning carries across
    turns.  A completed run relaying the tool's result, with a `ThinkingPart`
    present, proves the round-trip holds.
    """

    # GIVEN a reasoning-model agent offered a tool whose result it cannot know
    tool = _MagicNumber()
    agent = Agent(
        instructions=(
            "When asked for a city's magic number, you must call the "
            "get_magic_number tool, then state the number it returns."
        ),
        model_settings=ModelSettings(model=_REASONING_MODEL, max_tokens=4096),
        tools=[tool],
    )

    # WHEN we run a prompt that requires the tool
    result = await Runner(provider=openai_responses_provider).run(
        agent, "What is the magic number for Paris?"
    )

    # THEN the tool ran and the final answer relays its result, so the reasoning
    # item round-tripped through the tool loop
    assert tool.calls == ["Paris"]
    assert "4242" in result.output

    # AND the assistant turn carried a reasoning item (else the round-trip was
    # never exercised)
    thinking_parts = [
        part
        for message in result.messages
        if isinstance(message, AssistantMessage)
        for part in message.parts
        if isinstance(part, ThinkingPart)
    ]
    assert thinking_parts


@pytest.mark.parametrize("thinking", [True, "high"])
async def test_complete_enables_reasoning_on_off_by_default_model_against_openai(
    thinking: bool | Literal["low", "medium", "high"],
    openai_responses_provider: OpenAIResponsesProvider,
) -> None:
    """`thinking=True` or a level enables `off_by_default` reasoning.

    An `off_by_default` model does not reason on its own, so `True` or a
    level must send an explicit `reasoning.effort` (`True` -> a default
    effort, a level -> that effort); the assistant turn then carries a
    `ThinkingPart`, proving reasoning was actually enabled (not left at the
    off default).
    """

    # GIVEN settings that enable reasoning through the portable setting
    settings = ModelSettings(
        model=_OFF_BY_DEFAULT_MODEL,
        thinking=thinking,
        max_tokens=2048,
    )

    # WHEN `complete` is awaited on a prompt that needs reasoning
    result = await openai_responses_provider.complete(
        [UserMessage.from_text("What is 17 * 23?  Reason step by step.")],
        settings,
    )

    # THEN the model reasoned and no warning was recorded
    thinking_parts = [p for p in result.message.parts if isinstance(p, ThinkingPart)]
    assert thinking_parts
    assert result.warnings == []


async def test_complete_disables_reasoning_against_openai(
    openai_responses_provider: OpenAIResponsesProvider,
) -> None:
    """The portable `thinking=False` turns reasoning off on a capable model.

    On an `on_by_default` model - one that reasons when left alone - sending
    `thinking=False` maps to `reasoning.effort="none"`.  The call must be
    accepted (no 400), record no warning, and the assistant turn must carry no
    `ThinkingPart`, proving the model did not reason.  A custom `temperature`
    is sent alongside: with reasoning off, OpenAI accepts it.
    """

    # GIVEN settings that disable reasoning on an `on_by_default` model and
    # carry a custom temperature
    settings = ModelSettings(
        model=_ON_BY_DEFAULT_MODEL,
        thinking=False,
        temperature=0.3,
        max_tokens=2048,
    )

    # WHEN `complete` is awaited on a prompt that would otherwise reason
    result = await openai_responses_provider.complete(
        [UserMessage.from_text("What is 17 * 23?  Reason step by step.")],
        settings,
    )

    # THEN the model did not reason, and no warning was recorded - a dropped
    # temperature would have recorded one, so the temperature was forwarded
    # and OpenAI accepted the request carrying it
    thinking_parts = [p for p in result.message.parts if isinstance(p, ThinkingPart)]
    assert thinking_parts == []
    assert result.warnings == []


@pytest.mark.parametrize(
    ("model", "reasons_by_default"),
    [
        (_MODEL, False),
        (_OFF_BY_DEFAULT_MODEL, False),
        (_ON_BY_DEFAULT_MODEL, True),
        (_REASONING_MODEL, True),
    ],
    ids=["non-thinking", "off-by-default", "on-by-default", "always-on"],
)
async def test_complete_default_reasoning_matches_classification_against_openai(
    model: str,
    reasons_by_default: bool,
    openai_responses_provider: OpenAIResponsesProvider,
) -> None:
    """With `thinking` unset, each model shows its classified default behavior.

    avior's per-model reasoning modes assert which models reason when no
    reasoning config is sent; temperature dropping and reasoning replay
    depend on that classification.  This test pins one model per mode against
    the live API: a drifted default shows up as a `ThinkingPart` mismatch.
    """

    # GIVEN settings with `thinking` unset
    settings = ModelSettings(model=model, max_tokens=4096)

    # WHEN `complete` is awaited on a prompt that invites reasoning
    result = await openai_responses_provider.complete(
        [UserMessage.from_text("What is 17 * 23?  Reason step by step.")],
        settings,
    )

    # THEN a thinking part is present exactly when the model reasons by default
    thinking_parts = [p for p in result.message.parts if isinstance(p, ThinkingPart)]
    assert bool(thinking_parts) == reasons_by_default
    assert result.warnings == []


async def test_complete_drops_temperature_with_reasoning_against_openai(
    openai_responses_provider: OpenAIResponsesProvider,
) -> None:
    """avior drops a `temperature` that reasoning makes invalid, avoiding a 400.

    OpenAI rejects a non-default `temperature` while reasoning is active, and
    returns a 400.  With reasoning on and a custom temperature, avior drops the
    temperature before sending, so the call succeeds and records the drop as a
    warning.
    """

    # GIVEN settings with reasoning on and a temperature OpenAI would reject
    settings = ModelSettings(
        model=_OFF_BY_DEFAULT_MODEL, thinking="low", temperature=0.5
    )

    # WHEN `complete` is awaited
    result = await openai_responses_provider.complete(
        [UserMessage.from_text("What is 2 + 2?")],
        settings,
    )

    # THEN the call succeeded (no 400) and a temperature warning was recorded
    temperature_warnings = [
        w for w in result.warnings if w.setting_name == "temperature"
    ]
    assert len(temperature_warnings) == 1


async def test_complete_accepts_temperature_with_summary_only_config_against_openai(
    openai_responses_provider: OpenAIResponsesProvider,
) -> None:
    """A summary-only raw config leaves reasoning off and `temperature` sent.

    A raw reasoning config that carries only a `summary` sets no effort, so an
    `off_by_default` model stays at its default of not reasoning.  OpenAI must
    accept the request with the custom `temperature` (no 400), and the
    response must carry no `ThinkingPart` and no warning.
    """

    # GIVEN settings with an `off_by_default` model, a custom temperature, and
    # a raw reasoning config that carries only a summary
    settings = ModelSettings(
        model=_OFF_BY_DEFAULT_MODEL,
        temperature=0.3,
        max_tokens=2048,
        provider_options={"openai": {"reasoning": {"summary": "auto"}}},
    )

    # WHEN `complete` is awaited
    result = await openai_responses_provider.complete(
        [UserMessage.from_text("What is 17 * 23?")],
        settings,
    )

    # THEN the call succeeded, the model did not reason, and no warning was
    # recorded - a dropped temperature would have recorded one, so the
    # temperature was forwarded and OpenAI accepted the request carrying it
    thinking_parts = [p for p in result.message.parts if isinstance(p, ThinkingPart)]
    assert thinking_parts == []
    assert result.warnings == []


async def test_complete_accepts_reasoning_transcript_without_reasoning_against_openai(
    openai_responses_provider: OpenAIResponsesProvider,
) -> None:
    """A reasoning-run transcript replays into a reasoning-off request.

    With reasoning off, the adapter drops the transcript's reasoning items but
    keeps their tool calls, so the request carries a `function_call` without
    the reasoning item that preceded it.  OpenAI accepts that shape only when
    the `function_call` carries no item id, so this test fails if avior starts
    sending item ids.
    """

    # GIVEN a transcript from a reasoning run whose turn pairs a reasoning
    # item with a tool call
    tool = _MagicNumber()
    agent = Agent(
        instructions=(
            "When asked for a city's magic number, you must call the "
            "get_magic_number tool, then state the number it returns."
        ),
        model_settings=ModelSettings(
            model=_OFF_BY_DEFAULT_MODEL, thinking="high", max_tokens=4096
        ),
        tools=[tool],
    )
    reasoning_run = await Runner(provider=openai_responses_provider).run(
        agent, "What is the magic number for Paris?"
    )
    # The transcript is cut after the tool turn: the magic number then exists
    # only in the replayed call/result pair, so the continuation cannot answer
    # from a later assistant message instead
    last_tool_index = max(
        index
        for index, message in enumerate(reasoning_run.messages)
        if isinstance(message, ToolMessage)
    )
    transcript = reasoning_run.messages[: last_tool_index + 1]
    paired_turns = [
        message
        for message in transcript
        if isinstance(message, AssistantMessage)
        and any(isinstance(p, ThinkingPart) for p in message.parts)
        and any(isinstance(p, ToolCallPart) for p in message.parts)
    ]
    assert paired_turns, "precondition: the run must pair reasoning with a tool call"

    # AND settings that disable reasoning for the continuation
    settings = ModelSettings(
        model=_OFF_BY_DEFAULT_MODEL, thinking=False, max_tokens=2048
    )

    # WHEN `complete` is awaited on the transcript plus a follow-up user
    # message
    result = await openai_responses_provider.complete(
        [*transcript, UserMessage.from_text("Repeat the magic number.")],
        settings,
    )

    # THEN the request is accepted and answered from the replayed transcript
    text = result.message.text
    assert text
    assert "4242" in text


async def test_complete_returns_reasoning_summary_against_openai(
    openai_responses_provider: OpenAIResponsesProvider,
) -> None:
    """A raw `reasoning` option with a summary yields readable thinking text.

    The portable level does not request a summary, so it is asked for through
    the raw `reasoning` provider option (`summary="auto"`).  The reasoning item
    then carries summary text, which decodes into the `ThinkingPart` content.
    """

    # GIVEN settings whose raw reasoning option requests a summary
    settings = ModelSettings(
        model=_REASONING_MODEL,
        max_tokens=2048,
        provider_options={
            "openai": {"reasoning": {"effort": "high", "summary": "auto"}}
        },
    )

    # WHEN `complete` is awaited on a prompt that needs reasoning
    result = await openai_responses_provider.complete(
        [UserMessage.from_text("What is 17 * 23?  Reason step by step.")],
        settings,
    )

    # THEN a thinking block carries the summary text
    thinking_parts = [p for p in result.message.parts if isinstance(p, ThinkingPart)]
    assert thinking_parts
    assert any(p.content.strip() for p in thinking_parts)
