"""Integration smoke tests against the real Gemini API.

Gated by `GOOGLE_API_KEY` / `GEMINI_API_KEY`; skipped when neither is set.
Not run by the default `make test` / unit-test path - invoked separately via
`make test-integration` and a dedicated GitHub Actions workflow.
"""

import base64
import os

import pytest
from pydantic import BaseModel

from avior.core import Agent, ModelSettings, Runner
from avior.core.context import RunContext
from avior.core.exceptions import MaxTokensExceededError, ProviderHTTPError
from avior.core.messages import (
    AssistantMessage,
    Message,
    ThinkingPart,
    ToolCallPart,
    ToolMessage,
    ToolResultOk,
    ToolResultPart,
    UserMessage,
)
from avior.core.tools import Tool
from avior.providers.gemini import GeminiProvider

pytestmark = pytest.mark.skipif(
    not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")),
    reason="GOOGLE_API_KEY / GEMINI_API_KEY not set",
)

_MODEL = "gemini-2.5-flash"

# A model whose thinking config uses the `thinking_level` dialect and whose
# generation validates replayed thought signatures, unlike the budget-dialect
# `_MODEL`.
_LEVEL_MODEL = "gemini-3.6-flash"

# A model that does not think unless the config turns thinking on.
_OFF_BY_DEFAULT_MODEL = "gemini-3.1-flash-lite"

# A model whose thinking is always on and cannot be disabled.
_PRO_MODEL = "gemini-3.1-pro-preview"


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


async def test_runner_run_against_gemini_returns_non_empty_text(
    gemini_provider: GeminiProvider,
) -> None:
    """`Runner.run` against real Gemini returns a non-empty assistant reply.

    End-to-end smoke: avior `Agent` -> `GeminiProvider` -> `google-genai` SDK ->
    HTTP -> Gemini API -> decoded `Message`.  Asserts only on the transport
    contract (a non-empty string), not on response content.
    """

    # GIVEN an agent and a runner using the real Gemini provider
    agent = Agent(
        instructions="Reply with one short word.",
        model_settings=ModelSettings(model=_MODEL, max_tokens=256),
    )

    # WHEN we run a trivial prompt
    result = await Runner(provider=gemini_provider).run(agent, "Say hello.")

    # THEN we get a non-empty text response
    assert result.output.strip() != ""


async def test_runner_run_raises_max_tokens_exceeded_against_gemini(
    gemini_provider: GeminiProvider,
) -> None:
    """`Runner.run` raises `MaxTokensExceededError` when the token cap is hit.

    Confirms end-to-end mapping: Gemini returns `finish_reason=MAX_TOKENS` ->
    provider sets canonical `stop_reason="max_tokens"` -> Runner raises.
    """

    # GIVEN an agent with `model_settings.max_tokens` too small to complete
    agent = Agent(
        instructions="Write a long story.",
        model_settings=ModelSettings(model=_MODEL, max_tokens=1),
    )

    # WHEN `Runner.run` is invoked
    # THEN `MaxTokensExceededError` is raised
    with pytest.raises(MaxTokensExceededError):
        await Runner(provider=gemini_provider).run(agent, "Tell me a story.")


async def test_runner_run_against_gemini_calls_a_tool_end_to_end(
    gemini_provider: GeminiProvider,
) -> None:
    """`Runner.run` drives a full tool round-trip against real Gemini.

    Exercises the wire contract that mocks cannot: Gemini accepts our tool
    declaration, returns a `function_call` part, accepts the echoed call plus
    `function_response` on the continuation request, and produces a final answer
    relaying the tool's result.
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
    result = await Runner(provider=gemini_provider).run(
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


async def test_complete_enables_thinking_against_gemini(
    gemini_provider: GeminiProvider,
) -> None:
    """The portable `thinking` setting turns thinking on for a model that does
    not think by default, visible as a non-zero reasoning-token count.
    """

    # GIVEN settings enabling thinking on an off-by-default model
    settings = ModelSettings(
        model=_OFF_BY_DEFAULT_MODEL, max_tokens=4096, thinking="low"
    )

    # WHEN `complete` is awaited on a prompt with room to reason
    response = await gemini_provider.complete(
        [UserMessage.from_text("What is 17 * 23?  Answer with the number only.")],
        settings,
    )

    # THEN the model spent reasoning tokens
    assert response.usage is not None
    assert response.usage.reasoning_tokens is not None
    assert response.usage.reasoning_tokens > 0


@pytest.mark.parametrize(
    "model",
    [_MODEL, _LEVEL_MODEL],
    ids=["budget-dialect", "level-dialect"],
)
async def test_complete_disables_thinking_against_gemini(
    gemini_provider: GeminiProvider,
    model: str,
) -> None:
    """`thinking=False` turns thinking off for a model that thinks by
    default, visible as a zero reasoning-token count.
    """

    # GIVEN settings disabling thinking on an on-by-default model
    settings = ModelSettings(model=model, max_tokens=4096, thinking=False)

    # WHEN `complete` is awaited
    response = await gemini_provider.complete(
        [UserMessage.from_text("What is 17 * 23?  Answer with the number only.")],
        settings,
    )

    # THEN the model spent no reasoning tokens
    assert response.usage is not None
    assert response.usage.reasoning_tokens == 0


@pytest.mark.parametrize(
    ("model", "thinks_by_default"),
    [
        (_MODEL, True),
        (_LEVEL_MODEL, True),
        (_OFF_BY_DEFAULT_MODEL, False),
        (_PRO_MODEL, True),
        ("gemini-flash-latest", True),
        ("gemini-flash-lite-latest", False),
    ],
    ids=[
        "budget-on-by-default",
        "level-on-by-default",
        "level-off-by-default",
        "pro-always-on",
        "moving-alias-flash",
        "moving-alias-flash-lite",
    ],
)
async def test_complete_default_thinking_matches_classification_against_gemini(
    gemini_provider: GeminiProvider,
    model: str,
    thinks_by_default: bool,
) -> None:
    """With `thinking` unset, whether the model thinks matches avior's
    classification of its default.

    This is a drift alarm: if Google changes a model's server-side default,
    the classification in `_THINKING_MODELS` becomes wrong and this test
    fails.
    """

    # GIVEN settings with no thinking value for the classified model
    settings = ModelSettings(model=model, max_tokens=4096)

    # WHEN `complete` is awaited on a prompt with room to reason
    response = await gemini_provider.complete(
        [UserMessage.from_text("What is 17 * 23?  Answer with the number only.")],
        settings,
    )

    # THEN reasoning-token spend matches the classified default
    assert response.usage is not None
    assert response.usage.reasoning_tokens is not None
    assert (response.usage.reasoning_tokens > 0) is thinks_by_default


async def test_complete_returns_thinking_summary_against_gemini(
    gemini_provider: GeminiProvider,
) -> None:
    """A raw `thinking_config` with `include_thoughts` returns thought
    summaries, decoded into a `ThinkingPart` with readable content.
    """

    # GIVEN settings whose raw thinking config asks for thought summaries
    settings = ModelSettings(
        model=_MODEL,
        max_tokens=4096,
        provider_options={
            "gemini": {
                "thinking_config": {"thinking_budget": 2048, "include_thoughts": True}
            }
        },
    )

    # WHEN `complete` is awaited on a prompt with room to reason
    response = await gemini_provider.complete(
        [UserMessage.from_text("What is 17 * 23?  Answer with the number only.")],
        settings,
    )

    # THEN a thinking part with readable content is present
    thinking_parts = [
        part for part in response.message.parts if isinstance(part, ThinkingPart)
    ]
    assert thinking_parts
    assert any(part.content.strip() for part in thinking_parts)


async def test_complete_keeps_signature_on_tool_call_against_gemini(
    gemini_provider: GeminiProvider,
) -> None:
    """A thinking model's `function_call` carries a thought signature, stored
    on the decoded `ToolCallPart`'s `provider_details`.
    """

    # GIVEN a thinking model offered a tool it must call
    tool = _MagicNumber()
    settings = ModelSettings(model=_LEVEL_MODEL, max_tokens=4096)

    # WHEN `complete` is awaited on a prompt that requires the tool
    response = await gemini_provider.complete(
        [UserMessage.from_text("What is the magic number for Paris?")],
        settings,
        tools=[tool],
    )

    # THEN the decoded tool call keeps the signature for the round trip,
    # stored as non-empty base64 text
    tool_calls = [
        part for part in response.message.parts if isinstance(part, ToolCallPart)
    ]
    assert tool_calls
    details = tool_calls[0].provider_details
    assert details is not None
    signature = details.get("thought_signature")
    assert isinstance(signature, str)
    assert base64.b64decode(signature, validate=True) != b""


async def test_complete_replays_unsigned_tool_turn_against_gemini(
    gemini_provider: GeminiProvider,
) -> None:
    """A hand-built tool-call turn with no thought signature replays cleanly
    on a signature-validating model.

    The adapter stamps Google's documented skip placeholder on the unsigned
    call; if Google ever withdraws the placeholder, hand-built and
    cross-provider transcripts stop replaying and this test fails first.
    """

    # GIVEN a hand-built transcript whose tool-call turn carries no signature
    tool = _MagicNumber()
    history: list[Message] = [
        UserMessage.from_text("What is the magic number for Paris?"),
        AssistantMessage(
            parts=[
                ToolCallPart(
                    call_id="call_1",
                    tool_name="get_magic_number",
                    args={"city": "Paris"},
                )
            ],
            stop_reason="tool_use",
        ),
        ToolMessage(
            parts=[
                ToolResultPart(call_id="call_1", result=ToolResultOk(content="4242"))
            ]
        ),
    ]

    # WHEN `complete` is awaited on a signature-validating model
    response = await gemini_provider.complete(
        history,
        ModelSettings(model=_LEVEL_MODEL, max_tokens=4096),
        tools=[tool],
    )

    # THEN the replay is accepted and the answer relays the tool's result
    assert response.message.text is not None
    assert "4242" in response.message.text


async def test_runner_run_thinking_tool_chain_against_gemini(
    gemini_provider: GeminiProvider,
) -> None:
    """`Runner.run` drives a multi-step tool chain on a thinking Gemini model.

    The model signs its `function_call` parts and rejects a continuation
    request that replays any model turn without its signature, so this chain
    only completes if every signature survives the decode -> transcript ->
    encode round trip on each step.
    """

    # GIVEN a thinking model instructed to call the tool once per city
    tool = _MagicNumber()
    agent = Agent(
        instructions=(
            "When asked for magic numbers, call the get_magic_number tool "
            "once per city, one call at a time, then state the numbers it "
            "returned."
        ),
        model_settings=ModelSettings(model=_LEVEL_MODEL, max_tokens=4096),
        tools=[tool],
    )

    # WHEN we run a prompt that needs the tool for two cities
    result = await Runner(provider=gemini_provider).run(
        agent, "What are the magic numbers for Paris and for London?"
    )

    # THEN the tool ran for both cities and the answer relays its result
    assert sorted(tool.calls) == ["London", "Paris"]
    assert "4242" in result.output

    # AND every tool-calling turn's first call carried its signature through
    # the transcript (the model signs the first `function_call` of each turn;
    # later parallel calls may stay unsigned)
    for message in result.messages:
        if isinstance(message, AssistantMessage):
            calls = [p for p in message.parts if isinstance(p, ToolCallPart)]
            if calls:
                details = calls[0].provider_details
                assert details is not None
                assert "thought_signature" in details


async def test_complete_orphaned_tool_result_is_rejected_by_gemini(
    gemini_provider: GeminiProvider,
) -> None:
    """An orphaned tool result surfaces as `ProviderHTTPError`.

    avior does not pre-validate transcript structure.  A tool result whose
    `call_id` matches no prior tool call carries no recoverable tool name and
    yields an invalid request; Gemini rejects it with HTTP 400, mapped to
    `ProviderHTTPError` - the same category the other adapters surface for this
    caller mistake.
    """

    # GIVEN a transcript whose tool result references a call_id absent from it
    history: list[Message] = [
        UserMessage.from_text("What is the magic number for Paris?"),
        ToolMessage(
            parts=[
                ToolResultPart(call_id="ghost", result=ToolResultOk(content="4242")),
            ]
        ),
    ]

    # WHEN `complete` is invoked
    # THEN Gemini's 400 for the invalid request surfaces as `ProviderHTTPError`
    with pytest.raises(ProviderHTTPError) as exc_info:
        await gemini_provider.complete(
            history, ModelSettings(model=_MODEL, max_tokens=64)
        )
    assert exc_info.value.status_code == 400
