"""Integration smoke tests against the real Gemini API.

Gated by `GOOGLE_API_KEY` / `GEMINI_API_KEY`; skipped when neither is set.
Not run by the default `make test` / unit-test path - invoked separately via
`make test-integration` and a dedicated GitHub Actions workflow.
"""

import os

import pytest
from pydantic import BaseModel

from avior.core import Agent, ModelSettings, Runner
from avior.core.context import RunContext
from avior.core.exceptions import MaxTokensExceededError, ProviderHTTPError
from avior.core.messages import (
    AssistantMessage,
    Message,
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
