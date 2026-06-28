"""Tests for `avior.core.runner`."""

from collections.abc import Sequence

import pytest

from avior.core.agent import Agent
from avior.core.exceptions import (
    ContentFilterError,
    EmptyInputError,
    MaxTokensExceededError,
    ModelRefusalError,
    OrphanedToolResultError,
    UnansweredToolCallError,
    UnexpectedModelBehaviorError,
)
from avior.core.messages import (
    AssistantMessage,
    Message,
    TextPart,
    ToolCallPart,
    ToolMessage,
    ToolResultOk,
    ToolResultPart,
    UserMessage,
)
from avior.core.provider import ModelSettings, ProviderResponse
from avior.core.runner import Runner
from avior.core.testing import StubProvider
from avior.core.usage import Usage

# Basic run tests
# -----------------------------------------------------------------------------


async def test_runner_run_returns_assistant_text_for_hello_smoke() -> None:
    """`Runner.run` returns the assistant's text for a hello prompt."""

    # GIVEN an agent and a runner whose provider replies "Hi!"
    agent = Agent(
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model"),
    )
    runner = Runner(provider=StubProvider.from_responses(["Hi!"]))

    # WHEN the runner is invoked with a hello prompt
    result = await runner.run(agent, "hello")

    # THEN the result's output is the assistant's reply
    assert result.output == "Hi!"


# Input-validation tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_input",
    [
        pytest.param("", id="empty-str"),
        pytest.param("   ", id="whitespace-str"),
        pytest.param([], id="empty-list"),
        pytest.param([UserMessage.from_text("")], id="empty-user-message"),
        pytest.param(
            [UserMessage.from_text("hi"), UserMessage.from_text("")],
            id="contentful-then-empty",
        ),
        pytest.param(
            [AssistantMessage(parts=[], stop_reason="stop")], id="empty-assistant"
        ),
        pytest.param([ToolMessage(parts=[])], id="empty-tool"),
    ],
)
async def test_runner_run_rejects_empty_input(
    bad_input: str | Sequence[Message],
) -> None:
    """`Runner.run` raises `EmptyInputError` for input carrying no content.

    Covers the whole input being empty and an empty message among contentful
    ones - both would otherwise reach a provider as a request with nothing to
    send.
    """

    # GIVEN an agent and a runner whose provider is a recording stub
    agent = Agent(model_settings=ModelSettings(model="test-model"))
    provider = StubProvider.from_responses(["unused"])
    runner = Runner(provider=provider)

    # WHEN `Runner.run` is invoked with empty input
    # THEN `EmptyInputError` is raised, and the provider was never called
    with pytest.raises(EmptyInputError):
        await runner.run(agent, bad_input)
    assert provider.calls == []


async def test_runner_run_rejects_orphaned_tool_result() -> None:
    """`Runner.run` raises `OrphanedToolResultError` for an unmatched result.

    A tool result whose `call_id` matches no tool call in the input cannot be
    correlated with a request, so it is rejected up front rather than left to a
    confusing provider error.
    """

    # GIVEN a transcript whose tool result references no tool call
    history: list[Message] = [
        UserMessage.from_text("weather?"),
        ToolMessage(
            parts=[ToolResultPart(call_id="ghost", result=ToolResultOk(content="42"))]
        ),
    ]
    agent = Agent(model_settings=ModelSettings(model="test-model"))
    provider = StubProvider.from_responses(["unused"])
    runner = Runner(provider=provider)

    # WHEN `Runner.run` is invoked
    # THEN `OrphanedToolResultError` is raised, and the provider was never
    # called
    with pytest.raises(OrphanedToolResultError):
        await runner.run(agent, history)
    assert provider.calls == []


async def test_runner_run_accepts_tool_result_matching_a_prior_call() -> None:
    """A tool result matching a prior tool call passes validation and runs."""

    # GIVEN a continuation whose tool result references a present tool call
    history: list[Message] = [
        UserMessage.from_text("weather?"),
        AssistantMessage(
            parts=[ToolCallPart(call_id="c1", tool_name="get_weather", args={})],
            stop_reason="tool_use",
        ),
        ToolMessage(
            parts=[ToolResultPart(call_id="c1", result=ToolResultOk(content="sunny"))]
        ),
    ]
    agent = Agent(model_settings=ModelSettings(model="test-model"))
    runner = Runner(provider=StubProvider.from_responses(["Final answer"]))

    # WHEN `Runner.run` is invoked
    result = await runner.run(agent, history)

    # THEN validation passes and the run produces the provider's reply
    assert result.output == "Final answer"


async def test_runner_run_rejects_unanswered_tool_call() -> None:
    """`Runner.run` raises `UnansweredToolCallError` for an unanswered call.

    A tool call with no matching tool result in the input cannot be continued -
    the model expects every call it made to be answered.
    """

    # GIVEN a transcript whose tool call is never answered by a result
    history: list[Message] = [
        UserMessage.from_text("weather?"),
        AssistantMessage(
            parts=[ToolCallPart(call_id="c1", tool_name="get_weather", args={})],
            stop_reason="tool_use",
        ),
    ]
    agent = Agent(model_settings=ModelSettings(model="test-model"))
    provider = StubProvider.from_responses(["unused"])
    runner = Runner(provider=provider)

    # WHEN `Runner.run` is invoked
    # THEN `UnansweredToolCallError` is raised, and the provider was never
    # called
    with pytest.raises(UnansweredToolCallError):
        await runner.run(agent, history)
    assert provider.calls == []


async def test_runner_run_accepts_tool_calls_answered_across_messages() -> None:
    """Tool calls may be answered across separate tool messages, not just one.

    Pairing is by `call_id` coverage, not packaging, so two calls answered by
    two separate tool messages pass validation.
    """

    # GIVEN a transcript whose two tool calls are answered by two separate tool
    # messages
    history: list[Message] = [
        UserMessage.from_text("weather and time?"),
        AssistantMessage(
            parts=[
                ToolCallPart(call_id="c1", tool_name="get_weather", args={}),
                ToolCallPart(call_id="c2", tool_name="get_time", args={}),
            ],
            stop_reason="tool_use",
        ),
        ToolMessage(
            parts=[ToolResultPart(call_id="c1", result=ToolResultOk(content="sunny"))]
        ),
        ToolMessage(
            parts=[ToolResultPart(call_id="c2", result=ToolResultOk(content="noon"))]
        ),
    ]
    agent = Agent(model_settings=ModelSettings(model="test-model"))
    runner = Runner(provider=StubProvider.from_responses(["Final answer"]))

    # WHEN `Runner.run` is invoked
    result = await runner.run(agent, history)

    # THEN validation passes and the run produces the provider's reply
    assert result.output == "Final answer"


# System-prompt tests
# -----------------------------------------------------------------------------


async def test_runner_run_passes_agent_instructions_as_system_prompt() -> None:
    """`Runner.run` passes `agent.instructions` as the system prompt."""

    # GIVEN an agent and a runner whose provider is a recording stub
    agent = Agent(
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model"),
    )
    provider = StubProvider.from_responses(["ok"])
    runner = Runner(provider=provider)

    # WHEN the runner is invoked
    await runner.run(agent, "hello")

    # THEN the instructions are passed as the system prompt, not as a message
    assert provider.calls[-1].system_prompt == "you are helpful"


async def test_runner_run_passes_no_system_prompt_when_instructions_omitted() -> None:
    """`Runner.run` sends no system prompt when there are no instructions."""

    # GIVEN an agent with no instructions
    agent = Agent(model_settings=ModelSettings(model="test-model"))
    provider = StubProvider.from_responses(["ok"])
    runner = Runner(provider=provider)

    # WHEN the runner is invoked
    await runner.run(agent, "hello")

    # THEN no system prompt is sent
    assert provider.calls[-1].system_prompt is None


@pytest.mark.parametrize("instructions", ["", "   ", "\n\t"])
async def test_runner_run_normalizes_blank_instructions_to_none(
    instructions: str,
) -> None:
    """`Runner.run` sends `None` when `instructions` is blank or whitespace.

    The blank-instructions-means-no-system-prompt contract is enforced here,
    in the one place that interprets `instructions`, so every provider (the
    stub included) observes `None` rather than a meaningless blank string.
    """

    # GIVEN an agent whose instructions are blank or whitespace-only
    agent = Agent(
        instructions=instructions,
        model_settings=ModelSettings(model="test-model"),
    )
    provider = StubProvider.from_responses(["ok"])
    runner = Runner(provider=provider)

    # WHEN the runner is invoked
    await runner.run(agent, "hello")

    # THEN the provider receives no system prompt
    assert provider.calls[-1].system_prompt is None


# Provider request tests
# -----------------------------------------------------------------------------


async def test_runner_run_sends_only_the_user_message_as_input() -> None:
    """`Runner.run` sends the prompt as a user message, not a system one."""

    # GIVEN an agent and a runner whose provider is a recording stub
    agent = Agent(
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model"),
    )
    provider = StubProvider.from_responses(["ok"])
    runner = Runner(provider=provider)

    # WHEN the runner is invoked with a specific prompt
    await runner.run(agent, "what is 2+2?")

    # THEN exactly one message is sent: the user prompt (no system prompt in
    # the transcript)
    sent_messages = provider.calls[-1].messages
    assert len(sent_messages) == 1
    assert sent_messages[0] == UserMessage.from_text("what is 2+2?")


async def test_runner_run_passes_agent_model_settings_to_provider() -> None:
    """`Runner.run` forwards `agent.model_settings` to the provider as-is."""

    # GIVEN an agent with specific model settings, and a runner whose provider
    # is a recording stub
    settings = ModelSettings(model="claude-3-5-sonnet", temperature=0.2, max_tokens=512)
    agent = Agent(
        instructions="you are helpful",
        model_settings=settings,
    )
    provider = StubProvider.from_responses(["ok"])
    runner = Runner(provider=provider)

    # WHEN the runner is invoked
    await runner.run(agent, "hello")

    # THEN the provider receives the same settings object by identity
    assert provider.calls[-1].settings is settings


# Model-response handling tests
# -----------------------------------------------------------------------------


async def test_runner_run_raises_on_empty_response() -> None:
    """`Runner.run` raises `UnexpectedModelBehaviorError` on an empty response.

    A response with no parts - no text and no tool calls - is degenerate, not a
    usable answer, so the run surfaces it instead of returning an empty result.
    """

    # GIVEN an agent and a runner whose provider replies with an empty assistant
    # message
    agent = Agent(
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model"),
    )
    empty_response = AssistantMessage(parts=[], stop_reason="stop")
    runner = Runner(provider=StubProvider.from_responses([empty_response]))

    # WHEN `Runner.run` is invoked
    # THEN it raises `UnexpectedModelBehaviorError`
    with pytest.raises(UnexpectedModelBehaviorError):
        await runner.run(agent, "hello")


async def test_runner_run_raises_on_max_tokens_stop_reason() -> None:
    """`Runner.run` raises `MaxTokensExceededError` on max-tokens stop."""

    # GIVEN an agent and a runner whose provider returns a message marked
    # `max_tokens`
    agent = Agent(
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model", max_tokens=64),
    )
    truncated = AssistantMessage(parts=[], stop_reason="max_tokens")
    runner = Runner(provider=StubProvider.from_responses([truncated]))

    # WHEN `Runner.run` is invoked
    # THEN it raises `MaxTokensExceededError`
    with pytest.raises(MaxTokensExceededError):
        await runner.run(agent, "hello")


async def test_runner_run_max_tokens_message_omits_none_when_unset() -> None:
    """The exception message stays human-readable when `max_tokens` is `None`.

    When the user hasn't set `max_tokens` explicitly but the provider's default
    cap was hit, the exception text must not say "budget (None)" - it should
    describe the default-cap situation in actionable terms.
    """

    # GIVEN an agent with no explicit `max_tokens` and a provider that returns
    # a message marked `max_tokens`
    agent = Agent(
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model"),  # max_tokens=None
    )
    truncated = AssistantMessage(parts=[], stop_reason="max_tokens")
    runner = Runner(provider=StubProvider.from_responses([truncated]))

    # WHEN `Runner.run` is invoked
    # THEN the exception is raised, the message points at `max_tokens` as the
    # actionable knob, and it does not leak "None"
    with pytest.raises(MaxTokensExceededError) as exc_info:
        await runner.run(agent, "hello")
    message = str(exc_info.value)
    assert "max_tokens" in message
    assert "None" not in message


async def test_runner_run_raises_on_content_filter_stop_reason() -> None:
    """`Runner.run` raises `ContentFilterError` on content-filter stop."""

    # GIVEN an agent and a runner whose provider returns a message marked
    # `content_filter`
    agent = Agent(
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model"),
    )
    filtered = AssistantMessage(parts=[], stop_reason="content_filter")
    runner = Runner(provider=StubProvider.from_responses([filtered]))

    # WHEN `Runner.run` is invoked
    # THEN it raises `ContentFilterError`
    with pytest.raises(ContentFilterError):
        await runner.run(agent, "hello")


async def test_runner_run_raises_on_refusal_stop_reason() -> None:
    """`Runner.run` raises `ModelRefusalError` carrying the refusal text."""

    # GIVEN an agent and a runner whose provider returns a refusal-marked
    # message
    agent = Agent(
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model"),
    )
    refusal_text = "I can't help with that."
    refusal = AssistantMessage(
        parts=[TextPart(text=refusal_text)],
        stop_reason="refusal",
    )
    runner = Runner(provider=StubProvider.from_responses([refusal]))

    # WHEN `Runner.run` is invoked
    # THEN `ModelRefusalError` is raised with the model's refusal text
    # preserved on the exception
    with pytest.raises(ModelRefusalError) as exc_info:
        await runner.run(agent, "hello")
    assert exc_info.value.refusal_text == refusal_text


async def test_runner_run_raises_on_error_stop_reason() -> None:
    """`Runner.run` raises `UnexpectedModelBehaviorError` on `"error"`."""

    # GIVEN an agent and a runner whose provider returns a message marked
    # `error` (an abnormal termination, e.g. a malformed tool call)
    agent = Agent(
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model"),
    )
    errored = AssistantMessage(parts=[], stop_reason="error")
    runner = Runner(provider=StubProvider.from_responses([errored]))

    # WHEN `Runner.run` is invoked
    # THEN it raises `UnexpectedModelBehaviorError`
    with pytest.raises(UnexpectedModelBehaviorError):
        await runner.run(agent, "hello")


async def test_runner_run_accepts_normal_stop_reason() -> None:
    """`Runner.run` returns text when `stop_reason` is the normal `"stop"`."""

    # GIVEN an agent and a runner whose provider returns a normal completion
    agent = Agent(
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model"),
    )
    normal = AssistantMessage(
        parts=[TextPart(text="Hi!")],
        stop_reason="stop",
    )
    runner = Runner(provider=StubProvider.from_responses([normal]))

    # WHEN `Runner.run` is invoked
    result = await runner.run(agent, "hello")

    # THEN the assistant's text is returned as the output
    assert result.output == "Hi!"


# Usage tests
# -----------------------------------------------------------------------------


async def test_runner_run_carries_usage_from_provider_response() -> None:
    """`Runner.run` surfaces the provider response's usage on the result."""

    # GIVEN an agent and a runner whose provider reports token usage for its
    # call
    agent = Agent(
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model"),
    )
    usage = Usage(input_tokens=11, output_tokens=7)
    response = ProviderResponse(
        message=AssistantMessage(parts=[TextPart(text="Hi!")], stop_reason="stop"),
        usage=usage,
    )
    runner = Runner(provider=StubProvider.from_responses([response]))

    # WHEN the runner is invoked
    result = await runner.run(agent, "hello")

    # THEN the run result carries that usage
    assert result.usage == usage


async def test_runner_run_usage_is_none_when_provider_reports_none() -> None:
    """`Runner.run` leaves `usage` as `None` when the provider reports none."""

    # GIVEN an agent and a runner whose provider's response carries no usage
    agent = Agent(
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model"),
    )
    runner = Runner(provider=StubProvider.from_responses(["Hi!"]))

    # WHEN the runner is invoked
    result = await runner.run(agent, "hello")

    # THEN the result's usage is `None`
    assert result.usage is None


# Result and conversation-threading tests
# -----------------------------------------------------------------------------


async def test_runner_run_result_excludes_system_prompt_marks_new_turn() -> None:
    """The result stores input plus reply, but not the system prompt."""

    # GIVEN an agent and a runner whose provider replies "Hi!"
    agent = Agent(
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model"),
    )
    runner = Runner(provider=StubProvider.from_responses(["Hi!"]))

    # WHEN the runner is invoked with a string prompt
    result = await runner.run(agent, "hello")

    # THEN the transcript is the input user turn followed by the assistant
    # reply, with no system prompt
    assert result.messages == [
        UserMessage.from_text("hello"),
        AssistantMessage(
            parts=[TextPart(text="Hi!")],
            stop_reason="stop",
            provider_name="stub",
        ),
    ]
    # AND `new_messages` is just the run's output
    assert result.new_messages == [
        AssistantMessage(
            parts=[TextPart(text="Hi!")],
            stop_reason="stop",
            provider_name="stub",
        ),
    ]


async def test_runner_run_accepts_message_list_input() -> None:
    """`Runner.run` accepts existing conversation messages as input."""

    # GIVEN an agent, a runner whose provider is a recording stub, and a prior
    # conversation
    agent = Agent(
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model"),
    )
    provider = StubProvider.from_responses(["I'm well."])
    runner = Runner(provider=provider)
    history: list[Message] = [
        UserMessage.from_text("hi"),
        AssistantMessage(parts=[TextPart(text="Hello!")], stop_reason="stop"),
        UserMessage.from_text("how are you?"),
    ]

    # WHEN the runner is invoked with that transcript as input
    result = await runner.run(agent, history)

    # THEN the provider receives the whole input as the conversation, with the
    # instructions carried separately as the system prompt
    assert provider.calls[-1].messages == history
    assert provider.calls[-1].system_prompt == "you are helpful"

    # AND the result extends the transcript with the new reply, marking it new
    new_reply = AssistantMessage(
        parts=[TextPart(text="I'm well.")],
        stop_reason="stop",
        provider_name="stub",
    )
    assert result.messages == [*history, new_reply]
    assert result.new_messages == [new_reply]


async def test_runner_run_threads_result_messages_into_next_run() -> None:
    """A result's `messages` can be passed back as the next run's input."""

    # GIVEN an agent and a runner whose provider replies "A1" then "A2"
    agent = Agent(
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model"),
    )
    provider = StubProvider.from_responses(["A1", "A2"])
    runner = Runner(provider=provider)

    # WHEN a first run is continued by threading its messages plus a new turn
    first = await runner.run(agent, "Q1")
    second = await runner.run(agent, [*first.messages, UserMessage.from_text("Q2")])

    # THEN the second call carries the full prior conversation (no system
    # prompt in the transcript; instructions ride separately)
    assert provider.calls[-1].messages == [
        UserMessage.from_text("Q1"),
        AssistantMessage(
            parts=[TextPart(text="A1")],
            stop_reason="stop",
            provider_name="stub",
        ),
        UserMessage.from_text("Q2"),
    ]
    # AND the second result's `new_messages` is only its own reply
    assert second.new_messages == [
        AssistantMessage(
            parts=[TextPart(text="A2")],
            stop_reason="stop",
            provider_name="stub",
        ),
    ]
