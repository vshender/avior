"""Tests for `avior.core.runner`."""

import pytest

from avior.core.agent import Agent
from avior.core.exceptions import (
    ContentFilterError,
    MaxTokensExceededError,
    ModelRefusalError,
)
from avior.core.messages import Message, TextPart
from avior.core.provider import ModelSettings
from avior.core.runner import Runner
from avior.core.testing import StubProvider


async def test_runner_run_returns_assistant_text_for_hello_smoke() -> None:
    """`Runner.run` returns the assistant's text for a hello prompt."""

    # GIVEN an agent whose provider always replies "Hi!"
    agent = Agent(
        provider=StubProvider.from_responses(["Hi!"]),
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model"),
    )

    # WHEN the runner is invoked with a hello prompt
    result = await Runner.run(agent, "hello")

    # THEN the returned string is the assistant's reply
    assert result == "Hi!"


async def test_runner_run_prepends_system_message_with_agent_instructions() -> None:
    """`Runner.run` sends `agent.instructions` as the first (system) message."""

    # GIVEN an agent with known instructions and a recording stub
    provider = StubProvider(lambda _msgs, _settings: "ok")
    agent = Agent(
        provider=provider,
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model"),
    )

    # WHEN the runner is invoked
    await Runner.run(agent, "hello")

    # THEN the first message sent to the provider is a system message
    # carrying the agent's instructions
    sent_messages = provider.calls[-1].messages
    assert sent_messages[0].role == "system"
    assert sent_messages[0].parts == [TextPart(text="you are helpful")]


async def test_runner_run_appends_user_message_with_input() -> None:
    """`Runner.run` sends the prompt as a user message after the system one."""

    # GIVEN an agent backed by a recording stub
    provider = StubProvider(lambda _msgs, _settings: "ok")
    agent = Agent(
        provider=provider,
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model"),
    )

    # WHEN the runner is invoked with a specific prompt
    await Runner.run(agent, "what is 2+2?")

    # THEN the second message sent is a user message carrying the prompt,
    # and exactly two messages are sent (system + user)
    sent_messages = provider.calls[-1].messages
    assert len(sent_messages) == 2
    assert sent_messages[1].role == "user"
    assert sent_messages[1].parts == [TextPart(text="what is 2+2?")]


async def test_runner_run_passes_agent_model_settings_to_provider() -> None:
    """`Runner.run` forwards `agent.model_settings` to the provider as-is."""

    # GIVEN an agent with specific model settings
    settings = ModelSettings(model="claude-3-5-sonnet", temperature=0.2, max_tokens=512)
    provider = StubProvider(lambda _msgs, _settings: "ok")
    agent = Agent(
        provider=provider,
        instructions="you are helpful",
        model_settings=settings,
    )

    # WHEN the runner is invoked
    await Runner.run(agent, "hello")

    # THEN the provider receives the same settings object by identity
    assert provider.calls[-1].settings is settings


async def test_runner_run_returns_empty_string_when_response_has_no_text() -> None:
    """`Runner.run` returns `""` when the response has no text parts."""

    # GIVEN an agent whose provider replies with an empty assistant message
    empty_response = Message(role="assistant", parts=[])
    agent = Agent(
        provider=StubProvider(lambda _msgs, _settings: empty_response),
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model"),
    )

    # WHEN the runner is invoked
    result = await Runner.run(agent, "hello")

    # THEN the result is an empty string (not `None`)
    assert result == ""


async def test_runner_run_raises_on_max_tokens_stop_reason() -> None:
    """`Runner.run` raises `MaxTokensExceededError` on max-tokens stop."""

    # GIVEN an agent whose provider returns a message marked `max_tokens`
    truncated = Message(role="assistant", parts=[], stop_reason="max_tokens")
    agent = Agent(
        provider=StubProvider(lambda _msgs, _settings: truncated),
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model", max_tokens=64),
    )

    # WHEN `Runner.run` is invoked
    # THEN it raises `MaxTokensExceededError`
    with pytest.raises(MaxTokensExceededError):
        await Runner.run(agent, "hello")


async def test_runner_run_max_tokens_message_omits_none_when_unset() -> None:
    """The exception message stays human-readable when `max_tokens` is `None`.

    When the user hasn't set `max_tokens` explicitly but the provider's default
    cap was hit, the exception text must not say "budget (None)" - it should
    describe the default-cap situation in actionable terms.
    """

    # GIVEN an agent with no explicit `max_tokens`, whose provider truncated
    truncated = Message(role="assistant", parts=[], stop_reason="max_tokens")
    agent = Agent(
        provider=StubProvider(lambda _msgs, _settings: truncated),
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model"),  # max_tokens=None
    )

    # WHEN `Runner.run` is invoked
    # THEN the exception is raised and the message does not leak "None"
    with pytest.raises(MaxTokensExceededError) as exc_info:
        await Runner.run(agent, "hello")
    assert "None" not in str(exc_info.value)


async def test_runner_run_raises_on_content_filter_stop_reason() -> None:
    """`Runner.run` raises `ContentFilterError` on content-filter stop."""

    # GIVEN an agent whose provider returns a message marked `content_filter`
    filtered = Message(role="assistant", parts=[], stop_reason="content_filter")
    agent = Agent(
        provider=StubProvider(lambda _msgs, _settings: filtered),
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model"),
    )

    # WHEN `Runner.run` is invoked
    # THEN it raises `ContentFilterError`
    with pytest.raises(ContentFilterError):
        await Runner.run(agent, "hello")


async def test_runner_run_raises_on_refusal_stop_reason() -> None:
    """`Runner.run` raises `ModelRefusalError` carrying the refusal text."""

    # GIVEN an agent whose provider returns a refusal-marked message
    refusal_text = "I can't help with that."
    refusal = Message(
        role="assistant",
        parts=[TextPart(text=refusal_text)],
        stop_reason="refusal",
    )
    agent = Agent(
        provider=StubProvider(lambda _msgs, _settings: refusal),
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model"),
    )

    # WHEN `Runner.run` is invoked
    # THEN `ModelRefusalError` is raised with the model's refusal text
    # preserved on the exception
    with pytest.raises(ModelRefusalError) as exc_info:
        await Runner.run(agent, "hello")
    assert exc_info.value.refusal_text == refusal_text


async def test_runner_run_accepts_normal_stop_reason() -> None:
    """`Runner.run` returns text when `stop_reason` is the normal `"stop"`."""

    # GIVEN an agent whose provider returns a normal completion
    normal = Message(
        role="assistant",
        parts=[TextPart(text="Hi!")],
        stop_reason="stop",
    )
    agent = Agent(
        provider=StubProvider(lambda _msgs, _settings: normal),
        instructions="you are helpful",
        model_settings=ModelSettings(model="test-model"),
    )

    # WHEN `Runner.run` is invoked
    result = await Runner.run(agent, "hello")

    # THEN the assistant's text is returned
    assert result == "Hi!"
