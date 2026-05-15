"""Tests for `avior.core.runner`."""

from avior.core.agent import Agent
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
