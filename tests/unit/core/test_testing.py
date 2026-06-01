"""Tests for `avior.core.testing`."""

from collections.abc import Sequence

import pytest

from avior.core.messages import AssistantMessage, Message, TextPart, UserMessage
from avior.core.provider import ModelSettings, ProviderResponse
from avior.core.testing import StubCall, StubProvider
from avior.core.usage import Usage


def _settings(model: str = "test-model") -> ModelSettings:
    """Construct a minimal `ModelSettings` for use in tests."""

    return ModelSettings(model=model)


def _assistant_message(text: str) -> AssistantMessage:
    """Build a single-`TextPart` assistant message with `stop_reason="stop"`."""

    return AssistantMessage(parts=[TextPart(text=text)], stop_reason="stop")


async def test_stub_provider_canonical_callable_returns_message_directly() -> None:
    """An `AssistantMessage` from the callable is wrapped, identity kept."""

    # GIVEN a stub whose callable returns a pre-built `AssistantMessage`
    response = _assistant_message("hi there")
    provider = StubProvider(lambda _msgs, _settings: response)

    # WHEN `complete` is called
    result = await provider.complete([UserMessage.from_text("hello")], _settings())

    # THEN the message is wrapped in a `ProviderResponse` by identity
    assert result.message is response


async def test_stub_provider_canonical_callable_wraps_string_responses() -> None:
    """A `str` from the callable is wrapped as a `ProviderResponse` message."""

    # GIVEN a stub whose callable returns a plain string
    provider = StubProvider(lambda _msgs, _settings: "hi")

    # WHEN `complete` is called
    result = await provider.complete([UserMessage.from_text("hello")], _settings())

    # THEN the response message carries that text
    assert result.message.text == "hi"


async def test_stub_provider_passes_scripted_provider_response_through() -> None:
    """A scripted `ProviderResponse` is returned unchanged, metadata intact."""

    # GIVEN a stub scripted with a full `ProviderResponse` carrying metadata
    response = ProviderResponse(
        message=_assistant_message("hi"),
        usage=Usage(input_tokens=11, output_tokens=3),
        response_id="resp_42",
        model="test-model",
        provider_name="stub",
    )
    provider = StubProvider.from_responses([response])

    # WHEN `complete` is called
    result = await provider.complete([UserMessage.from_text("hello")], _settings())

    # THEN the scripted `ProviderResponse` is returned as-is
    assert result is response


async def test_stub_provider_canonical_callable_awaits_coroutine_results() -> None:
    """An awaitable returned from the callable is awaited."""

    # GIVEN a stub whose callable is an async function
    async def async_func(_msgs: Sequence[Message], _settings: ModelSettings) -> str:
        return "async hi"

    provider = StubProvider(async_func)

    # WHEN `complete` is called
    result = await provider.complete([UserMessage.from_text("hello")], _settings())

    # THEN the awaited string is wrapped as a `ProviderResponse` message
    assert result.message.text == "async hi"


async def test_stub_provider_callable_receives_messages_and_settings() -> None:
    """The callable receives the same `messages` and `settings` by identity."""

    # GIVEN a stub callable that records its arguments
    received: list[StubCall] = []

    def func(messages: Sequence[Message], settings: ModelSettings) -> str:
        received.append(StubCall(messages=messages, settings=settings))
        return "ok"

    provider = StubProvider(func)
    messages: list[Message] = [UserMessage.from_text("hi")]
    settings = _settings("claude-3-5-sonnet")

    # WHEN `complete` is called
    await provider.complete(messages, settings)

    # THEN the callable observed the same arguments by identity
    assert len(received) == 1
    assert received[0].messages is messages
    assert received[0].settings is settings


async def test_stub_provider_records_each_call_in_order() -> None:
    """Each `complete` invocation is appended to `.calls` in order."""

    # GIVEN a stub that always returns the same response
    provider = StubProvider(lambda _msgs, _settings: "ok")

    # WHEN `complete` is called three times with different messages
    settings = _settings()
    await provider.complete([UserMessage.from_text("first")], settings)
    await provider.complete([UserMessage.from_text("second")], settings)
    await provider.complete([UserMessage.from_text("third")], settings)

    # THEN `.calls` contains all three invocations in order
    assert len(provider.calls) == 3
    assert provider.calls[0].messages[-1].text == "first"
    assert provider.calls[1].messages[-1].text == "second"
    assert provider.calls[2].messages[-1].text == "third"


async def test_stub_provider_from_responses_returns_responses_in_order() -> None:
    """`from_responses` yields each response one per call, in order."""

    # GIVEN a stub built from a list of canned responses
    provider = StubProvider.from_responses(["hello", "world"])

    # WHEN `complete` is called twice
    settings = _settings()
    first = await provider.complete([UserMessage.from_text("a")], settings)
    second = await provider.complete([UserMessage.from_text("b")], settings)

    # THEN each call returns the next response in order
    assert first.message.text == "hello"
    assert second.message.text == "world"


async def test_stub_provider_from_responses_accepts_mixed_str_and_message() -> None:
    """`from_responses` accepts both `str` and `AssistantMessage` entries."""

    # GIVEN a stub built from a heterogeneous response list
    canned_message = _assistant_message("from message")
    provider = StubProvider.from_responses(["from str", canned_message])

    # WHEN `complete` is called twice
    settings = _settings()
    first = await provider.complete([UserMessage.from_text("a")], settings)
    second = await provider.complete([UserMessage.from_text("b")], settings)

    # THEN strings are wrapped and `AssistantMessage` entries keep their
    # identity on the wrapping `ProviderResponse`
    assert first.message.text == "from str"
    assert second.message is canned_message


async def test_stub_provider_from_responses_raises_when_exhausted() -> None:
    """`from_responses` raises `AssertionError` once exhausted."""

    # GIVEN a stub built with a single response, already consumed by one call
    provider = StubProvider.from_responses(["only one"])
    settings = _settings()
    await provider.complete([UserMessage.from_text("a")], settings)

    # WHEN `complete` is called a second time
    # THEN it raises `AssertionError` with a descriptive message
    with pytest.raises(AssertionError, match="exhausted after 1 call"):
        await provider.complete([UserMessage.from_text("b")], settings)


async def test_stub_provider_from_responses_records_call_on_failure() -> None:
    """A call is recorded in `.calls` even when responses are exhausted."""

    # GIVEN an already-exhausted stub
    provider = StubProvider.from_responses(["only one"])
    await provider.complete([UserMessage.from_text("a")], _settings())

    # WHEN `complete` is called once more and raises
    with pytest.raises(AssertionError):
        await provider.complete([UserMessage.from_text("hello")], _settings())

    # THEN both calls are recorded (recording precedes dispatch)
    assert len(provider.calls) == 2
    assert provider.calls[-1].messages[-1].text == "hello"


async def test_stub_provider_from_predicates_returns_matching_response() -> None:
    """`from_predicates` returns the response paired with first match."""

    # GIVEN a stub with predicates keyed on the last user message text
    provider = StubProvider.from_predicates(
        [
            (lambda msgs: msgs[-1].text == "ping", "pong"),
            (lambda msgs: msgs[-1].text == "hello", _assistant_message("hi there")),
        ]
    )
    settings = _settings()

    # WHEN `complete` is called with messages matching each predicate
    first = await provider.complete([UserMessage.from_text("ping")], settings)
    second = await provider.complete([UserMessage.from_text("hello")], settings)

    # THEN each call returns the paired response
    assert first.message.text == "pong"
    assert second.message.text == "hi there"


async def test_stub_provider_from_predicates_evaluates_in_order() -> None:
    """`from_predicates` returns the response of the FIRST match."""

    # GIVEN a stub whose predicates both match the same message
    provider = StubProvider.from_predicates(
        [
            (lambda _msgs: True, "first"),
            (lambda _msgs: True, "second"),
        ]
    )

    # WHEN `complete` is called
    result = await provider.complete([UserMessage.from_text("anything")], _settings())

    # THEN the response of the first predicate is returned
    assert result.message.text == "first"


async def test_stub_provider_from_predicates_raises_when_no_match() -> None:
    """`from_predicates` raises `AssertionError` if no predicate matches."""

    # GIVEN a stub whose single predicate matches nothing
    provider = StubProvider.from_predicates(
        [(lambda msgs: msgs[-1].text == "ping", "pong")]
    )

    # WHEN `complete` is called with a non-matching message
    # THEN an `AssertionError` is raised
    with pytest.raises(AssertionError, match="no predicate matched"):
        await provider.complete([UserMessage.from_text("hello")], _settings())


async def test_stub_provider_from_predicates_records_call_on_failure() -> None:
    """A call is recorded in `.calls` even when no predicate matches."""

    # GIVEN a stub whose predicates never match
    provider = StubProvider.from_predicates([(lambda _msgs: False, "never")])

    # WHEN `complete` is called and raises
    with pytest.raises(AssertionError):
        await provider.complete([UserMessage.from_text("hello")], _settings())

    # THEN the call was still recorded (recording precedes dispatch)
    assert len(provider.calls) == 1
    assert provider.calls[-1].messages[-1].text == "hello"
