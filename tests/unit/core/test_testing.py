"""Tests for `avior.core.testing`."""

from collections.abc import Callable

import pytest

from avior.core.messages import Message
from avior.core.provider import ModelSettings, Provider
from avior.core.testing import StubCall, StubProvider


def _settings(model: str = "test-model") -> ModelSettings:
    """Construct a minimal `ModelSettings` for use in tests."""

    return ModelSettings(model=model)


async def test_stub_provider_conforms_to_provider_protocol() -> None:
    """`StubProvider` satisfies the `@runtime_checkable` `Provider` Protocol."""

    # GIVEN a stub built from the canonical callable form
    provider = StubProvider(lambda _msgs, _settings: Message.assistant("ok"))

    # WHEN it is checked against the `Provider` runtime protocol
    is_provider = isinstance(provider, Provider)  # pyright: ignore[reportUnnecessaryIsInstance]

    # THEN it conforms (the `complete` method is structurally sufficient)
    assert is_provider


async def test_stub_provider_canonical_callable_returns_message_directly() -> None:
    """A `Message` returned by the callable is passed through unchanged."""

    # GIVEN a stub whose callable returns a pre-built `Message`
    response = Message.assistant("hi there")
    provider = StubProvider(lambda _msgs, _settings: response)

    # WHEN `complete` is called
    result = await provider.complete([Message.user("hello")], _settings())

    # THEN the configured `Message` is returned unchanged
    assert result is response


async def test_stub_provider_canonical_callable_wraps_string_responses() -> None:
    """A `str` returned from the callable is wrapped as `Message.assistant`."""

    # GIVEN a stub whose callable returns a plain string
    provider = StubProvider(lambda _msgs, _settings: "hi")

    # WHEN `complete` is called
    result = await provider.complete([Message.user("hello")], _settings())

    # THEN the result is an assistant-role message with that text
    assert result == Message.assistant("hi")


async def test_stub_provider_canonical_callable_awaits_coroutine_results() -> None:
    """An awaitable returned from the callable is awaited."""

    # GIVEN a stub whose callable is an async function
    async def async_func(_msgs: list[Message], _settings: ModelSettings) -> str:
        return "async hi"

    provider = StubProvider(async_func)

    # WHEN `complete` is called
    result = await provider.complete([Message.user("hello")], _settings())

    # THEN the awaited string is wrapped as `Message.assistant`
    assert result == Message.assistant("async hi")


async def test_stub_provider_callable_receives_messages_and_settings() -> None:
    """The callable receives the same `messages` and `settings` by identity."""

    # GIVEN a stub callable that records its arguments
    received: list[StubCall] = []

    def func(messages: list[Message], settings: ModelSettings) -> str:
        received.append(StubCall(messages=messages, settings=settings))
        return "ok"

    provider = StubProvider(func)
    messages = [Message.system("you are helpful"), Message.user("hi")]
    settings = _settings("claude-3-5-sonnet")

    # WHEN `complete` is called
    await provider.complete(messages, settings)

    # THEN the callable observed the same arguments by identity
    assert len(received) == 1
    assert received[0].messages is messages
    assert received[0].settings is settings


@pytest.mark.parametrize(
    ("role", "message_factory"),
    [
        pytest.param("user", Message.user, id="user"),
        pytest.param("system", Message.system, id="system"),
    ],
)
async def test_stub_provider_rejects_non_assistant_role_messages(
    role: str,
    message_factory: Callable[[str], Message],
) -> None:
    """A `Message` with non-assistant role raises `AssertionError`."""

    # GIVEN a stub whose callable returns a non-assistant-role message
    provider = StubProvider(lambda _msgs, _settings: message_factory("oops"))

    # WHEN `complete` is called
    # THEN it raises `AssertionError` whose message names both the
    # required role and the actual wrong role
    with pytest.raises(
        AssertionError,
        match=rf"role='assistant'.*role={role!r}",
    ):
        await provider.complete([Message.user("hi")], _settings())


async def test_stub_provider_records_each_call_in_order() -> None:
    """Each `complete` invocation is appended to `.calls` in order."""

    # GIVEN a stub that always returns the same response
    provider = StubProvider(lambda _msgs, _settings: "ok")

    # WHEN `complete` is called three times with different messages
    settings = _settings()
    await provider.complete([Message.user("first")], settings)
    await provider.complete([Message.user("second")], settings)
    await provider.complete([Message.user("third")], settings)

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
    first = await provider.complete([Message.user("a")], settings)
    second = await provider.complete([Message.user("b")], settings)

    # THEN each call returns the next response in order
    assert first == Message.assistant("hello")
    assert second == Message.assistant("world")


async def test_stub_provider_from_responses_accepts_mixed_str_and_message() -> None:
    """`from_responses` accepts both `str` and `Message` entries."""

    # GIVEN a stub built from a heterogeneous response list
    canned_message = Message.assistant("from message")
    provider = StubProvider.from_responses(["from str", canned_message])

    # WHEN `complete` is called twice
    settings = _settings()
    first = await provider.complete([Message.user("a")], settings)
    second = await provider.complete([Message.user("b")], settings)

    # THEN strings are wrapped as `Message.assistant` and `Message` entries
    # are returned unchanged
    assert first == Message.assistant("from str")
    assert second is canned_message


async def test_stub_provider_from_responses_raises_when_exhausted() -> None:
    """`from_responses` raises `AssertionError` once exhausted."""

    # GIVEN a stub built with a single response, already consumed by one call
    provider = StubProvider.from_responses(["only one"])
    settings = _settings()
    await provider.complete([Message.user("a")], settings)

    # WHEN `complete` is called a second time
    # THEN it raises `AssertionError` with a descriptive message
    with pytest.raises(AssertionError, match="exhausted after 1 call"):
        await provider.complete([Message.user("b")], settings)


async def test_stub_provider_from_responses_records_call_on_failure() -> None:
    """A call is recorded in `.calls` even when responses are exhausted."""

    # GIVEN an already-exhausted stub
    provider = StubProvider.from_responses(["only one"])
    await provider.complete([Message.user("a")], _settings())

    # WHEN `complete` is called once more and raises
    with pytest.raises(AssertionError):
        await provider.complete([Message.user("hello")], _settings())

    # THEN both calls are recorded (recording precedes dispatch)
    assert len(provider.calls) == 2
    assert provider.calls[-1].messages[-1].text == "hello"


async def test_stub_provider_from_predicates_returns_matching_response() -> None:
    """`from_predicates` returns the response paired with first match."""

    # GIVEN a stub with predicates keyed on the last user message text
    provider = StubProvider.from_predicates(
        [
            (lambda msgs: msgs[-1].text == "ping", "pong"),
            (lambda msgs: msgs[-1].text == "hello", Message.assistant("hi there")),
        ]
    )
    settings = _settings()

    # WHEN `complete` is called with messages matching each predicate
    first = await provider.complete([Message.user("ping")], settings)
    second = await provider.complete([Message.user("hello")], settings)

    # THEN each call returns the paired response
    assert first == Message.assistant("pong")
    assert second == Message.assistant("hi there")


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
    result = await provider.complete([Message.user("anything")], _settings())

    # THEN the response of the first predicate is returned
    assert result == Message.assistant("first")


async def test_stub_provider_from_predicates_raises_when_no_match() -> None:
    """`from_predicates` raises `AssertionError` if no predicate matches."""

    # GIVEN a stub whose single predicate matches nothing
    provider = StubProvider.from_predicates(
        [(lambda msgs: msgs[-1].text == "ping", "pong")]
    )

    # WHEN `complete` is called with a non-matching message
    # THEN an `AssertionError` is raised
    with pytest.raises(AssertionError, match="no predicate matched"):
        await provider.complete([Message.user("hello")], _settings())


async def test_stub_provider_from_predicates_records_call_on_failure() -> None:
    """A call is recorded in `.calls` even when no predicate matches."""

    # GIVEN a stub whose predicates never match
    provider = StubProvider.from_predicates([(lambda _msgs: False, "never")])

    # WHEN `complete` is called and raises
    with pytest.raises(AssertionError):
        await provider.complete([Message.user("hello")], _settings())

    # THEN the call was still recorded (recording precedes dispatch)
    assert len(provider.calls) == 1
    assert provider.calls[-1].messages[-1].text == "hello"
