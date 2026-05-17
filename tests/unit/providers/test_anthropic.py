"""Tests for `avior.providers.anthropic`."""

from typing import cast
from unittest.mock import AsyncMock

import httpx
import pytest
from anthropic import (
    AnthropicError,
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    AsyncAnthropic,
    RateLimitError,
    omit,
)
from anthropic.types import Message as AnthropicMessage
from anthropic.types import TextBlock, Usage

from avior.core.exceptions import (
    ProviderConnectionError,
    ProviderError,
    ProviderHTTPError,
    ProviderResponseValidationError,
)
from avior.core.messages import Message, TextPart
from avior.core.provider import ModelSettings
from avior.providers.anthropic import AnthropicProvider


def _settings(
    *,
    model: str = "claude-test",
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> ModelSettings:
    """Construct `ModelSettings` with sensible defaults for tests."""

    return ModelSettings(model=model, max_tokens=max_tokens, temperature=temperature)


def _response(*texts: str) -> AnthropicMessage:
    """Build a minimal `anthropic.types.Message` response with text blocks."""

    return AnthropicMessage(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-test",
        content=[TextBlock(type="text", text=t) for t in texts],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=0, output_tokens=0),
    )


def _mock_client_returning(response: AnthropicMessage) -> AsyncMock:
    """Mock `AsyncAnthropic` whose `messages.create` returns `response`."""

    mock = AsyncMock()
    mock.messages.create = AsyncMock(return_value=response)
    return mock


def _mock_client_raising(error: Exception) -> AsyncMock:
    """Mock `AsyncAnthropic` whose `messages.create` raises `error`."""

    mock = AsyncMock()
    mock.messages.create = AsyncMock(side_effect=error)
    return mock


def _provider(client: AsyncMock) -> AnthropicProvider:
    """Wrap a mock client in an `AnthropicProvider` for testing."""

    return AnthropicProvider(client=cast(AsyncAnthropic, client))


def _http_response(status_code: int) -> httpx.Response:
    """Minimal `httpx.Response` for constructing Anthropic SDK exceptions."""

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx.Response(status_code=status_code, request=request)


# Constructor tests
# -----------------------------------------------------------------------------


async def test_provider_prefers_explicit_client_over_api_key() -> None:
    """`client` wins when both `client` and `api_key` are supplied."""

    # GIVEN a pre-built mock client preset to return a known response
    mock_client = _mock_client_returning(_response("Hi from supplied client"))

    # WHEN the provider is constructed with both `client` and `api_key`
    # and `complete` is awaited
    provider = AnthropicProvider(
        client=cast(AsyncAnthropic, mock_client),
        api_key="ignored",
    )
    result = await provider.complete([Message.user("hello")], _settings())

    # THEN the supplied client handles the call (proven by its preset response)
    assert result == Message.assistant("Hi from supplied client")


# Behavioural tests on `complete()`
# -----------------------------------------------------------------------------


async def test_complete_returns_assistant_message_parsed_from_response() -> None:
    """`complete` returns the assistant message decoded from the response."""

    # GIVEN a mock client returning a single-text-block assistant message
    mock_client = _mock_client_returning(_response("Hi!"))
    provider = _provider(mock_client)

    # WHEN `complete` is awaited
    result = await provider.complete([Message.user("hello")], _settings())

    # THEN the result is the assistant message containing the response text
    assert result == Message.assistant("Hi!")


async def test_complete_sends_leading_system_message_as_top_level() -> None:
    """`complete` extracts a leading system message and sends it top-level."""

    # GIVEN a mock client and messages with a leading system message and a user
    # message
    mock_client = _mock_client_returning(_response("Hi!"))
    provider = _provider(mock_client)
    messages = [Message.system("be helpful"), Message.user("hello")]

    # WHEN `complete` is invoked
    await provider.complete(messages, _settings())

    # THEN the Anthropic SDK call receives the system text as a top-level
    # block and the user message goes in `messages` as a list of content blocks
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == [{"type": "text", "text": "be helpful"}]
    assert len(call_kwargs["messages"]) == 1
    assert call_kwargs["messages"][0]["role"] == "user"
    assert call_kwargs["messages"][0]["content"] == [{"type": "text", "text": "hello"}]


async def test_complete_passes_system_messages_as_separate_blocks() -> None:
    """`complete` passes all `system` messages as separate top-level blocks."""

    # GIVEN a mock client and messages with `system` messages at several spots
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    messages = [
        Message.system("first"),
        Message.user("hi"),
        Message.system("later"),
    ]

    # WHEN `complete` is invoked
    await provider.complete(messages, _settings())

    # THEN the Anthropic SDK call receives both system texts as separate
    # top-level blocks and the messages array contains only the user message
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "later"},
    ]
    assert len(call_kwargs["messages"]) == 1
    assert call_kwargs["messages"][0]["role"] == "user"


async def test_complete_preserves_non_system_order_after_extraction() -> None:
    """`complete` preserves the relative order of non-`system` messages."""

    # GIVEN a mock client and messages interleaving `system` with user/assistant
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    messages = [
        Message.system("a"),
        Message.user("u1"),
        Message.assistant("a1"),
        Message.system("b"),
        Message.user("u2"),
    ]

    # WHEN `complete` is invoked
    await provider.complete(messages, _settings())

    # THEN the wire `messages` array contains only the non-`system` messages
    # in original order
    call_kwargs = mock_client.messages.create.call_args.kwargs
    wire = call_kwargs["messages"]
    assert [m["role"] for m in wire] == ["user", "assistant", "user"]
    assert wire[0]["content"] == [{"type": "text", "text": "u1"}]
    assert wire[1]["content"] == [{"type": "text", "text": "a1"}]
    assert wire[2]["content"] == [{"type": "text", "text": "u2"}]


async def test_complete_skips_empty_system_messages() -> None:
    """`complete` skips `system` messages with empty text."""

    # GIVEN a mock client and messages including an empty `system` message
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    messages = [Message.system(""), Message.user("hi")]

    # WHEN `complete` is invoked
    await provider.complete(messages, _settings())

    # THEN the `system` kwarg is the `omit` sentinel (empty system is skipped)
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["system"] is omit


async def test_complete_omits_system_when_no_leading_system_message() -> None:
    """`complete` passes `omit` when the input has no system message."""

    # GIVEN a mock client and messages without any system message
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    messages = [Message.user("hello")]

    # WHEN `complete` is invoked
    await provider.complete(messages, _settings())

    # THEN the `system` kwarg is the `omit` sentinel
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["system"] is omit


async def test_complete_forwards_explicit_max_tokens_and_temperature() -> None:
    """`complete` forwards explicit `max_tokens` and `temperature` unchanged."""

    # GIVEN a mock client and settings with explicit `max_tokens` and
    # `temperature`
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    settings = _settings(max_tokens=2048, temperature=0.2)

    # WHEN `complete` is invoked
    await provider.complete([Message.user("hi")], settings)

    # THEN the Anthropic SDK call receives the exact values
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 2048
    assert call_kwargs["temperature"] == 0.2


async def test_complete_defaults_max_tokens_to_4096_when_unset() -> None:
    """`complete` falls back to 4096 when `settings.max_tokens` is `None`."""

    # GIVEN a mock client and settings without an explicit `max_tokens`
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    settings = _settings(max_tokens=None)

    # WHEN `complete` is invoked
    await provider.complete([Message.user("hi")], settings)

    # THEN the Anthropic SDK call receives `max_tokens=4096`
    assert mock_client.messages.create.call_args.kwargs["max_tokens"] == 4096


async def test_complete_omits_temperature_when_unset() -> None:
    """`complete` passes `omit` for `temperature` when not set on settings."""

    # GIVEN a mock client and settings without an explicit `temperature`
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    settings = _settings(temperature=None)

    # WHEN `complete` is invoked
    await provider.complete([Message.user("hi")], settings)

    # THEN the `temperature` kwarg is the `omit` sentinel
    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs["temperature"] is omit


async def test_complete_maps_each_response_text_block_to_a_part() -> None:
    """`complete` maps each response `TextBlock` to its own `TextPart`."""

    # GIVEN a mock client returning a response with two text blocks
    mock_client = _mock_client_returning(_response("hello ", "world"))
    provider = _provider(mock_client)

    # WHEN `complete` is awaited
    result = await provider.complete([Message.user("hi")], _settings())

    # THEN the returned message has one `TextPart` per response block, in order
    assert result.parts == [TextPart(text="hello "), TextPart(text="world")]


async def test_complete_returns_empty_parts_when_response_content_is_empty() -> None:
    """`complete` returns `parts=[]` when the response has no content blocks."""

    # GIVEN a mock client returning a response with empty content (zero blocks)
    mock_client = _mock_client_returning(_response())
    provider = _provider(mock_client)

    # WHEN `complete` is awaited
    result = await provider.complete([Message.user("hi")], _settings())

    # THEN the result has an empty parts list (not a single empty `TextPart`)
    assert result.role == "assistant"
    assert result.parts == []


# Exception translation tests
# -----------------------------------------------------------------------------


async def test_complete_translates_api_status_error_to_http_error() -> None:
    """`APIStatusError` becomes `ProviderHTTPError`, preserving status."""

    # GIVEN a mock client raising Anthropic's `RateLimitError` (status 429, a
    # subclass of `APIStatusError`)
    anthropic_error = RateLimitError(
        "rate limit hit",
        response=_http_response(429),
        body=None,
    )
    provider = _provider(_mock_client_raising(anthropic_error))

    # WHEN `complete` is invoked
    # THEN `ProviderHTTPError` is raised with the HTTP status, and the original
    # exception is preserved as `__cause__`
    with pytest.raises(ProviderHTTPError) as exc_info:
        await provider.complete([Message.user("hi")], _settings())
    assert exc_info.value.status_code == 429
    assert exc_info.value.__cause__ is anthropic_error


async def test_complete_translates_response_validation_error() -> None:
    """`APIResponseValidationError` maps to the avior counterpart."""

    # GIVEN a mock client raising `APIResponseValidationError` (the Anthropic
    # SDK could not decode an otherwise-successful HTTP 200 response)
    anthropic_error = APIResponseValidationError(
        response=_http_response(200),
        body=None,
        message="schema mismatch",
    )
    provider = _provider(_mock_client_raising(anthropic_error))

    # WHEN `complete` is invoked
    # THEN `ProviderResponseValidationError` is raised, with the original
    # exception preserved as `__cause__`
    with pytest.raises(ProviderResponseValidationError) as exc_info:
        await provider.complete([Message.user("hi")], _settings())
    assert exc_info.value.__cause__ is anthropic_error


async def test_complete_translates_connection_error() -> None:
    """`APIConnectionError` becomes `ProviderConnectionError`."""

    # GIVEN a mock client raising `APIConnectionError` (network failed before an
    # HTTP response was received)
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    anthropic_error = APIConnectionError(request=request)
    provider = _provider(_mock_client_raising(anthropic_error))

    # WHEN `complete` is invoked
    # THEN `ProviderConnectionError` is raised with the original exception
    # preserved as `__cause__`
    with pytest.raises(ProviderConnectionError) as exc_info:
        await provider.complete([Message.user("hi")], _settings())
    assert exc_info.value.__cause__ is anthropic_error


async def test_complete_translates_timeout_as_connection_error() -> None:
    """`APITimeoutError` maps to `ProviderConnectionError` via subclass."""

    # GIVEN a mock client raising `APITimeoutError` (subclass of
    # `APIConnectionError`)
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    anthropic_error = APITimeoutError(request=request)
    provider = _provider(_mock_client_raising(anthropic_error))

    # WHEN `complete` is invoked
    # THEN `ProviderConnectionError` is raised (timeouts surface as
    # connection-level failures)
    with pytest.raises(ProviderConnectionError) as exc_info:
        await provider.complete([Message.user("hi")], _settings())
    assert exc_info.value.__cause__ is anthropic_error


async def test_complete_translates_other_anthropic_errors_to_provider_error() -> None:
    """A generic `AnthropicError` maps to the base `ProviderError`."""

    # GIVEN a mock client raising a generic `AnthropicError` (not in the
    # `APIError` family that the specific handlers catch)
    anthropic_error = AnthropicError("unexpected SDK failure")
    provider = _provider(_mock_client_raising(anthropic_error))

    # WHEN `complete` is invoked
    # THEN `ProviderError` (the exact base class, not a subclass) is raised
    # with the original exception preserved as `__cause__`
    with pytest.raises(ProviderError) as exc_info:
        await provider.complete([Message.user("hi")], _settings())
    assert type(exc_info.value) is ProviderError
    assert exc_info.value.__cause__ is anthropic_error
