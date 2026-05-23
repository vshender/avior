"""Tests for `avior.providers.openai_responses`."""

from typing import cast
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAIError,
    omit,
)
from openai.types.responses import (
    Response,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
)

from avior.core.exceptions import (
    ProviderConnectionError,
    ProviderError,
    ProviderHTTPError,
    ProviderResponseValidationError,
)
from avior.core.messages import Message, TextPart
from avior.core.provider import ModelSettings
from avior.providers.openai_responses import OpenAIResponsesProvider


def _settings(
    *,
    model: str = "gpt-test",
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> ModelSettings:
    """Construct `ModelSettings` with sensible defaults for tests."""

    return ModelSettings(model=model, max_tokens=max_tokens, temperature=temperature)


def _response(*texts: str) -> Response:
    """Build a minimal `openai.types.responses.Response` with text items.

    One `ResponseOutputMessage` is emitted containing one `ResponseOutputText`
    per supplied text.  Empty `texts` produces a response with an empty
    `output` list (no message item at all).
    """

    output: list[ResponseOutputItem] = []
    if texts:
        output.append(
            ResponseOutputMessage(
                id="msg_test",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    ResponseOutputText(type="output_text", text=t, annotations=[])
                    for t in texts
                ],
            )
        )

    return Response(
        id="resp_test",
        object="response",
        created_at=0.0,
        model="gpt-test",
        output=output,
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def _mock_client_returning(response: Response) -> AsyncMock:
    """Mock `AsyncOpenAI` whose `responses.create` returns `response`."""

    mock = AsyncMock()
    mock.responses.create = AsyncMock(return_value=response)
    return mock


def _mock_client_raising(error: Exception) -> AsyncMock:
    """Mock `AsyncOpenAI` whose `responses.create` raises `error`."""

    mock = AsyncMock()
    mock.responses.create = AsyncMock(side_effect=error)
    return mock


def _provider(client: AsyncMock) -> OpenAIResponsesProvider:
    """Wrap a mock client in an `OpenAIResponsesProvider` for testing."""

    return OpenAIResponsesProvider(client=cast(AsyncOpenAI, client))


def _http_response(status_code: int) -> httpx.Response:
    """Minimal `httpx.Response` for constructing OpenAI SDK exceptions."""

    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return httpx.Response(status_code=status_code, request=request)


def _http_request() -> httpx.Request:
    """Minimal `httpx.Request` for connection-error construction."""

    return httpx.Request("POST", "https://api.openai.com/v1/responses")


# Constructor tests
# -----------------------------------------------------------------------------


async def test_provider_prefers_explicit_client_over_api_key() -> None:
    """`client` wins when both `client` and `api_key` are supplied."""

    # GIVEN a pre-built mock client preset to return a known response
    mock_client = _mock_client_returning(_response("Hi from supplied client"))

    # WHEN the provider is constructed with both `client` and `api_key` and
    # `complete` is awaited
    provider = OpenAIResponsesProvider(
        client=cast(AsyncOpenAI, mock_client),
        api_key="ignored",
    )
    result = await provider.complete([Message.user("hello")], _settings())

    # THEN the supplied client handles the call (proven by its preset response)
    assert result == Message.assistant("Hi from supplied client")


# Behavioural tests on `complete()`
# -----------------------------------------------------------------------------


async def test_complete_returns_assistant_message_parsed_from_response() -> None:
    """`complete` returns the assistant message decoded from the response."""

    # GIVEN a mock client returning a single-text-item assistant message
    mock_client = _mock_client_returning(_response("Hi!"))
    provider = _provider(mock_client)

    # WHEN `complete` is awaited
    result = await provider.complete([Message.user("hello")], _settings())

    # THEN the result is the assistant message containing the response text
    assert result == Message.assistant("Hi!")


async def test_complete_lifts_leading_system_message_to_instructions() -> None:
    """`complete` extracts a leading system message and sends it top-level."""

    # GIVEN a mock client and messages with a leading system message and a user
    # message
    mock_client = _mock_client_returning(_response("Hi!"))
    provider = _provider(mock_client)
    messages = [Message.system("be helpful"), Message.user("hello")]

    # WHEN `complete` is invoked
    await provider.complete(messages, _settings())

    # THEN the OpenAI SDK call receives the system text as the top-level
    # `instructions` string and the user message goes in `input`
    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert call_kwargs["instructions"] == "be helpful"
    assert len(call_kwargs["input"]) == 1
    assert call_kwargs["input"][0]["role"] == "user"
    assert call_kwargs["input"][0]["content"] == "hello"


async def test_complete_joins_multiple_system_messages_with_blank_lines() -> None:
    """`complete` joins all `system` messages into one `instructions` string."""

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

    # THEN the OpenAI SDK call receives both system texts joined into
    # `instructions` (blank-line-separated) and the `input` contains only the
    # user message
    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert call_kwargs["instructions"] == "first\n\nlater"
    assert len(call_kwargs["input"]) == 1
    assert call_kwargs["input"][0]["role"] == "user"
    assert call_kwargs["input"][0]["content"] == "hi"


async def test_complete_preserves_non_system_order_after_extraction() -> None:
    """`complete` preserves the relative order of non-`system` messages."""

    # GIVEN a mock client and messages interleaving `system` with user/assistant
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    messages = [
        Message.system("s1"),
        Message.user("u1"),
        Message.assistant("a1"),
        Message.system("s2"),
        Message.user("u2"),
    ]

    # WHEN `complete` is invoked
    await provider.complete(messages, _settings())

    # THEN the wire `input` array contains only the non-`system` messages in
    # original order
    call_kwargs = mock_client.responses.create.call_args.kwargs
    wire = call_kwargs["input"]
    assert [m["role"] for m in wire] == ["user", "assistant", "user"]
    assert wire[0]["content"] == "u1"
    assert wire[1]["content"] == "a1"
    assert wire[2]["content"] == "u2"


async def test_complete_skips_empty_system_messages() -> None:
    """`complete` skips `system` messages with empty text."""

    # GIVEN a mock client and messages including an empty `system` message
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    messages = [Message.system(""), Message.user("hi")]

    # WHEN `complete` is invoked
    await provider.complete(messages, _settings())

    # THEN the `instructions` kwarg is the `omit` sentinel (empty system is
    # skipped, no blank `instructions` string is sent)
    kwargs = mock_client.responses.create.call_args.kwargs
    assert kwargs["instructions"] is omit


async def test_complete_skips_empty_system_messages_when_joining() -> None:
    """`complete` skips empty `system` messages when joining `instructions`."""

    # GIVEN a mock client and messages mixing an empty `system` message with a
    # non-empty one
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    messages = [
        Message(role="system", parts=[]),
        Message.system("real instruction"),
        Message.user("hi"),
    ]

    # WHEN `complete` is invoked
    await provider.complete(messages, _settings())

    # THEN only the non-empty system text appears in `instructions` (no stray
    # `\n\n` separator from the empty message)
    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert call_kwargs["instructions"] == "real instruction"


async def test_complete_omits_instructions_when_no_system_message() -> None:
    """`complete` passes `omit` for `instructions` when no system is present."""

    # GIVEN a mock client and messages without any system message
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)

    # WHEN `complete` is invoked
    await provider.complete([Message.user("hi")], _settings())

    # THEN the `instructions` kwarg is the `omit` sentinel
    kwargs = mock_client.responses.create.call_args.kwargs
    assert kwargs["instructions"] is omit


async def test_complete_forwards_explicit_max_tokens_and_temperature() -> None:
    """`complete` forwards explicit `max_tokens` and `temperature` unchanged."""

    # GIVEN a mock client and settings with explicit `max_tokens` and
    # `temperature`
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    settings = _settings(max_tokens=2048, temperature=0.2)

    # WHEN `complete` is invoked
    await provider.complete([Message.user("hi")], settings)

    # THEN the OpenAI SDK call receives the exact values
    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert call_kwargs["max_output_tokens"] == 2048
    assert call_kwargs["temperature"] == 0.2


async def test_complete_omits_max_output_tokens_when_unset() -> None:
    """`complete` passes `omit` for `max_output_tokens` when not set.

    Unlike the Anthropic adapter (which defaults `max_tokens` to 4096), the
    Responses API does not require `max_output_tokens`, so the OpenAI adapter
    forwards no default - the SDK applies its own.
    """

    # GIVEN a mock client and settings without an explicit `max_tokens`
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    settings = _settings(max_tokens=None)

    # WHEN `complete` is invoked
    await provider.complete([Message.user("hi")], settings)

    # THEN the `max_output_tokens` kwarg is the `omit` sentinel
    kwargs = mock_client.responses.create.call_args.kwargs
    assert kwargs["max_output_tokens"] is omit


async def test_complete_omits_temperature_when_unset() -> None:
    """`complete` passes `omit` for `temperature` when not set on settings."""

    # GIVEN a mock client and settings without an explicit `temperature`
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    settings = _settings(temperature=None)

    # WHEN `complete` is invoked
    await provider.complete([Message.user("hi")], settings)

    # THEN the `temperature` kwarg is the `omit` sentinel
    kwargs = mock_client.responses.create.call_args.kwargs
    assert kwargs["temperature"] is omit


async def test_complete_passes_store_false() -> None:
    """`complete` always passes `store=False` (stateless wire)."""

    # GIVEN a mock client and any messages
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)

    # WHEN `complete` is invoked
    await provider.complete([Message.user("hi")], _settings())

    # THEN the `store` kwarg is `False`, so no server-side history is created
    kwargs = mock_client.responses.create.call_args.kwargs
    assert kwargs["store"] is False


async def test_complete_maps_each_response_text_item_to_a_part() -> None:
    """`complete` maps each response text item to its own `TextPart`."""

    # GIVEN a mock client returning a response with two text items
    mock_client = _mock_client_returning(_response("hello ", "world"))
    provider = _provider(mock_client)

    # WHEN `complete` is awaited
    result = await provider.complete([Message.user("hi")], _settings())

    # THEN the returned message has one `TextPart` per response item, in order
    assert result.parts == [TextPart(text="hello "), TextPart(text="world")]


async def test_complete_returns_empty_parts_when_response_output_is_empty() -> None:
    """`complete` returns `parts=[]` when the response has no output items."""

    # GIVEN a mock client returning a response with empty output (zero items)
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

    # GIVEN a mock client raising `APIStatusError` (status 429)
    openai_error = APIStatusError(
        "rate limit hit",
        response=_http_response(429),
        body=None,
    )
    provider = _provider(_mock_client_raising(openai_error))

    # WHEN `complete` is invoked
    # THEN `ProviderHTTPError` is raised with the HTTP status, and the original
    # exception is preserved as `__cause__`
    with pytest.raises(ProviderHTTPError) as exc_info:
        await provider.complete([Message.user("hi")], _settings())
    assert exc_info.value.status_code == 429
    assert exc_info.value.__cause__ is openai_error


async def test_complete_translates_response_validation_error() -> None:
    """`APIResponseValidationError` maps to the avior counterpart."""

    # GIVEN a mock client raising `APIResponseValidationError` (the OpenAI
    # SDK could not decode an otherwise-successful HTTP 200 response)
    openai_error = APIResponseValidationError(
        response=_http_response(200),
        body=None,
        message="schema mismatch",
    )
    provider = _provider(_mock_client_raising(openai_error))

    # WHEN `complete` is invoked
    # THEN `ProviderResponseValidationError` is raised, with the original
    # exception preserved as `__cause__`
    with pytest.raises(ProviderResponseValidationError) as exc_info:
        await provider.complete([Message.user("hi")], _settings())
    assert exc_info.value.__cause__ is openai_error


async def test_complete_translates_connection_error() -> None:
    """`APIConnectionError` becomes `ProviderConnectionError`."""

    # GIVEN a mock client raising `APIConnectionError` (network failed before
    # an HTTP response was received)
    openai_error = APIConnectionError(request=_http_request())
    provider = _provider(_mock_client_raising(openai_error))

    # WHEN `complete` is invoked
    # THEN `ProviderConnectionError` is raised with the original exception
    # preserved as `__cause__`
    with pytest.raises(ProviderConnectionError) as exc_info:
        await provider.complete([Message.user("hi")], _settings())
    assert exc_info.value.__cause__ is openai_error


async def test_complete_translates_timeout_as_connection_error() -> None:
    """`APITimeoutError` maps to `ProviderConnectionError` via subclass."""

    # GIVEN a mock client raising `APITimeoutError` (subclass of
    # `APIConnectionError`)
    openai_error = APITimeoutError(request=_http_request())
    provider = _provider(_mock_client_raising(openai_error))

    # WHEN `complete` is invoked
    # THEN `ProviderConnectionError` is raised (timeouts surface as
    # connection-level failures)
    with pytest.raises(ProviderConnectionError) as exc_info:
        await provider.complete([Message.user("hi")], _settings())
    assert exc_info.value.__cause__ is openai_error


async def test_complete_translates_other_openai_errors_to_provider_error() -> None:
    """A generic `OpenAIError` maps to the base `ProviderError`."""

    # GIVEN a mock client raising a generic `OpenAIError` (not in the
    # `APIError` family that the specific handlers catch)
    openai_error = OpenAIError("unexpected SDK failure")
    provider = _provider(_mock_client_raising(openai_error))

    # WHEN `complete` is invoked
    # THEN `ProviderError` (the exact base class, not a subclass) is raised
    # with the original exception preserved as `__cause__`
    with pytest.raises(ProviderError) as exc_info:
        await provider.complete([Message.user("hi")], _settings())
    assert type(exc_info.value) is ProviderError
    assert exc_info.value.__cause__ is openai_error


# Lifecycle tests
# -----------------------------------------------------------------------------


def _provider_owning(
    monkeypatch: pytest.MonkeyPatch,
    client: AsyncMock,
) -> OpenAIResponsesProvider:
    """Construct a provider that "owns" a mock client.

    Patches the `AsyncOpenAI` symbol in the provider module so that the
    no-`client=` path yields the supplied mock - giving the test a handle on
    the would-be-self-constructed client without making real network calls.
    """

    def _factory(**_: object) -> AsyncMock:
        return client

    monkeypatch.setattr("avior.providers.openai_responses.AsyncOpenAI", _factory)
    return OpenAIResponsesProvider(api_key="fake")


async def test_aclose_closes_self_constructed_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`aclose` closes the SDK client the provider constructed itself."""

    # GIVEN a provider that constructed its own (mock) client
    mock_client = AsyncMock()
    provider = _provider_owning(monkeypatch, mock_client)

    # WHEN `aclose` is awaited
    await provider.aclose()

    # THEN the underlying client is closed
    mock_client.close.assert_awaited_once()


async def test_aclose_leaves_user_supplied_client_open() -> None:
    """`aclose` does not close clients supplied by the caller."""

    # GIVEN a provider with a caller-supplied client
    mock_client = AsyncMock()
    provider = _provider(mock_client)

    # WHEN `aclose` is awaited
    await provider.aclose()

    # THEN the caller-owned client is left alone
    mock_client.close.assert_not_called()


async def test_async_cm_exit_calls_aclose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exiting `async with` runs `aclose` on the owning provider."""

    # GIVEN a provider that owns its (mock) client
    mock_client = AsyncMock()
    provider = _provider_owning(monkeypatch, mock_client)

    # WHEN used as an async context manager
    async with provider:
        pass

    # THEN `aclose` ran (visible via the underlying `close` call)
    mock_client.close.assert_awaited_once()


async def test_async_cm_exit_leaves_user_supplied_client_open() -> None:
    """`async with` on a user-supplied-client provider does not close it."""

    # GIVEN a provider with a caller-supplied client
    mock_client = AsyncMock()
    provider = _provider(mock_client)

    # WHEN used as an async context manager
    async with provider:
        pass

    # THEN the caller-owned client is left alone
    mock_client.close.assert_not_called()


async def test_nested_async_cm_closes_only_on_outermost_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reference-counted `async with`: inner exit does not close the client."""

    # GIVEN a provider that owns its (mock) client
    mock_client = AsyncMock()
    provider = _provider_owning(monkeypatch, mock_client)

    # WHEN entered twice (nested `async with`)
    async with provider:
        async with provider:
            # THEN inside the inner block the client is still open
            mock_client.close.assert_not_called()
        # AND after the inner exit the client is still open (refcount > 0)
        mock_client.close.assert_not_called()

    # AND after the outermost exit `aclose` has run exactly once
    mock_client.close.assert_awaited_once()
