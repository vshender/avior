"""Tests for `avior.providers.anthropic`."""

from typing import Literal, cast
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
from anthropic.types import TextBlock, ToolUseBlock, Usage
from pydantic import BaseModel

from avior.core.context import RunContext
from avior.core.exceptions import (
    ProviderConnectionError,
    ProviderError,
    ProviderHTTPError,
    ProviderResponseValidationError,
)
from avior.core.messages import (
    AssistantMessage,
    Message,
    TextPart,
    ToolCallPart,
    ToolMessage,
    ToolResultError,
    ToolResultOk,
    ToolResultPart,
    UserMessage,
)
from avior.core.provider import ModelSettings
from avior.core.tools import Tool
from avior.providers.anthropic import AnthropicProvider


def _settings(
    *,
    model: str = "claude-test",
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> ModelSettings:
    """Construct `ModelSettings` with sensible defaults for tests."""

    return ModelSettings(model=model, max_tokens=max_tokens, temperature=temperature)


def _response(*texts: str, usage: Usage | None = None) -> AnthropicMessage:
    """Build a minimal `anthropic.types.Message` response with text blocks.

    Pass `usage` to attach token usage (default: a zeroed `Usage`).
    """

    return AnthropicMessage(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-test",
        content=[TextBlock(type="text", text=t) for t in texts],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=usage if usage is not None else Usage(input_tokens=0, output_tokens=0),
    )


def _response_with_stop_reason(
    stop_reason: Literal["end_turn", "max_tokens", "refusal"],
) -> AnthropicMessage:
    """Build a minimal assistant response with the given `stop_reason`."""

    return AnthropicMessage(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-test",
        content=[TextBlock(type="text", text="...")],
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(input_tokens=0, output_tokens=0),
    )


class _CityArgs(BaseModel):
    city: str


class _Weather(Tool[_CityArgs, str]):
    """A trivial tool used to exercise tool-calling wire translation."""

    name = "get_weather"
    description = "Look up the weather for a city."
    args_model = _CityArgs

    async def execute(self, ctx: RunContext[object], args: _CityArgs) -> str:
        return "sunny"


def _tool_use_response(
    call_id: str,
    tool_name: str,
    args: dict[str, object],
) -> AnthropicMessage:
    """Build an assistant response carrying a single `tool_use` block."""

    return AnthropicMessage(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-test",
        content=[ToolUseBlock(type="tool_use", id=call_id, name=tool_name, input=args)],
        stop_reason="tool_use",
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

    return httpx.Response(status_code=status_code, request=_http_request())


def _http_request() -> httpx.Request:
    """Minimal `httpx.Request` for connection-error construction."""

    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


# Constructor tests
# -----------------------------------------------------------------------------


async def test_provider_prefers_explicit_client_over_api_key() -> None:
    """`client` wins when both `client` and `api_key` are supplied."""

    # GIVEN a pre-built mock client preset to return a known response
    mock_client = _mock_client_returning(_response("Hi from supplied client"))

    # WHEN the provider is constructed with both `client` and `api_key` and
    # `complete` is awaited
    provider = AnthropicProvider(
        client=cast(AsyncAnthropic, mock_client),
        api_key="ignored",
    )
    result = await provider.complete([UserMessage.from_text("hello")], _settings())

    # THEN the supplied client handles the call (proven by its preset response)
    assert result.message.text == "Hi from supplied client"


# Behavioural tests on `complete()`
# -----------------------------------------------------------------------------


async def test_complete_returns_assistant_message_parsed_from_response() -> None:
    """`complete` returns the assistant message decoded from the response."""

    # GIVEN a mock client returning a single-text-block assistant message
    mock_client = _mock_client_returning(_response("Hi!"))
    provider = _provider(mock_client)

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hello")], _settings())

    # THEN the result is the assistant message containing the response text
    assert result.message.text == "Hi!"


async def test_complete_sends_system_prompt_as_top_level_block() -> None:
    """`complete` sends the `system_prompt` as a top-level text block."""

    # GIVEN a mock client and a system prompt alongside a user message
    mock_client = _mock_client_returning(_response("Hi!"))
    provider = _provider(mock_client)

    # WHEN `complete` is invoked with a system prompt
    await provider.complete(
        [UserMessage.from_text("hello")],
        _settings(),
        system_prompt="be helpful",
    )

    # THEN the Anthropic SDK call receives the system prompt as a top-level
    # block and the user message goes in `messages` as a list of content blocks
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == [{"type": "text", "text": "be helpful"}]
    assert len(call_kwargs["messages"]) == 1
    assert call_kwargs["messages"][0]["role"] == "user"
    assert call_kwargs["messages"][0]["content"] == [{"type": "text", "text": "hello"}]


async def test_complete_omits_system_prompt_when_none() -> None:
    """`complete` passes `omit` when `system_prompt` is `None`."""

    # GIVEN a mock client
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)

    # WHEN `complete` is invoked with no system prompt
    await provider.complete([UserMessage.from_text("hello")], _settings())

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
    await provider.complete([UserMessage.from_text("hi")], settings)

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
    await provider.complete([UserMessage.from_text("hi")], settings)

    # THEN the Anthropic SDK call receives `max_tokens=4096`
    assert mock_client.messages.create.call_args.kwargs["max_tokens"] == 4096


async def test_complete_omits_temperature_when_unset() -> None:
    """`complete` passes `omit` for `temperature` when not set on settings."""

    # GIVEN a mock client and settings without an explicit `temperature`
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    settings = _settings(temperature=None)

    # WHEN `complete` is invoked
    await provider.complete([UserMessage.from_text("hi")], settings)

    # THEN the `temperature` kwarg is the `omit` sentinel
    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs["temperature"] is omit


async def test_complete_maps_each_response_text_block_to_a_part() -> None:
    """`complete` maps each response `TextBlock` to its own `TextPart`."""

    # GIVEN a mock client returning a response with two text blocks
    mock_client = _mock_client_returning(_response("hello ", "world"))
    provider = _provider(mock_client)

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the returned message has one `TextPart` per response block, in order
    assert result.message.parts == [TextPart(text="hello "), TextPart(text="world")]


async def test_complete_returns_empty_parts_when_response_content_is_empty() -> None:
    """`complete` returns `parts=[]` when the response has no content blocks."""

    # GIVEN a mock client returning a response with empty content (zero blocks)
    mock_client = _mock_client_returning(_response())
    provider = _provider(mock_client)

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the result has an empty parts list (not a single empty `TextPart`)
    assert result.message.parts == []


# Call-metadata mapping tests
# -----------------------------------------------------------------------------


async def test_complete_maps_usage_ids_and_model_onto_provider_response() -> None:
    """`complete` maps Anthropic usage, response id, and model to wrapper."""

    # GIVEN a mock client returning a response with the call metadata
    response = _response(
        "hi",
        usage=Usage(
            input_tokens=11,
            output_tokens=7,
            cache_read_input_tokens=5,
            cache_creation_input_tokens=2,
        ),
    )
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN usage is normalized: Anthropic's cache-excluding input (11) is
    # widened to include the cache sub-slices (11 + 5 + 2 = 18), which remain
    # available individually, and total is derived (18 + 7)
    assert result.usage is not None
    assert result.usage.input_tokens == 18
    assert result.usage.output_tokens == 7
    assert result.usage.cache_read_tokens == 5
    assert result.usage.cache_write_tokens == 2
    assert result.usage.total_tokens == 25

    # AND reasoning stays None: Anthropic does not itemize it out of output, so
    # it is unknown, not 0
    assert result.usage.reasoning_tokens is None

    # AND the provider-native usage is preserved beside the normalized counts
    assert result.raw_usage is not None
    assert result.raw_usage["input_tokens"] == 11

    # AND the response id, served model, and provider name are populated
    assert result.response_id == "msg_test"
    assert result.model == "claude-test"
    assert result.provider_name == "anthropic"


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
        await provider.complete([UserMessage.from_text("hi")], _settings())
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
        await provider.complete([UserMessage.from_text("hi")], _settings())
    assert exc_info.value.__cause__ is anthropic_error


async def test_complete_translates_connection_error() -> None:
    """`APIConnectionError` becomes `ProviderConnectionError`."""

    # GIVEN a mock client raising `APIConnectionError` (network failed before an
    # HTTP response was received)
    anthropic_error = APIConnectionError(request=_http_request())
    provider = _provider(_mock_client_raising(anthropic_error))

    # WHEN `complete` is invoked
    # THEN `ProviderConnectionError` is raised with the original exception
    # preserved as `__cause__`
    with pytest.raises(ProviderConnectionError) as exc_info:
        await provider.complete([UserMessage.from_text("hi")], _settings())
    assert exc_info.value.__cause__ is anthropic_error


async def test_complete_translates_timeout_as_connection_error() -> None:
    """`APITimeoutError` maps to `ProviderConnectionError` via subclass."""

    # GIVEN a mock client raising `APITimeoutError` (subclass of
    # `APIConnectionError`)
    anthropic_error = APITimeoutError(request=_http_request())
    provider = _provider(_mock_client_raising(anthropic_error))

    # WHEN `complete` is invoked
    # THEN `ProviderConnectionError` is raised (timeouts surface as
    # connection-level failures)
    with pytest.raises(ProviderConnectionError) as exc_info:
        await provider.complete([UserMessage.from_text("hi")], _settings())
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
        await provider.complete([UserMessage.from_text("hi")], _settings())
    assert type(exc_info.value) is ProviderError
    assert exc_info.value.__cause__ is anthropic_error


# Stop-reason mapping tests
# -----------------------------------------------------------------------------


async def test_complete_sets_stop_reason_stop_on_end_turn() -> None:
    """Anthropic `stop_reason="end_turn"` maps to canonical `"stop"`."""

    # GIVEN a mock client returning a normal end-of-turn response
    provider = _provider(_mock_client_returning(_response_with_stop_reason("end_turn")))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the canonical `stop_reason` is `"stop"`
    assert result.message.stop_reason == "stop"


async def test_complete_maps_max_tokens_to_max_tokens_stop_reason() -> None:
    """Anthropic `stop_reason="max_tokens"` maps to canonical `"max_tokens"`."""

    # GIVEN a mock client returning a max-tokens-truncated response
    provider = _provider(
        _mock_client_returning(_response_with_stop_reason("max_tokens"))
    )

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the canonical `stop_reason` is `"max_tokens"`
    assert result.message.stop_reason == "max_tokens"


async def test_complete_maps_refusal_to_refusal_stop_reason() -> None:
    """Anthropic `refusal` stop_reason maps to canonical `"refusal"`."""

    # GIVEN a mock client returning a refusal response
    provider = _provider(_mock_client_returning(_response_with_stop_reason("refusal")))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the canonical `stop_reason` is `"refusal"`
    assert result.message.stop_reason == "refusal"


# Tool-calling tests
# -----------------------------------------------------------------------------


async def test_complete_parses_tool_use_block_into_tool_call_part() -> None:
    """A response `tool_use` block decodes into a `ToolCallPart`."""

    # GIVEN a provider whose mock client returns a tool-use response
    response = _tool_use_response("call_1", "get_weather", {"city": "Paris"})
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("weather?")], _settings())

    # THEN the block decodes into a `ToolCallPart` with its id, name, and args
    assert result.message.parts == [
        ToolCallPart(call_id="call_1", tool_name="get_weather", args={"city": "Paris"})
    ]


async def test_complete_maps_tool_use_to_tool_use_stop_reason() -> None:
    """Anthropic `stop_reason="tool_use"` maps to canonical `"tool_use"`."""

    # GIVEN a provider whose mock client returns a tool-use response
    response = _tool_use_response("call_1", "get_weather", {"city": "Paris"})
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("weather?")], _settings())

    # THEN the canonical `stop_reason` is `"tool_use"`
    assert result.message.stop_reason == "tool_use"


async def test_complete_sends_tools_with_name_description_and_input_schema() -> None:
    """Each offered tool is sent with its name, description, and JSON schema."""

    # GIVEN a provider and an offered tool
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)

    # WHEN `complete` is invoked with that tool
    await provider.complete(
        [UserMessage.from_text("hi")],
        _settings(),
        tools=[_Weather()],
    )

    # THEN the Anthropic SDK call carries the tool's name, description, and args
    # schema
    tools_param = mock_client.messages.create.call_args.kwargs["tools"]
    assert tools_param == [
        {
            "name": "get_weather",
            "description": "Look up the weather for a city.",
            "input_schema": _CityArgs.model_json_schema(),
        }
    ]


async def test_complete_omits_tools_when_none_offered() -> None:
    """No offered tools means the `tools` kwarg is the `omit` sentinel."""

    # GIVEN a provider and no tools
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)

    # WHEN `complete` is invoked without tools
    await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the `tools` kwarg is omitted rather than sent as an empty list
    assert mock_client.messages.create.call_args.kwargs["tools"] is omit


async def test_complete_sends_assistant_tool_call_as_tool_use_block() -> None:
    """An assistant `ToolCallPart` in the input becomes a `tool_use` block."""

    # GIVEN a continuation transcript: the assistant requested a tool call and
    # its result was supplied (a re-entry into `complete`)
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    history: list[Message] = [
        UserMessage.from_text("weather?"),
        AssistantMessage(
            parts=[
                ToolCallPart(
                    call_id="call_1", tool_name="get_weather", args={"city": "Paris"}
                )
            ],
            stop_reason="tool_use",
        ),
        ToolMessage(
            parts=[
                ToolResultPart(call_id="call_1", result=ToolResultOk(content="sunny"))
            ]
        ),
    ]

    # WHEN `complete` is invoked
    await provider.complete(history, _settings())

    # THEN the assistant turn is sent with a matching `tool_use` block
    wire_messages = mock_client.messages.create.call_args.kwargs["messages"]
    assistant_wire = next(m for m in wire_messages if m["role"] == "assistant")
    assert assistant_wire == {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "get_weather",
                "input": {"city": "Paris"},
            }
        ],
    }


async def test_complete_sends_tool_message_as_user_tool_result_blocks() -> None:
    """A `ToolMessage` becomes a user turn of `tool_result` blocks."""

    # GIVEN a continuation transcript whose assistant requested two tool calls,
    # now answered with one ok and one error result
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    history: list[Message] = [
        UserMessage.from_text("weather?"),
        AssistantMessage(
            parts=[
                ToolCallPart(
                    call_id="ok_1", tool_name="get_weather", args={"city": "Paris"}
                ),
                ToolCallPart(
                    call_id="err_1", tool_name="get_weather", args={"city": "?"}
                ),
            ],
            stop_reason="tool_use",
        ),
        ToolMessage(
            parts=[
                ToolResultPart(call_id="ok_1", result=ToolResultOk(content="sunny")),
                ToolResultPart(call_id="err_1", result=ToolResultError(content="boom")),
            ]
        ),
    ]

    # WHEN `complete` is invoked
    await provider.complete(history, _settings())

    # THEN the results are sent as a user turn, with `is_error` set per status
    wire_messages = mock_client.messages.create.call_args.kwargs["messages"]
    assert wire_messages[-1] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "ok_1",
                "content": "sunny",
                "is_error": False,
            },
            {
                "type": "tool_result",
                "tool_use_id": "err_1",
                "content": "boom",
                "is_error": True,
            },
        ],
    }


# Lifecycle tests
# -----------------------------------------------------------------------------


def _provider_owning(
    monkeypatch: pytest.MonkeyPatch,
    client: AsyncMock,
) -> AnthropicProvider:
    """Construct a provider that "owns" a mock client.

    Patches the `AsyncAnthropic` symbol in the provider module so that the
    no-`client=` path yields the supplied mock - giving the test a handle on
    the would-be-self-constructed client without making real network calls.
    """

    def _factory(**_: object) -> AsyncMock:
        return client

    monkeypatch.setattr("avior.providers.anthropic.AsyncAnthropic", _factory)
    return AnthropicProvider(api_key="fake")


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
