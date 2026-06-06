"""Tests for `avior.providers.openai_responses`."""

from typing import Literal, cast
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
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseUsage,
)
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)
from pydantic import BaseModel

from avior.core.exceptions import (
    ProviderConnectionError,
    ProviderError,
    ProviderHTTPError,
    ProviderResponseValidationError,
)
from avior.core.messages import (
    AssistantMessage,
    Message,
    SystemMessage,
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
from avior.providers.openai_responses import OpenAIResponsesProvider


def _settings(
    *,
    model: str = "gpt-test",
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> ModelSettings:
    """Construct `ModelSettings` with sensible defaults for tests."""

    return ModelSettings(model=model, max_tokens=max_tokens, temperature=temperature)


def _response(*texts: str, usage: ResponseUsage | None = None) -> Response:
    """Build a minimal `openai.types.responses.Response` with text items.

    One `ResponseOutputMessage` is emitted containing one `ResponseOutputText`
    per supplied text.  Empty `texts` produces a response with an empty
    `output` list (no message item at all).  Pass `usage` to attach token
    usage (default: none).
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
        usage=usage,
    )


def _incomplete_response(
    reason: Literal["max_output_tokens", "content_filter"] | None,
    output: list[ResponseOutputItem] | None = None,
) -> Response:
    """Build a `Response` with `status="incomplete"` and the given reason.

    `reason=None` models an incomplete response whose reason the SDK left
    unset.  `output` defaults to empty; pass items (e.g. a truncated
    `function_call`) to model output the model had started before stopping.
    """

    return Response(
        id="resp_test",
        object="response",
        created_at=0.0,
        model="gpt-test",
        output=output or [],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        status="incomplete",
        incomplete_details=IncompleteDetails(reason=reason),
    )


def _refusal_response(refusal_text: str) -> Response:
    """Build a completed `Response` whose message content is a refusal."""

    return Response(
        id="resp_test",
        object="response",
        created_at=0.0,
        model="gpt-test",
        output=[
            ResponseOutputMessage(
                id="msg_test",
                type="message",
                role="assistant",
                status="completed",
                content=[ResponseOutputRefusal(type="refusal", refusal=refusal_text)],
            )
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        status="completed",
    )


class _CityArgs(BaseModel):
    city: str


class _Weather(Tool[_CityArgs, str]):
    """A trivial tool used to exercise tool-calling wire translation."""

    name = "get_weather"
    description = "Look up the weather for a city."
    args_model = _CityArgs

    async def execute(self, args: _CityArgs) -> str:
        return "sunny"


def _function_call_response(
    call_id: str,
    tool_name: str,
    arguments: str,
) -> Response:
    """Build a completed `Response` carrying a single `function_call` item.

    `arguments` is the raw JSON string the Responses API uses for call args.
    """

    return Response(
        id="resp_test",
        object="response",
        created_at=0.0,
        model="gpt-test",
        output=[
            ResponseFunctionToolCall(
                type="function_call",
                call_id=call_id,
                name=tool_name,
                arguments=arguments,
            )
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        status="completed",
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
    result = await provider.complete([UserMessage.from_text("hello")], _settings())

    # THEN the supplied client handles the call (proven by its preset response)
    assert result.message.text == "Hi from supplied client"


# Behavioural tests on `complete()`
# -----------------------------------------------------------------------------


async def test_complete_returns_assistant_message_parsed_from_response() -> None:
    """`complete` returns the assistant message decoded from the response."""

    # GIVEN a mock client returning a single-text-item assistant message
    mock_client = _mock_client_returning(_response("Hi!"))
    provider = _provider(mock_client)

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hello")], _settings())

    # THEN the result is the assistant message containing the response text
    assert result.message.text == "Hi!"


async def test_complete_lifts_leading_system_message_to_instructions() -> None:
    """`complete` extracts a leading system message and sends it top-level."""

    # GIVEN a mock client and messages with a leading system message and a user
    # message
    mock_client = _mock_client_returning(_response("Hi!"))
    provider = _provider(mock_client)
    messages: list[Message] = [
        SystemMessage.from_text("be helpful"),
        UserMessage.from_text("hello"),
    ]

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
    messages: list[Message] = [
        SystemMessage.from_text("first"),
        UserMessage.from_text("hi"),
        SystemMessage.from_text("later"),
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
    messages: list[Message] = [
        SystemMessage.from_text("s1"),
        UserMessage.from_text("u1"),
        AssistantMessage(parts=[TextPart(text="a1")], stop_reason="stop"),
        SystemMessage.from_text("s2"),
        UserMessage.from_text("u2"),
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
    messages: list[Message] = [SystemMessage.from_text(""), UserMessage.from_text("hi")]

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
    messages: list[Message] = [
        SystemMessage(parts=[]),
        SystemMessage.from_text("real instruction"),
        UserMessage.from_text("hi"),
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
    await provider.complete([UserMessage.from_text("hi")], _settings())

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
    await provider.complete([UserMessage.from_text("hi")], settings)

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
    await provider.complete([UserMessage.from_text("hi")], settings)

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
    await provider.complete([UserMessage.from_text("hi")], settings)

    # THEN the `temperature` kwarg is the `omit` sentinel
    kwargs = mock_client.responses.create.call_args.kwargs
    assert kwargs["temperature"] is omit


async def test_complete_passes_store_false() -> None:
    """`complete` always passes `store=False` (stateless wire)."""

    # GIVEN a mock client and any messages
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)

    # WHEN `complete` is invoked
    await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the `store` kwarg is `False`, so no server-side history is created
    kwargs = mock_client.responses.create.call_args.kwargs
    assert kwargs["store"] is False


async def test_complete_maps_each_response_text_item_to_a_part() -> None:
    """`complete` maps each response text item to its own `TextPart`."""

    # GIVEN a mock client returning a response with two text items
    mock_client = _mock_client_returning(_response("hello ", "world"))
    provider = _provider(mock_client)

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the returned message has one `TextPart` per response item, in order
    assert result.message.parts == [TextPart(text="hello "), TextPart(text="world")]


async def test_complete_returns_empty_parts_when_response_output_is_empty() -> None:
    """`complete` returns `parts=[]` when the response has no output items."""

    # GIVEN a mock client returning a response with empty output (zero items)
    mock_client = _mock_client_returning(_response())
    provider = _provider(mock_client)

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the result has an empty parts list (not a single empty `TextPart`)
    assert result.message.parts == []


# Call-metadata mapping tests
# -----------------------------------------------------------------------------


async def test_complete_maps_usage_ids_and_model_onto_provider_response() -> None:
    """`complete` maps OpenAI usage, response id, and model onto the wrapper."""

    # GIVEN a mock client returning a completed response with the call metadata
    usage = ResponseUsage(
        input_tokens=11,
        output_tokens=7,
        total_tokens=18,
        input_tokens_details=InputTokensDetails(cached_tokens=4),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=3),
    )
    provider = _provider(_mock_client_returning(_response("hi", usage=usage)))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN usage is normalized: OpenAI's input/output already include their
    # cache/reasoning sub-slices, so the totals are used as-is and the nested
    # details surface as sub-slices
    assert result.usage is not None
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 7
    assert result.usage.cache_read_tokens == 4
    assert result.usage.reasoning_tokens == 3

    # AND cache_write is 0: OpenAI has no separate cache-write counter
    assert result.usage.cache_write_tokens == 0

    # AND the derived total equals OpenAI's own reported total, confirming the
    # convention (input/output already include their sub-slices)
    assert result.usage.total_tokens == 18
    assert result.usage.total_tokens == usage.total_tokens

    # AND the response id, served model, and provider name are populated
    assert result.response_id == "resp_test"
    assert result.model == "gpt-test"
    assert result.provider_name == "openai"


async def test_complete_maps_absent_usage_to_none() -> None:
    """`complete` yields `usage=None` when the response carries no usage."""

    # GIVEN a mock client returning a response with no usage attached
    provider = _provider(_mock_client_returning(_response("hi")))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the wrapper's usage is None (not a zero-filled Usage)
    assert result.usage is None


async def test_complete_warns_when_provider_total_diverges_from_derived(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A provider total that disagrees with input+output logs a warning."""

    # GIVEN a mock client returning a response whose `total_tokens` contradicts
    # input+output
    response = _response(
        "hi",
        usage=ResponseUsage(
            input_tokens=11,
            output_tokens=7,
            total_tokens=42,  # inconsistent with 11 + 7
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        ),
    )
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    with caplog.at_level("WARNING", logger="avior.providers.openai_responses"):
        result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN exactly one WARNING is logged, naming both conflicting totals
    assert len(caplog.records) == 1
    warning = caplog.records[0]
    assert warning.levelname == "WARNING"
    assert "total_tokens=42" in warning.getMessage()
    assert "derived total 18" in warning.getMessage()

    # AND avior reports the derived total, with the provider's own kept in raw
    assert result.usage is not None
    assert result.usage.total_tokens == 18
    assert result.raw_usage is not None
    assert result.raw_usage["total_tokens"] == 42


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
        await provider.complete([UserMessage.from_text("hi")], _settings())
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
        await provider.complete([UserMessage.from_text("hi")], _settings())
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
        await provider.complete([UserMessage.from_text("hi")], _settings())
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
        await provider.complete([UserMessage.from_text("hi")], _settings())
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
        await provider.complete([UserMessage.from_text("hi")], _settings())
    assert type(exc_info.value) is ProviderError
    assert exc_info.value.__cause__ is openai_error


# Stop-reason mapping tests
# -----------------------------------------------------------------------------


async def test_complete_sets_stop_reason_stop_on_normal_completion() -> None:
    """`stop_reason="stop"` is set on a normal completed response."""

    # GIVEN a mock client returning a normal completed response
    provider = _provider(_mock_client_returning(_response("Hi!")))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the returned message carries `stop_reason="stop"`
    assert result.message.stop_reason == "stop"


async def test_complete_maps_max_output_tokens_to_max_tokens_stop_reason() -> None:
    """`incomplete_details.reason="max_output_tokens"` -> `"max_tokens"`."""

    # GIVEN a mock client returning a response truncated at max-tokens
    provider = _provider(
        _mock_client_returning(_incomplete_response("max_output_tokens"))
    )

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the canonical `stop_reason` is `"max_tokens"`
    assert result.message.stop_reason == "max_tokens"


async def test_complete_maps_content_filter_to_content_filter_stop_reason() -> None:
    """`incomplete_details.reason="content_filter"` -> `"content_filter"`."""

    # GIVEN a mock client returning an incomplete response due to content filter
    provider = _provider(_mock_client_returning(_incomplete_response("content_filter")))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the canonical `stop_reason` is `"content_filter"`
    assert result.message.stop_reason == "content_filter"


async def test_complete_maps_refusal_content_part_to_refusal_stop_reason() -> None:
    """A `ResponseOutputRefusal` content part maps to canonical `"refusal"`."""

    # GIVEN a mock client returning a completed response carrying a refusal part
    provider = _provider(_mock_client_returning(_refusal_response("I can't help.")))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the canonical `stop_reason` is `"refusal"` and the refusal text
    # is preserved in `parts`
    assert result.message.stop_reason == "refusal"
    assert result.message.text == "I can't help."


# Tool-calling tests
# -----------------------------------------------------------------------------


async def test_complete_parses_function_call_into_tool_call_part() -> None:
    """A response `function_call` item decodes into a `ToolCallPart`."""

    # GIVEN a provider whose mock client returns a function-call response
    response = _function_call_response("call_1", "get_weather", '{"city": "Paris"}')
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("weather?")], _settings())

    # THEN the item decodes into a `ToolCallPart`, with JSON-string arguments
    # parsed into a dict
    assert result.message.parts == [
        ToolCallPart(call_id="call_1", tool_name="get_weather", args={"city": "Paris"})
    ]


async def test_complete_parses_empty_arguments_into_empty_dict() -> None:
    """An empty `arguments` string decodes into an empty args dict."""

    # GIVEN a provider whose mock client returns a function-call response with
    # an empty arguments string
    response = _function_call_response("call_1", "ping", "")
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("ping")], _settings())

    # THEN the call decodes with empty args rather than failing to parse
    assert result.message.parts == [
        ToolCallPart(call_id="call_1", tool_name="ping", args={})
    ]


async def test_complete_raises_validation_error_on_malformed_arguments() -> None:
    """Non-JSON `arguments` map to `ProviderResponseValidationError`."""

    # GIVEN a provider whose mock client returns a function-call response with
    # arguments that are not valid JSON
    response = _function_call_response("call_1", "get_weather", "not json")
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    # THEN the decode failure surfaces as a provider validation error
    with pytest.raises(ProviderResponseValidationError):
        await provider.complete([UserMessage.from_text("weather?")], _settings())


async def test_complete_maps_function_call_to_tool_use_stop_reason() -> None:
    """A response carrying a `function_call` maps to canonical `"tool_use"`."""

    # GIVEN a provider whose mock client returns a function-call response
    response = _function_call_response("call_1", "get_weather", '{"city": "Paris"}')
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("weather?")], _settings())

    # THEN the canonical `stop_reason` is `"tool_use"`
    assert result.message.stop_reason == "tool_use"


async def test_complete_skips_decoding_truncated_tool_call_on_max_tokens() -> None:
    """A tool call truncated at max_output_tokens maps to max_tokens, not error.

    When OpenAI truncates a `function_call` mid-arguments, the leftover partial
    JSON must not surface as a schema error: the terminal incomplete reason
    wins and the truncated call is dropped rather than decoded.
    """

    # GIVEN a provider whose mock client returns an incomplete
    # (max_output_tokens) response with a function_call that was cut off
    # mid-arguments, leaving invalid JSON
    truncated = ResponseFunctionToolCall(
        type="function_call",
        call_id="call_1",
        name="get_weather",
        arguments='{"city":',
    )
    response = _incomplete_response("max_output_tokens", output=[truncated])
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("weather?")], _settings())

    # THEN it resolves to the max_tokens stop reason without raising, and the
    # truncated call is not surfaced as a tool-call part
    assert result.message.stop_reason == "max_tokens"
    assert result.message.parts == []


async def test_complete_skips_truncated_tool_call_on_reasonless_incomplete() -> None:
    """An incomplete response with no reason still skips truncated tool calls.

    The skip is gated on `status == "incomplete"`, not on a mapped reason, so a
    truncated `function_call` does not raise even when the reason is absent; the
    response maps to a plain `"stop"`.
    """

    # GIVEN a provider whose mock client returns an incomplete (no reason)
    # response with a function_call that was cut off mid-arguments, leaving
    # invalid JSON
    truncated = ResponseFunctionToolCall(
        type="function_call",
        call_id="call_1",
        name="get_weather",
        arguments='{"city":',
    )
    response = _incomplete_response(None, output=[truncated])
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("weather?")], _settings())

    # THEN the truncated call is dropped (not decoded) and no error is raised
    assert result.message.stop_reason == "stop"
    assert result.message.parts == []


async def test_complete_sends_tools_with_name_description_and_schema() -> None:
    """Each offered tool is sent as a non-strict function tool with a schema."""

    # GIVEN a provider and an offered tool
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)

    # WHEN `complete` is invoked with that tool
    await provider.complete([UserMessage.from_text("hi")], _settings(), [_Weather()])

    # THEN the OpenAI SDK call carries the tool's name, description, and args
    # schema, with `strict=False` (raw schema sent as advisory)
    tools_param = mock_client.responses.create.call_args.kwargs["tools"]
    assert tools_param == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Look up the weather for a city.",
            "parameters": _CityArgs.model_json_schema(),
            "strict": False,
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
    assert mock_client.responses.create.call_args.kwargs["tools"] is omit


async def test_complete_sends_assistant_tool_call_as_function_call_item() -> None:
    """An assistant `ToolCallPart` in the input becomes a `function_call`."""

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

    # THEN a `function_call` item carries the call's id, name, and JSON-encoded
    # arguments (this assistant turn has no text, so it emits no `message` item)
    wire_input = mock_client.responses.create.call_args.kwargs["input"]
    assistant_items = wire_input[1:]
    assert [i["type"] for i in assistant_items] == [
        "function_call",
        "function_call_output",
    ]
    assert assistant_items[0] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "get_weather",
        "arguments": '{"city": "Paris"}',
    }


async def test_complete_sends_assistant_text_before_function_call_items() -> None:
    """An assistant turn with text and a call emits the text `message` first."""

    # GIVEN a transcript whose assistant turn carries both text and a tool call
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    history: list[Message] = [
        UserMessage.from_text("weather?"),
        AssistantMessage(
            parts=[
                TextPart(text="Let me check."),
                ToolCallPart(
                    call_id="call_1", tool_name="get_weather", args={"city": "Paris"}
                ),
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

    # THEN the assistant's text leads as a `message` item, followed by the
    # `function_call` item
    wire_input = mock_client.responses.create.call_args.kwargs["input"]
    assistant_items = wire_input[1:]
    assert [i["type"] for i in assistant_items] == [
        "message",
        "function_call",
        "function_call_output",
    ]
    assert assistant_items[0]["role"] == "assistant"
    assert assistant_items[0]["content"] == "Let me check."


async def test_complete_sends_tool_message_as_function_call_output_items() -> None:
    """A `ToolMessage` becomes one `function_call_output` item per result."""

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

    # THEN each result is sent as a `function_call_output` item keyed by
    # `call_id`; the Responses API has no error flag, so the error status is
    # carried only in the output text
    wire_input = mock_client.responses.create.call_args.kwargs["input"]
    outputs = [i for i in wire_input if i["type"] == "function_call_output"]
    assert outputs == [
        {"type": "function_call_output", "call_id": "ok_1", "output": "sunny"},
        {"type": "function_call_output", "call_id": "err_1", "output": "boom"},
    ]


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
