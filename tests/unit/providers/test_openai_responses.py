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
    ResponseFunctionWebSearch,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseUsage,
)
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_function_web_search import ActionSearch
from openai.types.responses.response_reasoning_item import Summary
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)
from pydantic import BaseModel, JsonValue

from avior.core.context import RunContext
from avior.core.exceptions import (
    AviorUsageError,
    ProviderConnectionError,
    ProviderError,
    ProviderHTTPError,
    ProviderResponseValidationError,
)
from avior.core.messages import (
    AssistantMessage,
    Message,
    StopReason,
    TextPart,
    ThinkingPart,
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
    thinking: bool | Literal["low", "medium", "high"] | None = None,
    provider_options: dict[str, dict[str, JsonValue]] | None = None,
) -> ModelSettings:
    """Construct `ModelSettings` with sensible defaults for tests."""

    return ModelSettings(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking=thinking,
        provider_options=provider_options or {},
    )


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


def _status_response(
    status: Literal["failed", "cancelled", "queued", "in_progress"],
) -> Response:
    """Build an empty `Response` with the given status."""

    return Response(
        id="resp_test",
        object="response",
        created_at=0.0,
        model="gpt-test",
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        status=status,
    )


def _response_with_output(output: list[ResponseOutputItem]) -> Response:
    """Build a completed `Response` carrying the given output items."""

    return Response(
        id="resp_test",
        object="response",
        created_at=0.0,
        model="gpt-test",
        output=output,
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        status="completed",
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

    async def execute(self, ctx: RunContext[object], args: _CityArgs) -> str:
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

    # GIVEN a response with a single text item
    response = _response("Hi!")
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hello")], _settings())

    # THEN the result is the assistant message containing the response text
    assert result.message.text == "Hi!"


async def test_complete_sends_system_prompt_as_instructions() -> None:
    """`complete` sends the `system_prompt` as the `instructions` string."""

    # GIVEN a mock client and a system prompt alongside a user message
    mock_client = _mock_client_returning(_response("Hi!"))
    provider = _provider(mock_client)

    # WHEN `complete` is invoked with a system prompt
    await provider.complete(
        [UserMessage.from_text("hello")],
        _settings(),
        system_prompt="be helpful",
    )

    # THEN the OpenAI SDK call receives the system prompt as the top-level
    # `instructions` string and the user message goes in `input`
    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert call_kwargs["instructions"] == "be helpful"
    assert len(call_kwargs["input"]) == 1
    assert call_kwargs["input"][0]["role"] == "user"
    assert call_kwargs["input"][0]["content"] == "hello"


async def test_complete_omits_system_prompt_when_none() -> None:
    """`complete` passes `omit` when `system_prompt` is `None`."""

    # GIVEN a mock client
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)

    # WHEN `complete` is invoked with no system prompt
    await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the `instructions` kwarg is the `omit` sentinel
    assert mock_client.responses.create.call_args.kwargs["instructions"] is omit


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

    The Responses API does not require `max_output_tokens`, so an unset
    `max_tokens` forwards no default and the model uses its own maximum output.
    """

    # GIVEN a mock client and settings without an explicit `max_tokens`
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    settings = _settings(max_tokens=None)

    # WHEN `complete` is invoked
    await provider.complete([UserMessage.from_text("hi")], settings)

    # THEN the `max_output_tokens` kwarg is the `omit` sentinel
    assert mock_client.responses.create.call_args.kwargs["max_output_tokens"] is omit


async def test_complete_omits_temperature_when_unset() -> None:
    """`complete` passes `omit` for `temperature` when not set on settings."""

    # GIVEN a mock client and settings without an explicit `temperature`
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    settings = _settings(temperature=None)

    # WHEN `complete` is invoked
    await provider.complete([UserMessage.from_text("hi")], settings)

    # THEN the `temperature` kwarg is the `omit` sentinel
    assert mock_client.responses.create.call_args.kwargs["temperature"] is omit


async def test_complete_passes_store_false() -> None:
    """`complete` always passes `store=False` (stateless wire)."""

    # GIVEN a mock client and any messages
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)

    # WHEN `complete` is invoked
    await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the `store` kwarg is `False`, so no server-side history is created
    assert mock_client.responses.create.call_args.kwargs["store"] is False


async def test_complete_maps_each_response_text_item_to_a_part() -> None:
    """`complete` maps each response text item to its own `TextPart`."""

    # GIVEN a response with two text items
    response = _response("hello ", "world")
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the returned message has one `TextPart` per response item, in order
    assert result.message.parts == [TextPart(text="hello "), TextPart(text="world")]


async def test_complete_returns_empty_parts_when_response_output_is_empty() -> None:
    """`complete` returns `parts=[]` when the response has no output items."""

    # GIVEN a response with empty output (zero items)
    response = _response()
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the result has an empty parts list (not a single empty `TextPart`)
    assert result.message.parts == []


async def test_complete_decodes_reasoning_item_into_thinking_part() -> None:
    """A reasoning item decodes into a `ThinkingPart` before the message text.

    The summary joins into the readable content, and the reasoning item's id and
    `encrypted_content` are kept in `provider_details` for the round-trip.
    """

    # GIVEN a response carrying a reasoning item (summary + encrypted content)
    # before a text message
    response = _response_with_output(
        [
            ResponseReasoningItem(
                id="rs_1",
                type="reasoning",
                summary=[Summary(type="summary_text", text="thinking")],
                encrypted_content="enc-token",
            ),
            ResponseOutputMessage(
                id="msg_test",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    ResponseOutputText(type="output_text", text="Hi!", annotations=[])
                ],
            ),
        ]
    )
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN a `ThinkingPart` precedes the text, carrying the summary and the
    # round-trip token
    assert len(result.message.parts) == 2
    thinking, text = result.message.parts
    assert isinstance(thinking, ThinkingPart)
    assert isinstance(text, TextPart)
    assert thinking.content == "thinking"
    assert thinking.provider_details == {
        "reasoning_id": "rs_1",
        "encrypted_content": "enc-token",
    }
    assert text.text == "Hi!"


@pytest.mark.parametrize(
    ("model", "reasoning_active"),
    [
        ("o4-mini", True),
        ("gpt-5", True),
        ("gpt-5.5", True),
        ("gpt-5.1", False),
        ("gpt-5-chat-latest", False),
        ("gpt-4.1-nano", False),
    ],
    ids=[
        "o-series",
        "gpt-5",
        "on-by-default",
        "off-by-default",
        "gpt-5-chat",
        "non-reasoning-model",
    ],
)
async def test_complete_requests_encrypted_reasoning_by_model(
    model: str,
    reasoning_active: bool,
) -> None:
    """`complete` asks for encrypted reasoning content only when reasoning is
    active for the request, and omits the request otherwise.

    With `thinking` unset, the model's own default decides: an `always_on` or
    `on_by_default` model reasons, the others do not.
    """

    # GIVEN a mock client and settings for the given model
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    settings = _settings(model=model)

    # WHEN `complete` is invoked
    await provider.complete([UserMessage.from_text("hi")], settings)

    # THEN the request's `include` parameter tells OpenAI to return the
    # encrypted reasoning content only when reasoning is active
    include = mock_client.responses.create.call_args.kwargs["include"]
    if reasoning_active:
        assert include == ["reasoning.encrypted_content"]
    else:
        assert include is omit


async def test_complete_echoes_reasoning_item_with_its_round_trip_token() -> None:
    """A `ThinkingPart` echoes back as a `reasoning` item carrying its token.

    The provider rebuilds the wire reasoning item from `provider_details`: the
    `reasoning_id` becomes the item `id` and the `encrypted_content` rides
    along, so the reasoning round-trips to OpenAI intact.
    """

    # GIVEN a message history whose assistant turn, from this provider, carries
    # a reasoning step ahead of its tool call
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    history: list[Message] = [
        UserMessage.from_text("weather?"),
        AssistantMessage(
            parts=[
                ThinkingPart(
                    content="I need to look up the weather in Paris.",
                    provider_details={
                        "reasoning_id": "rs_1",
                        "encrypted_content": "enc-token",
                    },
                ),
                ToolCallPart(
                    call_id="call_1", tool_name="get_weather", args={"city": "Paris"}
                ),
            ],
            stop_reason="tool_use",
            provider_name="openai",
        ),
        ToolMessage(
            parts=[
                ToolResultPart(call_id="call_1", result=ToolResultOk(content="sunny"))
            ]
        ),
    ]

    # WHEN `complete` is invoked on a reasoning model
    await provider.complete(history, _settings(model="o4-mini"))

    # THEN the reasoning step is rebuilt as a `reasoning` item carrying its
    # round-trip token, ahead of the `function_call`
    wire_input = mock_client.responses.create.call_args.kwargs["input"]
    assistant_items = wire_input[1:]
    assert assistant_items[0] == {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "encrypted_content": "enc-token",
    }
    assert [i["type"] for i in assistant_items] == [
        "reasoning",
        "function_call",
        "function_call_output",
    ]


@pytest.mark.parametrize(
    ("model", "provider_name"),
    [("o4-mini", "anthropic"), ("gpt-4.1-nano", "openai")],
    ids=["foreign-provider", "non-reasoning-destination"],
)
async def test_complete_drops_reasoning_item_when_not_replayable(
    model: str,
    provider_name: str,
) -> None:
    """A reasoning item is left out when it cannot round-trip.

    A reasoning item is echoed only back to the provider that produced it and
    only when reasoning is active for the request; otherwise the opaque token
    would be rejected, so the item is dropped.
    """

    # GIVEN a message history whose assistant reasoning cannot round-trip - it
    # came from another provider, or reasoning is not active for the request
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    history: list[Message] = [
        UserMessage.from_text("hi"),
        AssistantMessage(
            parts=[
                ThinkingPart(
                    content="thinking",
                    provider_details={
                        "reasoning_id": "rs_1",
                        "encrypted_content": "enc-token",
                    },
                ),
                TextPart(text="Hi!"),
            ],
            stop_reason="stop",
            provider_name=provider_name,
        ),
    ]

    # WHEN `complete` is invoked
    await provider.complete(history, _settings(model=model))

    # THEN no `reasoning` item is sent
    wire_input = mock_client.responses.create.call_args.kwargs["input"]
    assert "reasoning" not in [item["type"] for item in wire_input]


# Reasoning-config tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "supports_thinking"),
    [
        ("o3", True),
        ("gpt-5", True),
        ("gpt-5.4-mini", True),
        ("gpt-5.6", True),
        ("gpt-5.6-sol", True),
        ("gpt-5.6-terra", True),
        ("gpt-5.6-luna", True),
        ("gpt-5.6-sol-2026-01-01", True),
        ("gpt-5-chat-latest", False),
        ("gpt-4o", False),
        ("gpt-5.4-cyber", False),
        ("gpt-5.60", False),
        ("gpt-5.6-solar", False),
    ],
    ids=[
        "o-series",
        "gpt-5",
        "recognized-variant",
        "on-by-default",
        "on-by-default-sol",
        "on-by-default-terra",
        "on-by-default-luna",
        "on-by-default-tier-snapshot",
        "gpt-5-chat",
        "non-reasoning",
        "unlisted-variant",
        "near-miss-version",
        "near-miss-variant",
    ],
)
def test_model_capabilities_reports_thinking_for_reasoning_models(
    model: str,
    supports_thinking: bool,
) -> None:
    """`model_capabilities` reports whether a model supports thinking.

    A recognized named variant (`gpt-5.4-mini`) reports support, while an
    unlisted specialized variant (`gpt-5.4-cyber`) is not recognized and reports
    none - it can instead be driven through the raw provider options.
    """

    # GIVEN a provider
    provider = _provider(_mock_client_returning(_response("ok")))

    # WHEN the capabilities for the model are read
    capabilities = provider.model_capabilities(model)

    # THEN `supports_thinking` reflects whether the model reasons
    assert capabilities.supports_thinking is supports_thinking


@pytest.mark.parametrize(
    ("model", "thinking", "expected_effort", "expect_warning"),
    [
        ("o4-mini", "high", "high", False),
        ("o4-mini", True, None, False),
        ("gpt-5.1", True, "medium", False),
        ("gpt-5.5", True, None, False),
        ("o4-mini", None, None, False),
        ("o4-mini", False, None, True),
        ("gpt-5", False, None, True),
        ("gpt-5-mini", False, None, True),
        ("gpt-5.1", False, "none", False),
        ("gpt-5.4-mini", False, "none", False),
        ("gpt-5.3-codex", False, "none", False),
        ("gpt-5.5", False, "none", False),
        ("gpt-5.6-sol", False, "none", False),
        ("gpt-5.1-codex", False, None, True),
        ("gpt-5.1-codex-max", False, None, True),
        ("gpt-5.5-pro", False, None, True),
        ("gpt-4o", "low", None, True),
        ("gpt-4o", False, None, False),
    ],
    ids=[
        "level-on-always-on",
        "true-on-always-on",
        "true-on-off-by-default",
        "true-on-on-by-default",
        "unset",
        "disable-on-always-on",
        "disable-base-gpt5",
        "disable-mini-on-always-on",
        "disable-on-off-by-default",
        "disable-off-by-default-variant",
        "disable-codex-opt-in",
        "disable-on-on-by-default",
        "disable-on-on-by-default-tier",
        "disable-codex-under-off-by-default-family",
        "disable-codex-max-variant",
        "disable-pro-under-on-by-default-family",
        "enable-on-non-reasoning",
        "disable-on-non-reasoning",
    ],
)
async def test_complete_maps_portable_thinking_to_reasoning(
    model: str,
    thinking: bool | Literal["low", "medium", "high"] | None,
    expected_effort: str | None,
    expect_warning: bool,
) -> None:
    """`complete` maps the portable `thinking` to OpenAI's `reasoning` param.

    - a level becomes `reasoning.effort`;
    - `True` enables reasoning: a model that already reasons by default keeps
      its own default (no config), while an `off_by_default` model gets the
      default effort (`medium`);
    - `False` becomes `effort="none"` on an `off_by_default` or `on_by_default`
      model, but is dropped with a warning on an `always_on` one;
    - enabling a non-reasoning model warns; disabling one is a silent no-op.
    """

    # GIVEN settings carrying the portable thinking value for the model
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    settings = _settings(model=model, thinking=thinking)

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], settings)

    # THEN the `reasoning` kwarg and any warning match the mapping
    reasoning = mock_client.responses.create.call_args.kwargs["reasoning"]
    if expected_effort is None:
        assert reasoning is omit
    else:
        assert reasoning == {"effort": expected_effort}
    thinking_warnings = [w for w in result.warnings if w.setting_name == "thinking"]
    assert len(thinking_warnings) == (1 if expect_warning else 0)


async def test_complete_raw_reasoning_option_overrides_portable_thinking() -> None:
    """A raw `reasoning` provider option overrides the portable mapping.

    It is sent in place of the portable value, which is not consulted - so the
    warning that value would otherwise raise never arises.
    """

    # GIVEN settings whose portable `thinking` would warn (disable on a
    # reasoning model) alongside a raw `reasoning` option
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    settings = _settings(
        model="o4-mini",
        thinking=False,
        provider_options={
            "openai": {"reasoning": {"effort": "low", "summary": "auto"}}
        },
    )

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], settings)

    # THEN the raw config is sent
    assert mock_client.responses.create.call_args.kwargs["reasoning"] == {
        "effort": "low",
        "summary": "auto",
    }
    # AND no thinking warning is recorded
    assert [w for w in result.warnings if w.setting_name == "thinking"] == []


async def test_complete_raw_reasoning_none_overrides_portable_thinking() -> None:
    """A raw `effort="none"` turns reasoning off despite a portable level.

    The raw option replaces the portable mapping, so the request is not
    reasoning-active: `effort="none"` is sent, and a custom `temperature` is
    forwarded without a warning.
    """

    # GIVEN settings whose portable `thinking` would enable reasoning,
    # alongside a raw `reasoning` option disabling it and a custom temperature
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    settings = _settings(
        model="gpt-5.5",
        thinking="high",
        temperature=0.3,
        provider_options={"openai": {"reasoning": {"effort": "none"}}},
    )

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], settings)

    # THEN `effort="none"` is sent, and the temperature is forwarded with no
    # warning
    create_kwargs = mock_client.responses.create.call_args.kwargs
    assert create_kwargs["reasoning"] == {"effort": "none"}
    assert create_kwargs["temperature"] == 0.3
    assert result.warnings == []


async def test_complete_requests_encrypted_reasoning_for_raw_option() -> None:
    """A raw `reasoning` option marks an unclassified model as a reasoning one.

    The raw option drives reasoning on a model avior does not classify, so the
    encrypted-content round-trip follows and `include` is requested.
    """

    # GIVEN an unclassified model carrying a raw `reasoning` option
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    settings = _settings(
        model="gpt-6-unknown",
        provider_options={"openai": {"reasoning": {"effort": "high"}}},
    )

    # WHEN `complete` is awaited
    await provider.complete([UserMessage.from_text("hi")], settings)

    # THEN encrypted reasoning content is requested despite the unknown model
    include = mock_client.responses.create.call_args.kwargs["include"]
    assert include == ["reasoning.encrypted_content"]


async def test_complete_omits_encrypted_reasoning_when_disabled() -> None:
    """Disabling reasoning turns off the encrypted-content round-trip.

    `thinking=False` on a disable-capable model sends `effort="none"`, so the
    model produces no reasoning items; `include` must not ask for encrypted
    content that will never come.
    """

    # GIVEN a disable-capable model with reasoning turned off
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    settings = _settings(model="gpt-5.1", thinking=False)

    # WHEN `complete` is awaited
    await provider.complete([UserMessage.from_text("hi")], settings)

    # THEN reasoning is disabled
    kwargs = mock_client.responses.create.call_args.kwargs
    assert kwargs["reasoning"] == {"effort": "none"}
    # AND no encrypted content is requested
    assert kwargs["include"] is omit


async def test_complete_omits_prior_reasoning_item_when_disabled() -> None:
    """Disabling reasoning also drops a prior reasoning item from the wire.

    `thinking=False` on a disable-capable model makes reasoning inactive, so an
    earlier same-provider reasoning step is not echoed back; its opaque token
    would otherwise ride along a request that produces no reasoning.
    """

    # GIVEN a history whose same-provider assistant turn carries a reasoning
    # step, and settings that disable reasoning
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    history: list[Message] = [
        UserMessage.from_text("weather?"),
        AssistantMessage(
            parts=[
                ThinkingPart(
                    content="checking",
                    provider_details={
                        "reasoning_id": "rs_1",
                        "encrypted_content": "enc-token",
                    },
                ),
                TextPart(text="Sunny."),
            ],
            stop_reason="stop",
            provider_name="openai",
        ),
        UserMessage.from_text("and tomorrow?"),
    ]
    settings = _settings(model="gpt-5.1", thinking=False)

    # WHEN `complete` is awaited
    await provider.complete(history, settings)

    # THEN the prior reasoning item is not echoed back
    wire_input = mock_client.responses.create.call_args.kwargs["input"]
    assert "reasoning" not in [item["type"] for item in wire_input]


async def test_complete_rejects_invalid_openai_provider_options() -> None:
    """An invalid `openai` provider-options slice raises `AviorUsageError`."""

    # GIVEN settings with an unknown key in the openai provider options
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    settings = _settings(provider_options={"openai": {"unknown": 1}})

    # WHEN `complete` is awaited
    # THEN it raises `AviorUsageError` before any request is sent
    with pytest.raises(AviorUsageError):
        await provider.complete([UserMessage.from_text("hi")], settings)
    mock_client.responses.create.assert_not_called()


async def test_complete_raises_on_unsupported_output_item() -> None:
    """An output item avior cannot represent raises rather than dropping.

    A built-in tool item (here a web-search call) carries output the adapter
    does not map, so it fails loud instead of returning a misleading success.
    """

    # GIVEN a response carrying a web-search call the adapter does not map
    response = _response_with_output(
        [
            ResponseFunctionWebSearch(
                id="ws_1",
                type="web_search_call",
                status="completed",
                action=ActionSearch(type="search", query="x"),
            )
        ]
    )
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    # THEN `ProviderResponseValidationError` is raised
    with pytest.raises(ProviderResponseValidationError):
        await provider.complete([UserMessage.from_text("hi")], _settings())


# Sampling-parameter compatibility tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "thinking", "provider_options"),
    [
        ("o4-mini", None, None),
        ("gpt-5.1", "medium", None),
        ("gpt-5.5", None, None),
        ("gpt-5.5", None, {"openai": {"reasoning": {"summary": "auto"}}}),
        ("gpt-test", None, {"openai": {"reasoning": {"effort": "medium"}}}),
    ],
    ids=[
        "always-on-model",
        "off-by-default-with-effort",
        "on-by-default-at-default",
        "on-by-default-summary-only",
        "raw-options-enable",
    ],
)
async def test_complete_drops_temperature_openai_rejects(
    model: str,
    thinking: Literal["medium"] | None,
    provider_options: dict[str, dict[str, JsonValue]] | None,
) -> None:
    """`complete` drops a non-default `temperature` OpenAI would reject.

    OpenAI rejects a custom `temperature` whenever reasoning is active for the
    request: on an `always_on` model unconditionally, on an `off_by_default`
    model running with an effort, on an `on_by_default` model left at its
    reasoning default (with or without a raw config lacking an effort), and
    when a raw `provider_options` reasoning config enables reasoning on a
    model avior does not classify.  Each drop carries a warning with the
    cause.
    """

    # GIVEN settings with a non-default temperature OpenAI would reject
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    settings = _settings(
        model=model,
        temperature=0.3,
        thinking=thinking,
        provider_options=provider_options,
    )

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], settings)

    # THEN no temperature is sent, and a warning records the drop and its cause
    assert mock_client.responses.create.call_args.kwargs["temperature"] is omit
    temperature_warnings = [
        w for w in result.warnings if w.setting_name == "temperature"
    ]
    assert len(temperature_warnings) == 1
    assert temperature_warnings[0].setting_value == 0.3
    assert temperature_warnings[0].reason == (
        "a custom temperature is not accepted while reasoning is active"
    )


@pytest.mark.parametrize(
    ("model", "temperature", "thinking", "provider_options"),
    [
        ("gpt-test", 0.3, None, None),
        ("gpt-5.1", 0.3, None, None),
        ("gpt-5.5", 0.3, False, None),
        ("gpt-5.1", 1.0, "medium", None),
        ("gpt-5.1", 0.3, None, {"openai": {"reasoning": {"summary": "auto"}}}),
    ],
    ids=[
        "non-thinking-model",
        "off-by-default-reasoning-unset",
        "on-by-default-reasoning-disabled",
        "default-temperature-with-effort",
        "off-by-default-summary-only",
    ],
)
async def test_complete_forwards_accepted_temperature(
    model: str,
    temperature: float,
    thinking: bool | Literal["medium"] | None,
    provider_options: dict[str, dict[str, JsonValue]] | None,
) -> None:
    """`complete` forwards a `temperature` OpenAI accepts, without warning.

    OpenAI accepts a custom `temperature` whenever reasoning is not active for
    the request - a non-reasoning model, an `off_by_default` model whose
    reasoning is unset (even when a raw config carries only a summary), or any
    model whose reasoning is disabled - and always accepts the default `1`.
    Each parametrized case is one such accepted combination, named by its id.
    """

    # GIVEN settings with a temperature OpenAI accepts
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    settings = _settings(
        model=model,
        temperature=temperature,
        thinking=thinking,
        provider_options=provider_options,
    )

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], settings)

    # THEN the temperature is sent unchanged, with no warning
    assert mock_client.responses.create.call_args.kwargs["temperature"] == temperature
    assert result.warnings == []


# Call-metadata mapping tests
# -----------------------------------------------------------------------------


async def test_complete_maps_usage_ids_and_model_onto_provider_response() -> None:
    """`complete` maps OpenAI usage, response id, and model onto the wrapper."""

    # GIVEN a response carrying the call metadata (usage, id, and served model)
    usage = ResponseUsage(
        input_tokens=11,
        output_tokens=7,
        total_tokens=18,
        input_tokens_details=InputTokensDetails(cached_tokens=4),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=3),
    )
    response = _response("hi", usage=usage)
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN usage is normalized: OpenAI's input/output already include their
    # cache/reasoning sub-slices, so the totals are used as-is and the nested
    # details surface as sub-slices; cache_write is 0 (OpenAI has no separate
    # cache-write counter); the derived total equals OpenAI's own reported total
    # (confirming input/output already include their sub-slices)
    assert result.usage is not None
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 7
    assert result.usage.reasoning_tokens == 3
    assert result.usage.cache_read_tokens == 4
    assert result.usage.cache_write_tokens == 0
    assert result.usage.total_tokens == 18
    assert result.usage.total_tokens == usage.total_tokens

    # AND the provider-native usage is preserved beside the normalized counts
    assert result.raw_usage is not None
    assert result.raw_usage["input_tokens"] == 11

    # AND the response id, served model, and provider name are populated
    assert result.response_id == "resp_test"
    assert result.model == "gpt-test"
    assert result.provider_name == "openai"


async def test_complete_maps_absent_usage_to_none() -> None:
    """`complete` yields `usage=None` when the response carries no usage."""

    # GIVEN a response with no usage attached
    response = _response("hi")
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN usage and raw usage are both `None` (not a zero-filled `Usage`)
    assert result.usage is None
    assert result.raw_usage is None


async def test_complete_warns_when_provider_total_diverges_from_derived(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A provider total that disagrees with input+output logs a warning."""

    # GIVEN a response whose `total_tokens` contradicts input+output
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
    mock_client = _mock_client_raising(openai_error)
    provider = _provider(mock_client)

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
    mock_client = _mock_client_raising(openai_error)
    provider = _provider(mock_client)

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
    mock_client = _mock_client_raising(openai_error)
    provider = _provider(mock_client)

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
    mock_client = _mock_client_raising(openai_error)
    provider = _provider(mock_client)

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
    mock_client = _mock_client_raising(openai_error)
    provider = _provider(mock_client)

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

    # GIVEN a normal completed response
    response = _response("Hi!")
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the returned message carries `stop_reason="stop"`
    assert result.message.stop_reason == "stop"


@pytest.mark.parametrize(
    ("reason", "expected_stop_reason"),
    [
        ("max_output_tokens", "max_tokens"),
        ("content_filter", "content_filter"),
    ],
    ids=["max_output_tokens", "content_filter"],
)
async def test_complete_maps_incomplete_reason_to_stop_reason(
    reason: Literal["max_output_tokens", "content_filter"],
    expected_stop_reason: StopReason,
) -> None:
    """An `incomplete_details.reason` maps to its canonical `StopReason`."""

    # GIVEN an incomplete response carrying the given reason
    response = _incomplete_response(reason)
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN it maps to the expected canonical stop reason
    assert result.message.stop_reason == expected_stop_reason


@pytest.mark.parametrize(
    "status",
    ["failed", "cancelled", "queued", "in_progress"],
    ids=["failed", "cancelled", "queued", "in_progress"],
)
async def test_complete_maps_abnormal_status_to_error(
    status: Literal["failed", "cancelled", "queued", "in_progress"],
) -> None:
    """A failed / cancelled / non-terminal status maps to `"error"`.

    The body decoded fine, but the response carries no usable result, so it must
    surface as the canonical `"error"` (a run failure) rather than a successful
    empty stop.
    """

    # GIVEN a response carrying an abnormal status
    response = _status_response(status)
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the canonical `stop_reason` is `"error"`, not `"stop"`
    assert result.message.stop_reason == "error"


async def test_complete_maps_refusal_content_part_to_refusal_stop_reason() -> None:
    """A `ResponseOutputRefusal` content part maps to canonical `"refusal"`."""

    # GIVEN a completed response carrying a refusal part
    response = _refusal_response("I can't help.")
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the canonical `stop_reason` is `"refusal"` and the refusal text
    # is preserved in `parts`
    assert result.message.stop_reason == "refusal"
    assert result.message.text == "I can't help."


async def test_complete_refusal_overrides_text_when_both_present() -> None:
    """When the response holds both text and a refusal, the refusal wins: the
    stop reason is `"refusal"` and the partial text is dropped.
    """

    # GIVEN a completed response whose message holds a text part followed by a
    # refusal part (rare, but possible in non-streaming output)
    response = Response(
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
                content=[
                    ResponseOutputText(
                        type="output_text", text="Here is how to", annotations=[]
                    ),
                    ResponseOutputRefusal(type="refusal", refusal="I can't help."),
                ],
            )
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        status="completed",
    )
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the refusal is the authoritative final word: it sets the stop reason
    # and replaces the partial text
    assert result.message.stop_reason == "refusal"
    assert result.message.text == "I can't help."


# Tool-calling tests
# -----------------------------------------------------------------------------


async def test_complete_parses_function_call_into_tool_call_part() -> None:
    """A response `function_call` item decodes into a `ToolCallPart`."""

    # GIVEN a function-call response
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

    # GIVEN a function-call response with an empty arguments string
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

    # GIVEN a function-call response whose arguments are not valid JSON
    response = _function_call_response("call_1", "get_weather", "not json")
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    # THEN the decode failure surfaces as a provider validation error
    with pytest.raises(ProviderResponseValidationError):
        await provider.complete([UserMessage.from_text("weather?")], _settings())


async def test_complete_maps_function_call_to_tool_use_stop_reason() -> None:
    """A response carrying a `function_call` maps to canonical `"tool_use"`."""

    # GIVEN a function-call response
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

    # GIVEN an incomplete (max_output_tokens) response with a `function_call`
    # cut off mid-arguments, leaving invalid JSON
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

    # GIVEN an incomplete (no reason) response with a `function_call` cut off
    # mid-arguments, leaving invalid JSON
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

    # GIVEN a mock client and an offered tool
    mock_client = _mock_client_returning(_response("ok"))
    provider = _provider(mock_client)
    tool = _Weather()

    # WHEN `complete` is invoked with that tool
    await provider.complete(
        [UserMessage.from_text("hi")],
        _settings(),
        tools=[tool],
    )

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

    # GIVEN a mock client and no tools offered
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
