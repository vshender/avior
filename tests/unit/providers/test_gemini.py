"""Tests for `avior.providers.gemini`."""

import base64
import logging
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, JsonValue

from avior.core.context import RunContext
from avior.core.exceptions import (
    AviorUsageError,
    ProviderConnectionError,
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
from avior.providers.gemini import GeminiProvider


def _settings(
    *,
    model: str = "gemini-test",
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


def _response(
    *parts: types.Part,
    finish_reason: types.FinishReason = types.FinishReason.STOP,
    usage: types.GenerateContentResponseUsageMetadata | None = None,
) -> types.GenerateContentResponse:
    """Build a minimal `GenerateContentResponse` with one model candidate."""

    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=list(parts)),
                finish_reason=finish_reason,
            )
        ],
        usage_metadata=usage,
        response_id="resp_test",
        model_version="gemini-test",
    )


def _text(
    *texts: str,
    usage: types.GenerateContentResponseUsageMetadata | None = None,
) -> types.GenerateContentResponse:
    """Build a response carrying the given text parts."""

    return _response(*[types.Part(text=t) for t in texts], usage=usage)


class _CityArgs(BaseModel):
    city: str


class _Weather(Tool[_CityArgs, str]):
    """A trivial tool used to exercise tool-calling wire translation."""

    name = "get_weather"
    description = "Look up the weather for a city."
    args_model = _CityArgs

    async def execute(self, ctx: RunContext[object], args: _CityArgs) -> str:
        return "sunny"


def _mock_client_returning(response: types.GenerateContentResponse) -> AsyncMock:
    """Mock `genai.Client` whose `generate_content` returns `response`."""

    mock = AsyncMock()
    mock.aio.models.generate_content = AsyncMock(return_value=response)
    return mock


def _mock_client_raising(error: Exception) -> AsyncMock:
    """Mock `genai.Client` whose `generate_content` raises `error`."""

    mock = AsyncMock()
    mock.aio.models.generate_content = AsyncMock(side_effect=error)
    return mock


def _provider(client: AsyncMock) -> GeminiProvider:
    """Wrap a mock client in a `GeminiProvider` for testing."""

    return GeminiProvider(client=cast(genai.Client, client))


def _call_config(client: AsyncMock) -> types.GenerateContentConfig:
    """Return the `config` passed to the mock's `generate_content`."""

    return cast(
        types.GenerateContentConfig,
        client.aio.models.generate_content.call_args.kwargs["config"],
    )


def _call_contents(client: AsyncMock) -> list[types.Content]:
    """Return the `contents` passed to the mock's `generate_content`."""

    return cast(
        list[types.Content],
        client.aio.models.generate_content.call_args.kwargs["contents"],
    )


def _call_config_dump(
    client: AsyncMock,
    *,
    exclude_none: bool = False,
) -> dict[str, Any]:
    """Return the call's `config` as a plain dict for assertions.

    Some `GenerateContentConfig` fields - `system_instruction` and `tools` -
    are typed by the Gemini SDK as unions that include an `Unknown` member
    (a `PIL.Image` from the optional `Pillow` dependency we do not install).
    Reading them off the object directly therefore fails strict type checking
    with `reportUnknownMemberType`.  `model_dump()` flattens the config to plain
    values, so tests can assert on them without a `cast` or `ignore` per
    assertion.  Cleanly-typed fields (`max_output_tokens`, `temperature`) are
    asserted on the object directly via `_call_config`.

    Pass `exclude_none=True` to drop unset fields, which the Gemini SDK fills
    with `None`.
    """

    return _call_config(client).model_dump(exclude_none=exclude_none)


# Constructor tests
# -----------------------------------------------------------------------------


async def test_provider_prefers_explicit_client_over_api_key() -> None:
    """`client` wins when both `client` and `api_key` are supplied."""

    # GIVEN a pre-built mock client preset to return a known response
    mock_client = _mock_client_returning(_text("Hi from supplied client"))

    # WHEN the provider is constructed with both `client` and `api_key` and
    # `complete` is awaited
    provider = GeminiProvider(
        client=cast(genai.Client, mock_client),
        api_key="ignored",
    )
    result = await provider.complete([UserMessage.from_text("hello")], _settings())

    # THEN the supplied client handles the call (proven by its preset response)
    assert result.message.text == "Hi from supplied client"


# Behavioural tests on `complete()`
# -----------------------------------------------------------------------------


async def test_complete_returns_assistant_message_parsed_from_response() -> None:
    """`complete` returns the assistant message decoded from the response."""

    # GIVEN a response with a single text part
    response = _text("Hi!")
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hello")], _settings())

    # THEN the result is the assistant message containing the response text
    assert result.message.text == "Hi!"


async def test_complete_sends_system_prompt_as_system_instruction() -> None:
    """`complete` sends the `system_prompt` in `config.system_instruction`."""

    # GIVEN a mock client, a system prompt, and a user message
    mock_client = _mock_client_returning(_text("Hi!"))
    provider = _provider(mock_client)
    system_prompt = "be helpful"
    user_message = UserMessage.from_text("hello")

    # WHEN `complete` is invoked with the system prompt
    await provider.complete(
        [user_message],
        _settings(),
        system_prompt=system_prompt,
    )

    # THEN the system prompt rides in the config
    assert _call_config_dump(mock_client)["system_instruction"] == system_prompt
    # AND the user message becomes a `"user"` content of text parts
    contents = _call_contents(mock_client)
    assert len(contents) == 1
    assert contents[0].role == "user"
    assert contents[0].parts == [types.Part(text="hello")]


async def test_complete_omits_system_prompt_when_none() -> None:
    """`complete` leaves `config.system_instruction` unset when `None`."""

    # GIVEN a mock client
    mock_client = _mock_client_returning(_text("ok"))
    provider = _provider(mock_client)

    # WHEN `complete` is invoked with no system prompt
    await provider.complete([UserMessage.from_text("hello")], _settings())

    # THEN `system_instruction` is `None`
    assert _call_config_dump(mock_client)["system_instruction"] is None


async def test_complete_forwards_explicit_max_tokens_and_temperature() -> None:
    """`complete` forwards explicit `max_tokens` and `temperature` unchanged."""

    # GIVEN a mock client and settings with explicit `max_tokens` and
    # `temperature`
    mock_client = _mock_client_returning(_text("ok"))
    provider = _provider(mock_client)

    # WHEN `complete` is invoked
    await provider.complete(
        [UserMessage.from_text("hi")],
        _settings(max_tokens=2048, temperature=0.2),
    )

    # THEN the request's `GenerateContentConfig` carries the exact values
    config = _call_config(mock_client)
    assert config.max_output_tokens == 2048
    assert config.temperature == 0.2


async def test_complete_leaves_max_tokens_unset_when_none() -> None:
    """`complete` leaves `max_output_tokens` unset when not on settings.

    Gemini does not require an output-token cap, so - unlike Anthropic - there
    is no default fallback.
    """

    # GIVEN a mock client and settings without an explicit `max_tokens`
    mock_client = _mock_client_returning(_text("ok"))
    provider = _provider(mock_client)
    settings = _settings(max_tokens=None)

    # WHEN `complete` is invoked
    await provider.complete([UserMessage.from_text("hi")], settings)

    # THEN `max_output_tokens` stays `None` (no provider-side default applied)
    assert _call_config(mock_client).max_output_tokens is None


async def test_complete_leaves_temperature_unset_when_none() -> None:
    """`complete` leaves `temperature` unset when not on settings."""

    # GIVEN a mock client and settings without an explicit `temperature`
    mock_client = _mock_client_returning(_text("ok"))
    provider = _provider(mock_client)
    settings = _settings(temperature=None)

    # WHEN `complete` is invoked
    await provider.complete([UserMessage.from_text("hi")], settings)

    # THEN `temperature` stays `None`
    assert _call_config(mock_client).temperature is None


async def test_complete_maps_each_response_text_part_to_a_part() -> None:
    """`complete` maps each response text part to its own `TextPart`."""

    # GIVEN a response with two text parts
    response = _text("hello ", "world")
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the returned message has one `TextPart` per response part, in order
    assert result.message.parts == [TextPart(text="hello "), TextPart(text="world")]


async def test_complete_returns_empty_parts_when_candidate_content_is_empty() -> None:
    """`complete` returns `parts=[]` when the candidate's content has no
    parts.
    """

    # GIVEN a response whose single candidate carries content with zero parts
    response = _response()
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the result has an empty parts list (not a single empty `TextPart`)
    # and a normal stop reason
    assert result.message.parts == []
    assert result.message.stop_reason == "stop"


async def test_complete_maps_no_candidates_to_error() -> None:
    """A response with no candidates and no prompt block maps to `"error"`.

    An empty response carrying no candidate is abnormal, not a normal empty
    stop: reporting it as `"stop"` would hand back a successful empty answer and
    hide the anomaly.  (A prompt block is the separate `"content_filter"` case.)
    """

    # GIVEN a response with no candidates and no prompt-block feedback
    response = types.GenerateContentResponse(candidates=[])
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the stop reason is `"error"` (with no content)
    assert result.message.parts == []
    assert result.message.stop_reason == "error"


async def test_complete_decodes_thought_part_into_thinking_part() -> None:
    """`complete` decodes a thought-summary part into a `ThinkingPart`,
    keeping the answer text separate.
    """

    # GIVEN a response interleaving a thought part with the answer text
    response = _response(
        types.Part(text="thinking...", thought=True),
        types.Part(text="A"),
    )
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the thought becomes a `ThinkingPart` and the answer stays text
    assert result.message.parts == [
        ThinkingPart(content="thinking..."),
        TextPart(text="A"),
    ]


async def test_complete_maps_blocked_prompt_to_content_filter() -> None:
    """A prompt blocked before generation maps to `"content_filter"`.

    Gemini blocks an unsafe prompt by returning no candidate plus a
    `prompt_feedback.block_reason`; that must surface as a content-filter stop,
    not a successful empty answer.
    """

    # GIVEN a response with no candidates and a prompt-block reason
    response = types.GenerateContentResponse(
        candidates=[],
        prompt_feedback=types.GenerateContentResponsePromptFeedback(
            block_reason=types.BlockedReason.SAFETY
        ),
    )
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the stop reason is `"content_filter"` (with no content)
    assert result.message.parts == []
    assert result.message.stop_reason == "content_filter"


# Call-metadata mapping tests
# -----------------------------------------------------------------------------


async def test_complete_maps_usage_ids_and_model_onto_provider_response() -> None:
    """`complete` maps Gemini usage, response id, and model onto the wrapper."""

    # GIVEN a response whose usage itemizes thinking, tool-use, and cached
    # tokens (Gemini reports prompt, candidates, tool-use-prompt, and thoughts
    # as separate addends: 11 + 7 + 4 + 3 == 25)
    response = _text(
        "hi",
        usage=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=11,
            candidates_token_count=7,
            tool_use_prompt_token_count=4,
            thoughts_token_count=3,
            cached_content_token_count=5,
            total_token_count=25,
        ),
    )
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN usage is normalized: input folds prompt + tool-use-prompt (11 + 4 =
    # 15; cache is a subset of prompt), thinking folds into output (7 + 3 = 10)
    # and is itemized as reasoning; cache reads kept, no cache write; derived
    # total matches Gemini's (15 + 10 == 25)
    assert result.usage is not None
    assert result.usage.input_tokens == 15
    assert result.usage.output_tokens == 10
    assert result.usage.reasoning_tokens == 3
    assert result.usage.cache_read_tokens == 5
    assert result.usage.cache_write_tokens == 0
    assert result.usage.total_tokens == 25

    # AND the provider-native usage is preserved beside the normalized counts
    assert result.raw_usage is not None
    assert result.raw_usage["prompt_token_count"] == 11

    # AND the response id, served model, and provider name are populated
    assert result.response_id == "resp_test"
    assert result.model == "gemini-test"
    assert result.provider_name == "gemini"


async def test_complete_maps_absent_usage_to_none() -> None:
    """`complete` returns `usage=None` when the response carries no metadata."""

    # GIVEN a response with no usage metadata
    response = _text("hi")
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN usage and raw usage are both `None`
    assert result.usage is None
    assert result.raw_usage is None


async def test_complete_warns_when_provider_total_diverges_from_derived(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A `total_token_count` disagreeing with the derived total logs a warning.

    Guards against Google silently changing which sub-counts the total includes
    (mirrors the OpenAI provider's divergence check).
    """

    # GIVEN a response whose reported total disagrees with the component sum
    response = _text(
        "hi",
        usage=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=999,  # != derived 10 + 5
        ),
    )
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    with caplog.at_level(logging.WARNING, logger="avior.providers.gemini"):
        result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN avior reports the derived total and logs the divergence
    assert result.usage is not None
    assert result.usage.total_tokens == 15
    assert "total_token_count" in caplog.text


# Exception translation tests
# -----------------------------------------------------------------------------


async def test_complete_translates_api_error_to_http_error() -> None:
    """`errors.APIError` becomes `ProviderHTTPError`, preserving status."""

    # GIVEN a mock client raising an `APIError` carrying HTTP status 429
    gemini_error = errors.APIError(429, {"error": {"message": "rate limit hit"}})
    mock_client = _mock_client_raising(gemini_error)
    provider = _provider(mock_client)

    # WHEN `complete` is invoked
    # THEN `ProviderHTTPError` is raised with the HTTP status, and the original
    # exception is preserved as `__cause__`
    with pytest.raises(ProviderHTTPError) as exc_info:
        await provider.complete([UserMessage.from_text("hi")], _settings())
    assert exc_info.value.status_code == 429
    assert exc_info.value.__cause__ is gemini_error


async def test_complete_translates_unknown_response_error() -> None:
    """`UnknownApiResponseError` maps to the avior counterpart."""

    # GIVEN a mock client raising `UnknownApiResponseError` (the Gemini SDK
    # could not decode an otherwise-successful response)
    gemini_error = errors.UnknownApiResponseError("schema mismatch")
    provider = _provider(_mock_client_raising(gemini_error))

    # WHEN `complete` is invoked
    # THEN `ProviderResponseValidationError` is raised, preserving `__cause__`
    with pytest.raises(ProviderResponseValidationError) as exc_info:
        await provider.complete([UserMessage.from_text("hi")], _settings())
    assert exc_info.value.__cause__ is gemini_error


async def test_complete_translates_connection_error() -> None:
    """An httpx transport failure becomes `ProviderConnectionError`."""

    # GIVEN a mock client raising an httpx connect error (network failed before
    # an HTTP response was received)
    transport_error = httpx.ConnectError("connection refused")
    mock_client = _mock_client_raising(transport_error)
    provider = _provider(mock_client)

    # WHEN `complete` is invoked
    # THEN `ProviderConnectionError` is raised with the original `__cause__`
    with pytest.raises(ProviderConnectionError) as exc_info:
        await provider.complete([UserMessage.from_text("hi")], _settings())
    assert exc_info.value.__cause__ is transport_error


async def test_complete_translates_timeout_as_connection_error() -> None:
    """An httpx timeout maps to `ProviderConnectionError` via subclass."""

    # GIVEN a mock client raising an httpx timeout (subclass of
    # `httpx.TransportError`)
    transport_error = httpx.ReadTimeout("timed out")
    mock_client = _mock_client_raising(transport_error)
    provider = _provider(mock_client)

    # WHEN `complete` is invoked
    # THEN `ProviderConnectionError` is raised (timeouts surface as
    # connection-level failures)
    with pytest.raises(ProviderConnectionError) as exc_info:
        await provider.complete([UserMessage.from_text("hi")], _settings())
    assert exc_info.value.__cause__ is transport_error


# Stop-reason mapping tests
# -----------------------------------------------------------------------------


async def test_complete_sets_stop_reason_stop_on_stop_finish() -> None:
    """Gemini `finish_reason=STOP` maps to canonical `"stop"`."""

    # GIVEN a response that finished on a normal stop
    response = _text("done")
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the canonical `stop_reason` is `"stop"`
    assert result.message.stop_reason == "stop"


async def test_complete_maps_max_tokens_finish_to_max_tokens() -> None:
    """Gemini `finish_reason=MAX_TOKENS` maps to canonical `"max_tokens"`."""

    # GIVEN a mock client returning a max-tokens-truncated response
    response = _response(
        types.Part(text="..."), finish_reason=types.FinishReason.MAX_TOKENS
    )
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the canonical `stop_reason` is `"max_tokens"`
    assert result.message.stop_reason == "max_tokens"


async def test_complete_maps_safety_finish_to_content_filter() -> None:
    """Gemini `finish_reason=SAFETY` maps to canonical `"content_filter"`."""

    # GIVEN a response blocked by Gemini's safety filter
    response = _response(types.Part(text=""), finish_reason=types.FinishReason.SAFETY)
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the canonical `stop_reason` is `"content_filter"`
    assert result.message.stop_reason == "content_filter"


async def test_complete_maps_image_safety_finish_to_content_filter() -> None:
    """An `IMAGE_*` safety finish also maps to `"content_filter"`."""

    # GIVEN a response blocked by an image-safety finish reason
    response = _response(
        types.Part(text=""), finish_reason=types.FinishReason.IMAGE_SAFETY
    )
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the canonical `stop_reason` is `"content_filter"`
    assert result.message.stop_reason == "content_filter"


async def test_complete_maps_language_finish_to_error() -> None:
    """A `LANGUAGE` finish (unsupported language) maps to `"error"`."""

    # GIVEN a response terminated for an unsupported language
    response = _response(finish_reason=types.FinishReason.LANGUAGE)
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the canonical `stop_reason` is `"error"`, not a successful `"stop"`
    assert result.message.stop_reason == "error"


async def test_complete_maps_malformed_function_call_to_error() -> None:
    """`MALFORMED_FUNCTION_CALL` with no usable call maps to `"error"`."""

    # GIVEN a response that finished on a malformed function call, with no parts
    response = _response(finish_reason=types.FinishReason.MALFORMED_FUNCTION_CALL)
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the canonical `stop_reason` is `"error"`, not a successful `"stop"`
    assert result.message.stop_reason == "error"


async def test_complete_maps_unexpected_tool_call_to_error() -> None:
    """`UNEXPECTED_TOOL_CALL` with no usable call maps to `"error"`."""

    # GIVEN a response that finished on an unexpected tool call, with no parts
    response = _response(finish_reason=types.FinishReason.UNEXPECTED_TOOL_CALL)
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the canonical `stop_reason` is `"error"`
    assert result.message.stop_reason == "error"


async def test_complete_maps_other_finish_to_error() -> None:
    """An `OTHER` finish maps to `"error"`.

    `OTHER` (with `FINISH_REASON_UNSPECIFIED` and `IMAGE_OTHER`) is an abnormal
    or unspecified termination, so it must not pass as a successful empty
    response.
    """

    # GIVEN a response that finished on `OTHER`
    response = _response(finish_reason=types.FinishReason.OTHER)
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the canonical `stop_reason` is `"error"`
    assert result.message.stop_reason == "error"


@pytest.mark.parametrize(
    ("finish_reason", "expected_stop_reason"),
    [
        (types.FinishReason.MAX_TOKENS, "max_tokens"),
        (types.FinishReason.SAFETY, "content_filter"),
        (types.FinishReason.MALFORMED_FUNCTION_CALL, "error"),
        (types.FinishReason.UNEXPECTED_TOOL_CALL, "error"),
    ],
    ids=["max_tokens", "safety", "malformed_call", "unexpected_call"],
)
async def test_complete_prefers_terminal_finish_over_inferred_tool_use(
    finish_reason: types.FinishReason,
    expected_stop_reason: StopReason,
) -> None:
    """A terminal finish reason wins over a tool call inferred from a
    `function_call` part.

    Gemini infers `"tool_use"` from the presence of a `function_call`, but a
    truncated, blocked, or malformed finish must not be reported as a tool
    request and executed.
    """

    # GIVEN a response carrying a function call but a terminal finish reason
    call = types.Part(
        function_call=types.FunctionCall(
            id="call_1",
            name="get_weather",
            args={"city": "Paris"},
        )
    )
    response = _response(call, finish_reason=finish_reason)
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("weather?")], _settings())

    # THEN the terminal reason wins; the call is not reported as `"tool_use"`
    assert result.message.stop_reason == expected_stop_reason


async def test_complete_maps_terminal_finish_with_nameless_call_to_terminal() -> None:
    """A nameless call on a terminal finish yields the terminal stop, not an
    error.

    A `MAX_TOKENS` finish can truncate a tool call mid-stream, leaving it
    without a name.  That partial call is an artifact of the truncation, so it
    is dropped and the finish surfaces as `"max_tokens"` - it must not raise a
    validation error as a nameless call on a normal finish would.
    """

    # GIVEN a max-tokens-truncated response carrying a nameless function call
    call = types.Part(function_call=types.FunctionCall(args={"city": "Paris"}))
    response = _response(call, finish_reason=types.FinishReason.MAX_TOKENS)
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("weather?")], _settings())

    # THEN the truncation surfaces as `"max_tokens"` and the partial call is
    # dropped rather than raising
    assert result.message.stop_reason == "max_tokens"
    assert result.message.parts == []


# Tool-calling tests
# -----------------------------------------------------------------------------


async def test_complete_parses_function_call_into_tool_call_part() -> None:
    """A response `function_call` part decodes into a `ToolCallPart`."""

    # GIVEN a function-call response whose call carries an id, name, and args
    call = types.Part(
        function_call=types.FunctionCall(
            id="call_1",
            name="get_weather",
            args={"city": "Paris"},
        )
    )
    provider = _provider(_mock_client_returning(_response(call)))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("weather?")], _settings())

    # THEN the part decodes into a `ToolCallPart` with its id, name, and args
    assert result.message.parts == [
        ToolCallPart(
            call_id="call_1",
            tool_name="get_weather",
            args={"city": "Paris"},
        )
    ]


async def test_complete_parses_empty_arguments_into_empty_dict() -> None:
    """A `function_call` with no arguments decodes into an empty args dict."""

    # GIVEN a function-call response whose call carries no arguments
    call = types.Part(
        function_call=types.FunctionCall(
            id="call_1",
            name="ping",
            args=None,
        )
    )
    provider = _provider(_mock_client_returning(_response(call)))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("ping")], _settings())

    # THEN the call decodes with empty args rather than `None`
    assert result.message.parts == [
        ToolCallPart(
            call_id="call_1",
            tool_name="ping",
            args={},
        )
    ]


async def test_complete_maps_function_call_to_tool_use_stop_reason() -> None:
    """A response carrying a `function_call` maps to canonical `"tool_use"`.

    Gemini has no tool-use finish reason; the request is detected from the
    presence of a `function_call` part.
    """

    # GIVEN a function-call response
    call = types.Part(
        function_call=types.FunctionCall(
            id="call_1",
            name="get_weather",
            args={"city": "Paris"},
        )
    )
    provider = _provider(_mock_client_returning(_response(call)))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("weather?")], _settings())

    # THEN the canonical `stop_reason` is `"tool_use"`
    assert result.message.stop_reason == "tool_use"


async def test_complete_synthesizes_call_id_when_function_call_has_no_id() -> None:
    """A `function_call` without an id gets a synthesized non-empty id.

    Gemini omits the id for some calls; the synthesized id keeps the call
    correlated with its result.
    """

    # GIVEN a function-call response whose call carries no id
    call = types.Part(
        function_call=types.FunctionCall(name="get_weather", args={"city": "Paris"})
    )
    provider = _provider(_mock_client_returning(_response(call)))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("weather?")], _settings())

    # THEN the call decodes with a synthesized, non-empty `call_id`
    assert len(result.message.parts) == 1
    part = result.message.parts[0]
    assert isinstance(part, ToolCallPart)
    assert part.tool_name == "get_weather"
    assert part.args == {"city": "Paris"}
    assert part.call_id


async def test_complete_synthesizes_distinct_ids_for_parallel_same_name_calls() -> None:
    """Two id-less calls to the same tool get distinct synthesized `call_id`s.

    Gemini can return parallel calls to one tool that omit an id; each must get
    its own id so results stay correlated to the right call.
    """

    # GIVEN two id-less function calls to the same tool in one response
    call_a = types.Part(
        function_call=types.FunctionCall(name="get_weather", args={"city": "Paris"})
    )
    call_b = types.Part(
        function_call=types.FunctionCall(name="get_weather", args={"city": "Berlin"})
    )
    provider = _provider(_mock_client_returning(_response(call_a, call_b)))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("weather?")], _settings())

    # THEN the two calls carry distinct ids
    call_ids = [p.call_id for p in result.message.parts if isinstance(p, ToolCallPart)]
    assert len(call_ids) == 2
    assert len(set(call_ids)) == 2


async def test_complete_disambiguates_duplicate_provider_call_ids() -> None:
    """Two function calls sharing the same `id` get distinct `call_id`s.

    Gemini can repeat an id across parallel calls; keeping both would collide
    and break call-to-result correlation, so the duplicate is replaced with a
    freshly minted id.
    """

    # GIVEN two function calls that share the same `id`
    call_a = types.Part(
        function_call=types.FunctionCall(
            id="dup",
            name="get_weather",
            args={"city": "Paris"},
        )
    )
    call_b = types.Part(
        function_call=types.FunctionCall(
            id="dup",
            name="get_weather",
            args={"city": "Berlin"},
        )
    )
    provider = _provider(_mock_client_returning(_response(call_a, call_b)))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("weather?")], _settings())

    # THEN the first keeps the id, the second is disambiguated, and they differ
    call_ids = [p.call_id for p in result.message.parts if isinstance(p, ToolCallPart)]
    assert call_ids[0] == "dup"
    assert len(set(call_ids)) == 2


async def test_complete_synthesizes_id_when_provider_reuses_one_across_turns() -> None:
    """A function-call `id` reused in a later turn gets a fresh, distinct id.

    When the model reuses a call id it already used earlier in the transcript,
    the later call is given a different id rather than the repeated one, so each
    call stays uniquely identified and a tool result cannot correlate to the
    wrong call.
    """

    # GIVEN a mock client that returns a function call with the same `id` on
    # every turn
    call = types.Part(
        function_call=types.FunctionCall(
            id="dup", name="get_weather", args={"city": "Paris"}
        )
    )
    mock_client = _mock_client_returning(_response(call))
    provider = _provider(mock_client)

    # WHEN `complete` is awaited twice, as a two-turn tool loop would: first
    # on the lone user message, then on the continuation - that user message,
    # the first turn's assistant tool call (carrying id `"dup"`) fed back, and
    # its tool result - so the second turn returns another call reusing `"dup"`
    first = await provider.complete([UserMessage.from_text("weather?")], _settings())
    first_call = first.message.parts[0]
    assert isinstance(first_call, ToolCallPart)
    continuation: list[Message] = [
        UserMessage.from_text("weather?"),
        first.message,
        ToolMessage(
            parts=[
                ToolResultPart(
                    call_id=first_call.call_id, result=ToolResultOk(content="sunny")
                )
            ]
        ),
    ]
    second = await provider.complete(continuation, _settings())
    second_call = second.message.parts[0]
    assert isinstance(second_call, ToolCallPart)

    # THEN the first call keeps its `id`, but the later turn's reuse of it is
    # replaced with a fresh, distinct id
    assert first_call.call_id == "dup"
    assert second_call.call_id != "dup"
    assert second_call.call_id


async def test_complete_maps_nameless_function_call_to_error() -> None:
    """A nameless `function_call` on a normal finish maps to `"error"`.

    The Gemini SDK decoded the response fine, but a tool call with no name is
    malformed tool-call data from the model - the same class as
    `MALFORMED_FUNCTION_CALL` - so it surfaces as the canonical `"error"` stop
    reason (a model failure), not a provider decode error.  The unusable call is
    dropped.
    """

    # GIVEN a `STOP`-finish response whose only call has no name
    call = types.Part(function_call=types.FunctionCall(args={"city": "Paris"}))
    provider = _provider(_mock_client_returning(_response(call)))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("weather?")], _settings())

    # THEN the stop reason is `"error"` and the nameless call is dropped
    assert result.message.stop_reason == "error"
    assert result.message.parts == []


async def test_complete_raises_on_unsupported_content_part() -> None:
    """A content part the adapter cannot represent raises rather than dropping.

    A part that is neither text, a function call, nor a thought (here an
    `inline_data` part) carries content avior's IR has no slot for.  Silently
    skipping it would return a misleadingly-successful response with the content
    lost, so it fails loud as `ProviderResponseValidationError`.
    """

    # GIVEN a response carrying an `inline_data` part the adapter does not map
    part = types.Part(inline_data=types.Blob(mime_type="image/png", data=b"x"))
    provider = _provider(_mock_client_returning(_response(part)))

    # WHEN `complete` is awaited
    # THEN `ProviderResponseValidationError` is raised
    with pytest.raises(ProviderResponseValidationError):
        await provider.complete([UserMessage.from_text("hi")], _settings())


async def test_complete_sends_tools_with_name_description_and_schema() -> None:
    """Each offered tool is sent as a function declaration with its schema."""

    # GIVEN a mock client and an offered tool
    mock_client = _mock_client_returning(_text("ok"))
    provider = _provider(mock_client)
    tool = _Weather()

    # WHEN `complete` is invoked with that tool
    await provider.complete(
        [UserMessage.from_text("hi")],
        _settings(),
        tools=[tool],
    )

    # THEN the request's `GenerateContentConfig` carries one function
    # declaration with the tool's name, description, and arguments JSON schema
    # (sent as `parameters_json_schema`)
    tools = _call_config_dump(mock_client, exclude_none=True)["tools"]
    assert tools is not None
    assert len(tools) == 1
    assert tools[0]["function_declarations"] == [
        {
            "name": "get_weather",
            "description": "Look up the weather for a city.",
            "parameters_json_schema": _CityArgs.model_json_schema(),
        }
    ]


async def test_complete_omits_tools_when_none_offered() -> None:
    """No offered tools means `config.tools` stays unset."""

    # GIVEN a mock client and no tools offered
    mock_client = _mock_client_returning(_text("ok"))
    provider = _provider(mock_client)

    # WHEN `complete` is invoked without tools
    await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the request's `GenerateContentConfig` carries no tools rather than an
    # empty list
    assert _call_config_dump(mock_client)["tools"] is None


async def test_complete_sends_assistant_tool_call_as_function_call_part() -> None:
    """An assistant `ToolCallPart` becomes a `function_call` part."""

    # GIVEN a continuation transcript: the assistant requested a tool call and
    # its result was supplied (a re-entry into `complete`)
    mock_client = _mock_client_returning(_text("ok"))
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

    # THEN the assistant turn is sent as a `"model"` content with a matching
    # `function_call` part
    contents = _call_contents(mock_client)
    model_turn = next(c for c in contents if c.role == "model")
    assert model_turn.parts is not None
    call = model_turn.parts[0].function_call
    assert call is not None
    assert call.id == "call_1"
    assert call.name == "get_weather"
    assert call.args == {"city": "Paris"}


async def test_complete_sends_tool_message_as_user_function_responses() -> None:
    """A `ToolMessage` becomes a `"user"` turn of `function_response` parts.

    The tool name (which `ToolResultPart` does not store) is recovered from the
    matching tool call, and the result rides under Gemini's `"output"` /
    `"error"` response key.
    """

    # GIVEN a continuation transcript whose assistant requested two tool calls,
    # now answered with one ok and one error result
    mock_client = _mock_client_returning(_text("ok"))
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

    # THEN the results travel in a `"user"` turn, each `function_response`
    # carrying the recovered name, the correlating id, and the wrapped payload
    contents = _call_contents(mock_client)
    tool_turn = contents[-1]
    assert tool_turn.role == "user"
    assert tool_turn.parts is not None

    ok_response = tool_turn.parts[0].function_response
    assert ok_response is not None
    assert ok_response.id == "ok_1"
    assert ok_response.name == "get_weather"
    assert ok_response.response == {"output": "sunny"}

    err_response = tool_turn.parts[1].function_response
    assert err_response is not None
    assert err_response.id == "err_1"
    assert err_response.name == "get_weather"
    assert err_response.response == {"error": "boom"}


# Capability tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gemini-2.5-flash", True),
        ("gemini-2.5-flash-preview-05-20", True),
        ("gemini-2.5-flash-lite", True),
        ("gemini-2.5-pro", True),
        ("gemini-3-flash-preview", True),
        ("gemini-3-pro-preview", True),
        ("gemini-3.1-flash-lite", True),
        ("gemini-3.1-pro-preview", True),
        ("gemini-3.5-flash", True),
        ("gemini-3.5-flash-lite", True),
        ("gemini-3.6-flash", True),
        ("gemini-flash-latest", True),
        ("gemini-flash-lite-latest", True),
        ("gemini-pro-latest", True),
        ("models/gemini-2.5-flash", True),
        ("gemini-2.5-flash-image", False),
        ("gemini-2.5-flash-preview-tts", False),
        ("gemini-2.0-flash", False),
        ("tunedModels/my-fine-tune", False),
        ("gemini-test", False),
        ("claude-haiku-4-5", False),
    ],
    ids=[
        "budget-family",
        "dated-preview-snapshot",
        "budget-lite-family",
        "budget-pro-family",
        "level-flash-3-family",
        "level-pro-3-family",
        "level-lite-31-family",
        "level-pro-31-family",
        "level-flash-35-family",
        "level-lite-35-family",
        "level-flash-36-family",
        "flash-alias",
        "flash-lite-alias",
        "pro-alias",
        "qualified-resource-name",
        "image-variant",
        "tts-variant",
        "non-thinking-family",
        "tuned-model",
        "unknown-model",
        "foreign-model",
    ],
)
def test_model_capabilities_reports_thinking_support(
    model: str, expected: bool
) -> None:
    """`model_capabilities` reports thinking support for every classified
    thinking family - by bare id or qualified resource name - and the
    conservative default otherwise, notably for a named variant (`-image`,
    `-tts`) that extends a classified family's id but is a different model.
    """

    # GIVEN a provider (its client is never used to read capabilities)
    provider = _provider(AsyncMock())

    # WHEN capabilities are read for the model
    capabilities = provider.model_capabilities(model)

    # THEN thinking support matches the classification
    assert capabilities.supports_thinking is expected


# Thinking config (send) tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "thinking", "expected"),
    [
        ("gemini-3.6-flash", "high", {"thinking_level": types.ThinkingLevel.HIGH}),
        ("gemini-3.6-flash", "low", {"thinking_level": types.ThinkingLevel.LOW}),
        ("gemini-3.6-flash", False, {"thinking_level": types.ThinkingLevel.MINIMAL}),
        (
            "gemini-3.1-flash-lite",
            False,
            {"thinking_level": types.ThinkingLevel.MINIMAL},
        ),
        ("gemini-3.6-flash", True, None),
        (
            "gemini-3.1-flash-lite",
            True,
            {"thinking_level": types.ThinkingLevel.MEDIUM},
        ),
        (
            "gemini-3-flash-preview",
            "medium",
            {"thinking_level": types.ThinkingLevel.MEDIUM},
        ),
        (
            "gemini-3.1-pro-preview",
            "low",
            {"thinking_level": types.ThinkingLevel.LOW},
        ),
        ("gemini-3-pro-preview", True, None),
        ("gemini-2.5-flash", "low", {"thinking_budget": 2048}),
        ("gemini-2.5-flash", "medium", {"thinking_budget": 8192}),
        ("gemini-2.5-flash", False, {"thinking_budget": 0}),
        ("gemini-2.5-flash-lite", False, {"thinking_budget": 0}),
        ("gemini-2.5-flash", True, None),
        ("gemini-2.5-flash-lite", True, {"thinking_budget": -1}),
        ("gemini-2.5-pro", "high", {"thinking_budget": 24576}),
        ("gemini-2.5-pro", True, None),
        ("gemini-2.5-flash", None, None),
    ],
    ids=[
        "level-high",
        "level-low",
        "level-disable-on-by-default",
        "level-disable-off-by-default",
        "level-true-on-by-default-omits",
        "level-true-off-by-default-medium",
        "level-preview-tail-resolves",
        "level-low-on-always-on",
        "level-true-always-on-omits",
        "budget-low",
        "budget-medium",
        "budget-disable-on-by-default",
        "budget-disable-off-by-default",
        "budget-true-on-by-default-omits",
        "budget-true-off-by-default-dynamic",
        "budget-high-on-always-on",
        "budget-true-always-on-omits",
        "unset-omits",
    ],
)
async def test_complete_maps_thinking_to_native_config(
    model: str,
    thinking: bool | Literal["low", "medium", "high"] | None,
    expected: dict[str, Any] | None,
) -> None:
    """`complete` maps the portable `thinking` setting to the model's native
    `thinking_config` dialect: `thinking_budget` tokens on a Gemini 2.5 model,
    `thinking_level` on a Gemini 3+ model, omitted when `thinking` is unset or
    `True` asks for what the model's default already does.
    """

    # GIVEN settings carrying a portable thinking value for the model
    mock_client = _mock_client_returning(_text("ok"))
    provider = _provider(mock_client)
    settings = _settings(model=model, thinking=thinking)

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], settings)

    # THEN the native config matches the expected dialect, and nothing warns
    config_dump = _call_config_dump(mock_client, exclude_none=True)
    assert config_dump.get("thinking_config") == expected
    assert result.warnings == []


@pytest.mark.parametrize(
    ("model", "thinking", "reason"),
    [
        (
            "gemini-2.0-flash",
            "high",
            "the model is not a recognized thinking model; if it does think, "
            "configure thinking via the `gemini` provider options",
        ),
        (
            "gemini-test",
            True,
            "the model is not a recognized thinking model; if it does think, "
            "configure thinking via the `gemini` provider options",
        ),
        (
            "gemini-2.5-pro",
            False,
            "the model's thinking is always on and cannot be disabled",
        ),
        (
            "gemini-3-pro-preview",
            False,
            "the model's thinking is always on and cannot be disabled",
        ),
    ],
    ids=[
        "enable-on-non-thinking-family",
        "enable-on-unknown-model",
        "disable-on-always-on-budget",
        "disable-on-always-on-level",
    ],
)
async def test_complete_warns_and_drops_unhonorable_thinking(
    model: str,
    thinking: bool | Literal["low", "medium", "high"],
    reason: str,
) -> None:
    """`complete` drops a `thinking` request the model cannot honor, sends no
    `thinking_config`, and records a warning naming the cause.
    """

    # GIVEN settings with a thinking request the model cannot honor
    mock_client = _mock_client_returning(_text("ok"))
    provider = _provider(mock_client)
    settings = _settings(model=model, thinking=thinking)

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], settings)

    # THEN no thinking config is sent, and a warning records the drop
    config_dump = _call_config_dump(mock_client, exclude_none=True)
    assert "thinking_config" not in config_dump
    thinking_warnings = [w for w in result.warnings if w.setting_name == "thinking"]
    assert len(thinking_warnings) == 1
    assert thinking_warnings[0].setting_value == thinking
    assert thinking_warnings[0].reason == reason


async def test_complete_ignores_disable_on_unrecognized_model() -> None:
    """`thinking=False` on an unrecognized model is a no-op: the model is not
    classified as thinking, so there is nothing to disable and no warning.
    """

    # GIVEN settings disabling thinking on an unrecognized model
    mock_client = _mock_client_returning(_text("ok"))
    provider = _provider(mock_client)
    settings = _settings(model="gemini-test", thinking=False)

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], settings)

    # THEN no thinking config is sent and nothing warns
    assert "thinking_config" not in _call_config_dump(mock_client, exclude_none=True)
    assert result.warnings == []


async def test_complete_sends_raw_thinking_config_on_unclassified_model() -> None:
    """A raw `thinking_config` in the `gemini` provider options is passed
    through even for a model avior does not classify.
    """

    # GIVEN settings whose provider options carry a raw thinking config for an
    # unclassified model
    mock_client = _mock_client_returning(_text("ok"))
    provider = _provider(mock_client)
    settings = _settings(
        model="gemini-test",
        provider_options={
            "gemini": {
                "thinking_config": {"thinking_budget": 512, "include_thoughts": True}
            }
        },
    )

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], settings)

    # THEN the raw config is sent as given, and nothing warns
    config_dump = _call_config_dump(mock_client, exclude_none=True)
    assert config_dump.get("thinking_config") == {
        "thinking_budget": 512,
        "include_thoughts": True,
    }
    assert result.warnings == []


async def test_complete_raw_thinking_config_overrides_portable_thinking() -> None:
    """A raw `thinking_config` takes precedence over the portable `thinking`
    setting.
    """

    # GIVEN settings carrying both a portable thinking level and a raw config
    mock_client = _mock_client_returning(_text("ok"))
    provider = _provider(mock_client)
    settings = _settings(
        model="gemini-2.5-flash",
        thinking="high",
        provider_options={"gemini": {"thinking_config": {"thinking_budget": 128}}},
    )

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], settings)

    # THEN the raw config wins over the portable mapping
    config_dump = _call_config_dump(mock_client, exclude_none=True)
    assert config_dump.get("thinking_config") == {"thinking_budget": 128}
    assert result.warnings == []


async def test_complete_rejects_unknown_provider_options_key() -> None:
    """An unknown key in the `gemini` provider options slice raises
    `AviorUsageError` before any request is sent.
    """

    # GIVEN settings whose `gemini` provider options carry an unknown key
    mock_client = _mock_client_returning(_text("ok"))
    provider = _provider(mock_client)
    settings = _settings(
        model="gemini-2.5-flash",
        provider_options={"gemini": {"thinking": {"thinking_budget": 128}}},
    )

    # WHEN `complete` is invoked
    # THEN the invalid slice is rejected and no request is sent
    with pytest.raises(AviorUsageError, match="provider_options"):
        await provider.complete([UserMessage.from_text("hi")], settings)
    mock_client.aio.models.generate_content.assert_not_called()


# Thinking round-trip tests
# -----------------------------------------------------------------------------


_SIGNATURE = b"\x00\x01binary-signature\xff"
"""A signature with non-UTF-8 bytes, as the Gemini API may produce."""

_SIGNATURE_B64 = base64.b64encode(_SIGNATURE).decode("ascii")
"""The base64 text form `provider_details` stores for `_SIGNATURE`."""


async def test_complete_keeps_signature_from_function_call_part() -> None:
    """`complete` stores a function-call part's thought signature
    base64-encoded in the `ToolCallPart`'s `provider_details`.
    """

    # GIVEN a response whose function-call part carries a thought signature
    response = _response(
        types.Part(
            function_call=types.FunctionCall(
                id="call_1",
                name="get_weather",
                args={"city": "Paris"},
            ),
            thought_signature=_SIGNATURE,
        )
    )
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the tool call keeps the signature, base64-encoded
    assert result.message.parts == [
        ToolCallPart(
            call_id="call_1",
            tool_name="get_weather",
            args={"city": "Paris"},
            provider_details={"thought_signature": _SIGNATURE_B64},
        )
    ]


async def test_complete_keeps_signature_from_text_part() -> None:
    """`complete` stores a text part's thought signature base64-encoded in the
    `TextPart`'s `provider_details`.
    """

    # GIVEN a response whose text part carries a thought signature
    response = _response(types.Part(text="391", thought_signature=_SIGNATURE))
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the text part keeps the signature, base64-encoded
    assert result.message.parts == [
        TextPart(
            text="391",
            provider_details={"thought_signature": _SIGNATURE_B64},
        )
    ]


async def test_complete_keeps_signature_from_thought_part() -> None:
    """`complete` stores a thought part's thought signature base64-encoded in
    the `ThinkingPart`'s `provider_details`.
    """

    # GIVEN a response whose thought part carries a thought signature
    response = _response(
        types.Part(text="thinking...", thought=True, thought_signature=_SIGNATURE),
        types.Part(text="A"),
    )
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the thinking part keeps the signature, base64-encoded
    assert result.message.parts == [
        ThinkingPart(
            content="thinking...",
            provider_details={"thought_signature": _SIGNATURE_B64},
        ),
        TextPart(text="A"),
    ]


async def test_complete_echoes_own_signatures_verbatim() -> None:
    """`complete` echoes this provider's thought signatures back on their
    original parts, decoded to the exact bytes the Gemini API produced.
    """

    # GIVEN a continuation transcript whose Gemini turn carries a signed
    # thinking part, a signed text part, and a signed tool call
    mock_client = _mock_client_returning(_text("ok"))
    provider = _provider(mock_client)
    history: list[Message] = [
        UserMessage.from_text("weather?"),
        AssistantMessage(
            parts=[
                ThinkingPart(
                    content="hmm",
                    provider_details={"thought_signature": _SIGNATURE_B64},
                ),
                TextPart(
                    text="Checking.",
                    provider_details={"thought_signature": _SIGNATURE_B64},
                ),
                ToolCallPart(
                    call_id="call_1",
                    tool_name="get_weather",
                    args={"city": "Paris"},
                    provider_details={"thought_signature": _SIGNATURE_B64},
                ),
            ],
            stop_reason="tool_use",
            provider_name="gemini",
        ),
        ToolMessage(
            parts=[
                ToolResultPart(call_id="call_1", result=ToolResultOk(content="sunny"))
            ]
        ),
    ]

    # WHEN `complete` is invoked
    await provider.complete(history, _settings())

    # THEN each wire part carries the original signature bytes, and the
    # thinking part is echoed as a thought part
    model_turn = _call_contents(mock_client)[1]
    assert model_turn.parts is not None
    thought_part, text_part, call_part = model_turn.parts
    assert thought_part.thought is True
    assert thought_part.text == "hmm"
    assert thought_part.thought_signature == _SIGNATURE
    assert text_part.text == "Checking."
    assert text_part.thought_signature == _SIGNATURE
    assert call_part.function_call is not None
    assert call_part.thought_signature == _SIGNATURE


async def test_complete_serialized_transcript_round_trips_signature() -> None:
    """A binary signature survives the full round trip: decoded from a
    response, serialized to JSON, loaded back, and echoed with the exact
    bytes the Gemini API produced.
    """

    # GIVEN an assistant turn decoded from a response with a signed call
    decode_client = _mock_client_returning(
        _response(
            types.Part(
                function_call=types.FunctionCall(
                    id="call_1", name="get_weather", args={"city": "Paris"}
                ),
                thought_signature=_SIGNATURE,
            ),
        )
    )
    decode_provider = _provider(decode_client)
    decoded = await decode_provider.complete(
        [UserMessage.from_text("weather?")], _settings()
    )

    # AND that turn serialized to JSON and restored
    restored = AssistantMessage.model_validate_json(decoded.message.model_dump_json())

    # WHEN a continuation carrying the restored turn is completed
    echo_client = _mock_client_returning(_text("ok"))
    echo_provider = _provider(echo_client)
    history: list[Message] = [
        UserMessage.from_text("weather?"),
        restored,
        ToolMessage(
            parts=[
                ToolResultPart(call_id="call_1", result=ToolResultOk(content="sunny"))
            ]
        ),
    ]
    await echo_provider.complete(history, _settings())

    # THEN the echoed wire part carries the original signature bytes
    model_turn = _call_contents(echo_client)[1]
    assert model_turn.parts is not None
    assert model_turn.parts[0].thought_signature == _SIGNATURE


async def test_complete_treats_empty_signature_bytes_as_absent() -> None:
    """A response part whose `thought_signature` is empty bytes decodes into
    a part with no `provider_details`: there is no signature to carry.
    """

    # GIVEN a response whose function-call part carries empty signature bytes
    response = _response(
        types.Part(
            function_call=types.FunctionCall(
                id="call_1", name="get_weather", args={"city": "Paris"}
            ),
            thought_signature=b"",
        )
    )
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN the tool call carries no provider details
    assert result.message.parts == [
        ToolCallPart(call_id="call_1", tool_name="get_weather", args={"city": "Paris"})
    ]


@pytest.mark.parametrize(
    "provider_name",
    ["openai", None],
    ids=["foreign-provider", "hand-built"],
)
async def test_complete_stamps_skip_placeholder_on_unsigned_first_call(
    provider_name: str | None,
) -> None:
    """A turn whose first `function_call` has no signature to echo gets the
    signature-skip placeholder, and later calls stay unsigned.

    A foreign or hand-built turn carries no Gemini signature, yet thinking
    Gemini models reject a replayed turn whose first `function_call` part has
    none - the placeholder is what keeps such transcripts replayable.
    """

    # GIVEN a continuation transcript whose tool-calling turn is not
    # signature-bearing (foreign or hand-built, parametrized)
    mock_client = _mock_client_returning(_text("ok"))
    provider = _provider(mock_client)
    history: list[Message] = [
        UserMessage.from_text("weather?"),
        AssistantMessage(
            parts=[
                ToolCallPart(call_id="a_1", tool_name="get_weather", args={}),
                ToolCallPart(call_id="a_2", tool_name="get_weather", args={}),
            ],
            stop_reason="tool_use",
            provider_name=provider_name,
        ),
        ToolMessage(
            parts=[
                ToolResultPart(call_id="a_1", result=ToolResultOk(content="sunny")),
                ToolResultPart(call_id="a_2", result=ToolResultOk(content="rainy")),
            ]
        ),
    ]

    # WHEN `complete` is invoked
    await provider.complete(history, _settings())

    # THEN the first call carries the placeholder and the second none
    model_turn = _call_contents(mock_client)[1]
    assert model_turn.parts is not None
    first, second = model_turn.parts
    assert first.thought_signature == b"skip_thought_signature_validator"
    assert second.thought_signature is None


async def test_complete_drops_foreign_signature_and_stamps_placeholder() -> None:
    """A foreign turn's `provider_details` are never echoed: the foreign
    token is dropped and the first `function_call` gets the placeholder.
    """

    # GIVEN a continuation transcript whose signed tool-call turn came from a
    # different provider
    mock_client = _mock_client_returning(_text("ok"))
    provider = _provider(mock_client)
    history: list[Message] = [
        UserMessage.from_text("weather?"),
        AssistantMessage(
            parts=[
                ToolCallPart(
                    call_id="call_1",
                    tool_name="get_weather",
                    args={},
                    provider_details={"thought_signature": _SIGNATURE_B64},
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

    # WHEN `complete` is invoked
    await provider.complete(history, _settings())

    # THEN the foreign token is not sent; the placeholder stands in
    model_turn = _call_contents(mock_client)[1]
    assert model_turn.parts is not None
    assert model_turn.parts[0].thought_signature == (
        b"skip_thought_signature_validator"
    )


async def test_complete_stamps_placeholder_on_own_unsigned_first_call() -> None:
    """An own-provider turn whose first `function_call` carries no signature
    gets the placeholder: a Gemini model with thinking disabled signs
    nothing, and its replay must still pass a validating model.
    """

    # GIVEN a continuation transcript whose Gemini turn has an unsigned call
    mock_client = _mock_client_returning(_text("ok"))
    provider = _provider(mock_client)
    history: list[Message] = [
        UserMessage.from_text("weather?"),
        AssistantMessage(
            parts=[ToolCallPart(call_id="call_1", tool_name="get_weather", args={})],
            stop_reason="tool_use",
            provider_name="gemini",
        ),
        ToolMessage(
            parts=[
                ToolResultPart(call_id="call_1", result=ToolResultOk(content="sunny"))
            ]
        ),
    ]

    # WHEN `complete` is invoked
    await provider.complete(history, _settings())

    # THEN the unsigned call carries the placeholder
    model_turn = _call_contents(mock_client)[1]
    assert model_turn.parts is not None
    assert model_turn.parts[0].thought_signature == (
        b"skip_thought_signature_validator"
    )


async def test_complete_stamps_placeholder_on_each_unsigned_turn() -> None:
    """Placeholder stamping is per turn: in a multi-step chain, every model
    turn's first unsigned `function_call` gets the placeholder.

    The Gemini API validates each replayed model turn independently, so a
    placeholder on only the newest turn would still leave older turns
    rejected.
    """

    # GIVEN a two-step tool-chain transcript with unsigned tool calls
    mock_client = _mock_client_returning(_text("ok"))
    provider = _provider(mock_client)
    history: list[Message] = [
        UserMessage.from_text("weather?"),
        AssistantMessage(
            parts=[ToolCallPart(call_id="a_1", tool_name="get_weather", args={})],
            stop_reason="tool_use",
            provider_name="gemini",
        ),
        ToolMessage(
            parts=[ToolResultPart(call_id="a_1", result=ToolResultOk(content="sunny"))]
        ),
        AssistantMessage(
            parts=[ToolCallPart(call_id="b_1", tool_name="get_weather", args={})],
            stop_reason="tool_use",
            provider_name="gemini",
        ),
        ToolMessage(
            parts=[ToolResultPart(call_id="b_1", result=ToolResultOk(content="rainy"))]
        ),
    ]

    # WHEN `complete` is invoked
    await provider.complete(history, _settings())

    # THEN each model turn's call carries the placeholder
    contents = _call_contents(mock_client)
    first_turn, second_turn = contents[1], contents[3]
    assert first_turn.parts is not None and second_turn.parts is not None
    assert first_turn.parts[0].thought_signature == (
        b"skip_thought_signature_validator"
    )
    assert second_turn.parts[0].thought_signature == (
        b"skip_thought_signature_validator"
    )


@pytest.mark.parametrize(
    "corrupted",
    ["!!!not-base64!!!", "sign\u00e4ture", 42],
    ids=["invalid-base64", "non-ascii", "non-string"],
)
async def test_complete_rejects_corrupted_signature(corrupted: JsonValue) -> None:
    """A `thought_signature` that is not a string or not valid base64 raises
    `AviorUsageError` before any request is sent
    """

    # GIVEN a transcript whose Gemini turn carries a corrupted signature
    # (parametrized as `corrupted`)
    mock_client = _mock_client_returning(_text("ok"))
    provider = _provider(mock_client)
    history: list[Message] = [
        UserMessage.from_text("weather?"),
        AssistantMessage(
            parts=[
                ToolCallPart(
                    call_id="call_1",
                    tool_name="get_weather",
                    args={},
                    provider_details={"thought_signature": corrupted},
                )
            ],
            stop_reason="tool_use",
            provider_name="gemini",
        ),
        ToolMessage(
            parts=[ToolResultPart(call_id="call_1", result=ToolResultOk(content="x"))]
        ),
    ]

    # WHEN `complete` is invoked
    # THEN the corrupted signature is rejected and no request is sent
    with pytest.raises(AviorUsageError, match="thought_signature"):
        await provider.complete(history, _settings())
    mock_client.aio.models.generate_content.assert_not_called()


async def test_complete_treats_empty_signature_as_absent() -> None:
    """An empty `thought_signature` string counts as no signature: the first
    `function_call` gets the placeholder instead of an empty signature.
    """

    # GIVEN a transcript whose Gemini turn carries an empty signature
    mock_client = _mock_client_returning(_text("ok"))
    provider = _provider(mock_client)
    history: list[Message] = [
        UserMessage.from_text("weather?"),
        AssistantMessage(
            parts=[
                ToolCallPart(
                    call_id="call_1",
                    tool_name="get_weather",
                    args={},
                    provider_details={"thought_signature": ""},
                )
            ],
            stop_reason="tool_use",
            provider_name="gemini",
        ),
        ToolMessage(
            parts=[ToolResultPart(call_id="call_1", result=ToolResultOk(content="x"))]
        ),
    ]

    # WHEN `complete` is invoked
    await provider.complete(history, _settings())

    # THEN the placeholder stands in for the empty signature
    model_turn = _call_contents(mock_client)[1]
    assert model_turn.parts is not None
    assert model_turn.parts[0].thought_signature == (
        b"skip_thought_signature_validator"
    )


async def test_complete_keeps_later_calls_unsigned_when_first_is_signed() -> None:
    """A signed first `function_call` suppresses the placeholder: later
    unsigned calls in the turn stay unsigned, matching the shape the Gemini
    API itself produces for parallel calls.
    """

    # GIVEN a continuation transcript whose Gemini turn has a signed first
    # call and an unsigned second call
    mock_client = _mock_client_returning(_text("ok"))
    provider = _provider(mock_client)
    history: list[Message] = [
        UserMessage.from_text("weather?"),
        AssistantMessage(
            parts=[
                ToolCallPart(
                    call_id="a_1",
                    tool_name="get_weather",
                    args={},
                    provider_details={"thought_signature": _SIGNATURE_B64},
                ),
                ToolCallPart(
                    call_id="a_2",
                    tool_name="get_weather",
                    args={},
                ),
            ],
            stop_reason="tool_use",
            provider_name="gemini",
        ),
        ToolMessage(
            parts=[
                ToolResultPart(call_id="a_1", result=ToolResultOk(content="sunny")),
                ToolResultPart(call_id="a_2", result=ToolResultOk(content="rainy")),
            ]
        ),
    ]

    # WHEN `complete` is invoked
    await provider.complete(history, _settings())

    # THEN the first call echoes its signature and the second gets nothing
    model_turn = _call_contents(mock_client)[1]
    assert model_turn.parts is not None
    first, second = model_turn.parts
    assert first.thought_signature == _SIGNATURE
    assert second.thought_signature is None


async def test_complete_stamps_placeholder_on_first_call_not_first_part() -> None:
    """The placeholder targets the turn's first `function_call` part: a text
    part before it does not receive the placeholder.
    """

    # GIVEN a continuation transcript whose hand-built turn puts text before
    # an unsigned tool call
    mock_client = _mock_client_returning(_text("ok"))
    provider = _provider(mock_client)
    history: list[Message] = [
        UserMessage.from_text("weather?"),
        AssistantMessage(
            parts=[
                TextPart(text="Checking."),
                ToolCallPart(call_id="call_1", tool_name="get_weather", args={}),
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

    # THEN the text part stays unsigned and the call carries the placeholder
    model_turn = _call_contents(mock_client)[1]
    assert model_turn.parts is not None
    text_part, call_part = model_turn.parts
    assert text_part.thought_signature is None
    assert call_part.thought_signature == b"skip_thought_signature_validator"


async def test_complete_drops_thinking_part_without_token() -> None:
    """A reasoning step with no signature is dropped from the wire turn: the
    Gemini API needs nothing back from it.
    """

    # GIVEN a continuation transcript whose Gemini turn holds an unsigned
    # thinking part next to the answer text
    mock_client = _mock_client_returning(_text("ok"))
    provider = _provider(mock_client)
    history: list[Message] = [
        UserMessage.from_text("hi"),
        AssistantMessage(
            parts=[ThinkingPart(content="hmm"), TextPart(text="Hello.")],
            stop_reason="stop",
            provider_name="gemini",
        ),
        UserMessage.from_text("and again?"),
    ]

    # WHEN `complete` is invoked
    await provider.complete(history, _settings())

    # THEN only the text part is sent
    model_turn = _call_contents(mock_client)[1]
    assert model_turn.parts is not None
    assert len(model_turn.parts) == 1
    assert model_turn.parts[0].text == "Hello."
    assert model_turn.parts[0].thought is None


async def test_complete_omits_assistant_turn_that_filters_to_no_content() -> None:
    """An assistant turn whose parts all drop out (only reasoning steps, none
    of them echoable) is omitted from the request instead of sent empty.
    """

    # GIVEN a transcript whose middle turn is a foreign thinking-only turn
    mock_client = _mock_client_returning(_text("ok"))
    provider = _provider(mock_client)
    history: list[Message] = [
        UserMessage.from_text("hi"),
        AssistantMessage(
            parts=[
                ThinkingPart(
                    content="hmm",
                    provider_details={"signature": "sig"},
                )
            ],
            stop_reason="stop",
            provider_name="anthropic",
        ),
        UserMessage.from_text("and again?"),
    ]

    # WHEN `complete` is invoked
    await provider.complete(history, _settings())

    # THEN the thinking-only turn is omitted from the wire transcript
    contents = _call_contents(mock_client)
    assert len(contents) == 2
    assert all(content.role == "user" for content in contents)


# Lifecycle tests
# -----------------------------------------------------------------------------


def _provider_owning(
    monkeypatch: pytest.MonkeyPatch,
    client: AsyncMock,
) -> GeminiProvider:
    """Construct a provider that "owns" a mock client.

    Patches the `genai.Client` symbol in the provider module so the no-`client=`
    path yields the supplied mock - giving the test a handle on the would-be
    self-constructed client without making real network calls.
    """

    # The SDK's `Client.close()` is synchronous, so model it with a plain mock
    # (the async `aio.aclose` stays an `AsyncMock`).
    client.close = MagicMock()

    def _factory(**_: object) -> AsyncMock:
        return client

    monkeypatch.setattr("avior.providers.gemini.genai.Client", _factory)
    return GeminiProvider(api_key="fake")


async def test_aclose_closes_self_constructed_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`aclose` closes the SDK client the provider constructed itself."""

    # GIVEN a provider that constructed its own (mock) client
    mock_client = AsyncMock()
    provider = _provider_owning(monkeypatch, mock_client)

    # WHEN `aclose` is awaited
    await provider.aclose()

    # THEN both the async and the sync client pools are closed
    mock_client.aio.aclose.assert_awaited_once()
    mock_client.close.assert_called_once()


async def test_aclose_leaves_user_supplied_client_open() -> None:
    """`aclose` does not close clients supplied by the caller."""

    # GIVEN a provider with a caller-supplied client
    mock_client = AsyncMock()
    mock_client.close = MagicMock()
    provider = _provider(mock_client)

    # WHEN `aclose` is awaited
    await provider.aclose()

    # THEN neither client pool is closed
    mock_client.aio.aclose.assert_not_called()
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

    # THEN `aclose` ran (visible via both underlying close calls)
    mock_client.aio.aclose.assert_awaited_once()
    mock_client.close.assert_called_once()


async def test_async_cm_exit_leaves_user_supplied_client_open() -> None:
    """`async with` on a user-supplied-client provider does not close it."""

    # GIVEN a provider with a caller-supplied client
    mock_client = AsyncMock()
    mock_client.close = MagicMock()
    provider = _provider(mock_client)

    # WHEN used as an async context manager
    async with provider:
        pass

    # THEN neither client pool is closed
    mock_client.aio.aclose.assert_not_called()
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
            mock_client.aio.aclose.assert_not_called()
        # AND after the inner exit the client is still open (refcount > 0)
        mock_client.aio.aclose.assert_not_called()

    # AND after the outermost exit `aclose` has run exactly once (both pools)
    mock_client.aio.aclose.assert_awaited_once()
    mock_client.close.assert_called_once()
