"""Tests for `avior.providers.gemini`."""

import logging
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from avior.core.context import RunContext
from avior.core.exceptions import (
    ProviderConnectionError,
    ProviderHTTPError,
    ProviderResponseValidationError,
)
from avior.core.messages import (
    AssistantMessage,
    Message,
    StopReason,
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
from avior.providers.gemini import GeminiProvider


def _settings(
    *,
    model: str = "gemini-test",
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> ModelSettings:
    """Construct `ModelSettings` with sensible defaults for tests."""

    return ModelSettings(model=model, max_tokens=max_tokens, temperature=temperature)


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


async def test_complete_skips_thought_parts() -> None:
    """`complete` drops thought-summary parts from the assistant text."""

    # GIVEN a response interleaving a thought part with the answer text
    response = _response(
        types.Part(text="thinking...", thought=True),
        types.Part(text="A"),
    )
    provider = _provider(_mock_client_returning(response))

    # WHEN `complete` is awaited
    result = await provider.complete([UserMessage.from_text("hi")], _settings())

    # THEN only the non-thought text survives
    assert result.message.parts == [TextPart(text="A")]


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
