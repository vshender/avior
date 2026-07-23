"""Native Gemini provider adapter.

Wraps `google.genai.Client` and implements `Provider` against the Gemini API
(`generate_content`).

Install via the optional extra: `pip install avior[gemini]`.
"""

import base64
import logging
import uuid
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypedDict, assert_never

from pydantic import ConfigDict, JsonValue, TypeAdapter

try:
    import httpx
    from google import genai
    from google.genai import errors, types
except ImportError as e:
    raise ImportError(
        "The `google-genai` package is required to use `avior.providers.gemini`. "
        "Install with: pip install avior[gemini]"
    ) from e

from avior.core.exceptions import (
    AviorUsageError,
    ProviderConnectionError,
    ProviderHTTPError,
    ProviderResponseValidationError,
)
from avior.core.messages import (
    AssistantMessage,
    AssistantPart,
    Message,
    StopReason,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolMessage,
    UserMessage,
)
from avior.core.provider import (
    ModelCapabilities,
    ModelSettings,
    Provider,
    ProviderResponse,
    resolve_provider_options,
)
from avior.core.tools import Tool
from avior.core.usage import Usage
from avior.core.warnings import RunWarning, UnsupportedSettingRunWarning

logger = logging.getLogger(__name__)


type _ThinkingShape = Literal["budget", "level"]
"""Which dialect of `thinking_config` a model speaks.

- `"budget"` - thinking depth is a token count in `thinking_budget`; `0`
  turns thinking off and `-1` asks for a model-chosen dynamic depth (the
  Gemini 2.5 generation).  The model rejects `thinking_level`.
- `"level"` - thinking depth is a named level in `thinking_level`; `MINIMAL`
  is the lowest (the Gemini 3 generations).
"""

type _ThinkingMode = Literal["off_by_default", "on_by_default", "always_on"]
"""How a model treats thinking.

- `"off_by_default"` - the model does not think unless the config turns
  thinking on.
- `"on_by_default"` - the model thinks unless the config turns thinking off.
- `"always_on"` - the model thinks on every response and cannot be turned off,
  so a request to disable it is dropped and the model keeps thinking.

Which config values turn thinking on or off is the other axis: the model's
`_ThinkingShape`.
"""

_THINKING_MODELS: dict[str, tuple[_ThinkingShape, _ThinkingMode]] = {
    "gemini-2.5-flash": ("budget", "on_by_default"),
    "gemini-2.5-flash-lite": ("budget", "off_by_default"),
    "gemini-2.5-pro": ("budget", "always_on"),
    "gemini-3-flash": ("level", "on_by_default"),
    "gemini-3-pro": ("level", "always_on"),
    "gemini-3.1-flash-lite": ("level", "off_by_default"),
    "gemini-3.1-pro": ("level", "always_on"),
    "gemini-3.5-flash": ("level", "on_by_default"),
    "gemini-3.5-flash-lite": ("level", "off_by_default"),
    "gemini-3.6-flash": ("level", "on_by_default"),
    # Moving aliases for the newest flash / flash-lite / pro model.
    # Hand-maintained: re-check the classification when Google repoints an
    # alias at a new generation.
    "gemini-flash-latest": ("level", "on_by_default"),
    "gemini-flash-lite-latest": ("level", "off_by_default"),
    "gemini-pro-latest": ("level", "always_on"),
}
"""Config shape and thinking mode per model, keyed by model-id family.

A model matches a family when its id is the family itself or the family plus
a version tail (see `_matches_family`), so `gemini-3-flash-preview` and
`gemini-2.5-flash-preview-05-20` resolve to their families while a named
variant with its own behavior (`gemini-2.5-flash-image`,
`gemini-2.5-flash-preview-tts`) matches nothing and is not treated as a
thinking model.  The families are disjoint under that matching, so match
order does not matter.

The classifications are seeded from probing the live Gemini API.
"""


def _thinking_support(model: str) -> tuple[_ThinkingShape, _ThinkingMode] | None:
    """Return how `model` supports thinking - its config shape and thinking
    mode - or `None` if avior does not treat it as thinking: either a known
    non-thinking model or one it does not recognize.
    """

    # The Gemini SDK also accepts a model's fully-qualified resource name
    # (`models/gemini-2.5-flash`); classify it like the bare id.  A
    # `tunedModels/...` resource stays unrecognized: a fine-tune carries a
    # user-chosen name that says nothing about the base model.
    model = model.removeprefix("models/")

    for family, support in _THINKING_MODELS.items():
        if _matches_family(model, family):
            return support

    return None


def _matches_family(model: str, family: str) -> bool:
    """Whether `model` is `family` itself or `family` plus a version tail.

    A version tail is one or more hyphen-separated segments, each either all
    digits, `preview`, or `latest` - so `gemini-2.5-flash-preview-05-20`
    matches the family `gemini-2.5-flash`, while `gemini-2.5-flash-image`
    does not: `image` names a different model, not a version of the family.
    """

    if model == family:
        return True
    if not model.startswith(family + "-"):
        return False

    tail = model[len(family) + 1 :]
    return all(
        segment.isdigit() or segment in ("preview", "latest")
        for segment in tail.split("-")
    )


_THINKING_BUDGET_TOKENS: dict[Literal["low", "medium", "high"], int] = {
    "low": 2048,
    "medium": 8192,
    "high": 24576,
}
"""`thinking_budget` for each portable thinking level on a budget-shape model.

The values fit the `thinking_budget` range of every budget-shape model;
`high` is the largest budget the 2.5-flash family accepts.
"""

_THINKING_LEVELS: dict[Literal["low", "medium", "high"], types.ThinkingLevel] = {
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
}
"""`thinking_level` for each portable thinking level on a level-shape model."""

_DEFAULT_LEVEL = types.ThinkingLevel.MEDIUM
"""The level that `thinking=True` selects on an `off_by_default` level-shape
model.

An `off_by_default` model needs an explicit config to start thinking.
`MEDIUM` is a moderate default: a middle level, so `thinking=True` turns
thinking on without committing to the lightest or deepest setting.  A
budget-shape model does not need this constant: its dialect has a native
"model-chosen depth" value, `thinking_budget=-1`.
"""

_SKIP_SIGNATURE_VALIDATOR = b"skip_thought_signature_validator"
"""Placeholder `thought_signature` accepted by the Gemini API in place of a
real one.

Thinking Gemini models attach an opaque `thought_signature` - an encrypted
snapshot of the model's reasoning state - to parts they emit, and the Gemini 3
generations validate replay: a model turn whose first `function_call` part
carries no signature is rejected.  This placeholder is Google's documented
escape hatch - it tells the Gemini API to skip signature validation for the
part.  avior stamps it on a model turn's first `function_call` part when
that part carries no signature to echo, which happens when:

- the turn was produced by another provider, so no Gemini signature exists;
- the turn was built by hand (`provider_name` is `None`);
- the turn was produced by a Gemini model that does not sign - for example a
  Gemini 2.5 model with thinking disabled;
- the turn was produced by a signing Gemini model but avior lost the signature -
  a round-trip bug.  The placeholder masks such a bug as silent quality
  degradation instead of a loud rejection, so the signature round-trip is
  guarded by avior's own tests rather than left to the Gemini API's check.

The first three are legitimate and expected; they are what keeps a
hand-built, cross-provider, or non-signing Gemini transcript replayable on a
validating model.
"""


class GeminiProviderOptions(TypedDict, total=False):
    """Gemini-specific `provider_options["gemini"]` settings.

    A raw Gemini thinking config, for control that the portable
    `ModelSettings.thinking` setting does not reach.  It takes precedence over
    that portable setting.  Unknown keys are rejected when the slice is
    validated.
    """

    # Forbid unknown keys when validating the slice.  The type-checker ignore
    # comments are needed because a `TypedDict` body normally holds only field
    # annotations, not this assignment.
    __pydantic_config__ = ConfigDict(  # type: ignore[misc]  # pyright: ignore
        extra="forbid"
    )

    thinking_config: types.ThinkingConfigDict
    """A raw Gemini thinking config, passed through in place of the portable
    setting.  It bypasses the portable mapping, so it still drives thinking on
    a model avior does not yet classify, and reaches settings the portable
    setting does not - notably `include_thoughts`, to return thought summaries.
    avior validates its shape against the installed Gemini SDK types before
    sending, so it must match that version's config, not a newer one.
    """


_GEMINI_OPTIONS_ADAPTER = TypeAdapter(GeminiProviderOptions)
"""`TypeAdapter` for the `provider_options["gemini"]` slice."""


class GeminiProvider(Provider):
    """Async adapter to the Gemini Developer API.

    Translates avior's canonical `Message` shape to and from Gemini's
    `Content` shape.  Gemini's wire format diverges from the OpenAI-style
    convention in a few ways the adapter absorbs:

    - the assistant role is `"model"`, not `"assistant"`;
    - tool calls and results are `function_call` / `function_response` parts,
      not a dedicated tool role - and a `function_response` carries the tool's
      `name`, which avior's `ToolResultPart` does not store, so it is recovered
      from the matching tool call earlier in the transcript;
    - tool results travel in a `"user"` turn.

    Exceptions from the Gemini SDK are translated to avior's provider-agnostic
    hierarchy (`ProviderError` and subclasses), with the original exception
    preserved as `__cause__`.
    """

    def __init__(
        self,
        *,
        client: genai.Client | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the provider.

        Args:
            client: A pre-built `google.genai.Client` instance.  Takes
                precedence over `api_key` if both are supplied.  Lifecycle stays
                with the caller; `aclose` will not close it.
            api_key: API key for a freshly constructed `Client`.  If both
                `client` and `api_key` are `None`, `Client` reads
                `GOOGLE_API_KEY` / `GEMINI_API_KEY` from the environment.  A
                self-constructed client is closed by `aclose`.
        """

        super().__init__()
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = genai.Client(api_key=api_key)
            self._owns_client = True

    @property
    def name(self) -> str:
        """The provider's canonical name."""

        return "gemini"

    def model_capabilities(self, model: str) -> ModelCapabilities:
        """Report what `model` supports.

        Reports `supports_thinking=True` for a recognized thinking model - one
        that `_thinking_support` classifies - and the conservative default
        otherwise.
        """

        return ModelCapabilities(supports_thinking=_thinking_support(model) is not None)

    async def complete(
        self,
        messages: Sequence[Message],
        settings: ModelSettings,
        *,
        tools: Sequence[Tool[Any, Any, Any]] = (),
        system_prompt: str | None = None,
    ) -> ProviderResponse:
        """Send the conversation to Gemini and return the response.

        The portable `settings` map to Gemini's request as follows:

        - `temperature` and `max_output_tokens` - forwarded only when explicitly
          set on `settings`.  Gemini accepts a custom `temperature` with
          thinking active, so none is dropped.
        - `thinking` - the portable setting maps to the chosen model's native
          `thinking_config`:

          - a level (`low` / `medium` / `high`) becomes a `thinking_budget`
            token count on a Gemini 2.5 model and a `thinking_level` on a
            Gemini 3 or newer model;
          - `True` sends an enabling config on an `off_by_default` model - a
            model-chosen depth (Gemini 2.5) or the default `medium` level
            (Gemini 3 and newer) - and leaves a model that already thinks by
            default at its own depth;
          - `False` turns thinking off on an `off_by_default` or
            `on_by_default` model: `thinking_budget=0` on a Gemini 2.5 model,
            `thinking_level=MINIMAL` - the dialect's lowest setting - on a
            Gemini 3 or newer model.

          Thought summaries are not requested; ask for them with
          `include_thoughts` in the raw config.  A request the model cannot
          honor is dropped, with an `UnsupportedSettingRunWarning` on the
          response: enabling thinking on a model avior does not treat as
          thinking, or disabling it on an `always_on` model.

          `provider_options["gemini"]` overrides this mapping; see
          `GeminiProviderOptions`.

        Args:
            messages: Conversation transcript (user / assistant / tool turns).
            settings: Per-call invocation settings.
            tools: Tools to offer the model.  Each is sent as a
                `FunctionDeclaration` whose `parameters_json_schema` carries the
                tool's arguments JSON schema unchanged; `function_call` parts in
                the response are parsed back into `ToolCallPart`s.
            system_prompt: The system prompt, sent in Gemini's
                `system_instruction` config field, or `None` for no system
                prompt.

        Returns:
            A `ProviderResponse` wrapping the assistant message together with
            the call metadata.

        Raises:
            AviorUsageError: The `gemini` `provider_options` slice is invalid
                (an unknown key or a value of the wrong type), or a transcript
                part carries a corrupted `thought_signature`; raised before
                the request is sent.
            ProviderHTTPError: The provider returned a 4xx or 5xx HTTP response.
                `status_code` carries the wire status.
            ProviderResponseValidationError: The provider returned a successful
                response that could not be decoded (typically an outdated
                `google-genai` package) or that carries a content part the
                adapter does not support.
            ProviderConnectionError: Network-level failure (DNS / TCP / TLS /
                timeout) - no HTTP response was received.

        These are the known Gemini SDK failures the adapter translates; each
        derives from `ProviderError` and preserves the original Gemini SDK
        exception as `__cause__`.  The SDK exposes no single base exception to
        catch as a catch-all, so an unrecognized SDK error propagates
        untranslated rather than being masked by an over-broad `except`.
        (An abnormal terminal finish is not raised here - it becomes the
        `"error"` stop reason; see `_map_stop_reason`.)
        """

        logger.debug("complete: model=%s, messages=%d", settings.model, len(messages))

        # A `function_response` needs the tool's name, which `ToolResultPart`
        # does not carry; recover it from the matching `ToolCallPart` earlier
        # in the transcript, keyed by the shared `call_id`.
        call_names = {
            part.call_id: part.tool_name
            for message in messages
            if isinstance(message, AssistantMessage)
            for part in message.parts
            if isinstance(part, ToolCallPart)
        }
        wire_messages = [
            wire_message
            for m in messages
            if (wire_message := self._to_wire(m, call_names)) is not None
        ]

        # The Gemini SDK types its `tools` field as an invariant
        # `list[Tool | Callable]`; a mutable `list[Tool]` is not assignable to
        # it, so the wider element type is declared here.
        tools_param: list[types.Tool | Callable[..., Any]] | None = (
            [
                types.Tool(
                    function_declarations=[
                        self._to_function_declaration(t) for t in tools
                    ]
                )
            ]
            if tools
            else None
        )
        warnings: list[RunWarning] = []
        thinking_config = self._resolve_thinking(settings, warnings)
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=settings.max_tokens,
            temperature=settings.temperature,
            tools=tools_param,
            thinking_config=thinking_config,
        )

        try:
            # The Gemini SDK's `contents` parameter type includes `PIL.Image`
            # from the optional `Pillow` dependency, which we do not install, so
            # part of the signature resolves to `Unknown` and the call is
            # flagged.  We pass `Content` objects, never images, so this is
            # safe.
            response = await self._client.aio.models.generate_content(  # pyright: ignore[reportUnknownMemberType]
                model=settings.model,
                contents=wire_messages,
                config=config,
            )
        # google-genai has no single base exception class to catch as a
        # catch-all: its API errors derive from `errors.APIError` (an
        # `Exception`), its other errors from `ValueError`.  So we translate the
        # concrete types it raises and add no bare `ProviderError` fallback - an
        # `except Exception` would be too broad.
        except errors.APIError as e:
            raise ProviderHTTPError(str(e), status_code=e.code) from e
        except errors.UnknownApiResponseError as e:
            raise ProviderResponseValidationError(str(e)) from e
        except httpx.TransportError as e:
            raise ProviderConnectionError(str(e)) from e

        parts: list[AssistantPart] = []
        candidate = response.candidates[0] if response.candidates else None
        saw_nameless_call = False
        if candidate is not None and candidate.content is not None:
            # Gemini omits the `id` on some function calls and can repeat one
            # across parallel calls or turns.  A call id must stay unique across
            # the transcript: a tool result recovers its tool name via
            # `call_names` keyed by id, so a duplicate would cross-wire the
            # result to the wrong call.  Use the function call's own `id` only
            # when present and not already taken (the seen set is seeded with
            # the transcript's ids); otherwise mint a random one.  A call id
            # only needs to be unique within the transcript, which stores it
            # once generated, so a non-deterministic id is fine.
            seen_call_ids: set[str] = set(call_names)

            for part in candidate.content.parts or []:
                # A thinking model attaches an opaque `thought_signature` to
                # parts it may later validate on replay.  It is `bytes` in the
                # Gemini SDK while `provider_details` holds only JSON values,
                # so it is stored base64-encoded and decoded back on echo.
                provider_details: dict[str, JsonValue] | None = None
                if part.thought_signature:
                    provider_details = {
                        "thought_signature": base64.b64encode(
                            part.thought_signature
                        ).decode("ascii")
                    }

                if part.thought:
                    # A thought summary is a reasoning step, not the answer.
                    # Summaries are only present when the config requests them
                    # via `include_thoughts`.
                    parts.append(
                        ThinkingPart(
                            content=part.text or "",
                            provider_details=provider_details,
                        )
                    )

                elif part.text is not None:
                    parts.append(
                        TextPart(text=part.text, provider_details=provider_details)
                    )

                elif part.function_call is not None:
                    call = part.function_call
                    if not call.name:
                        # A nameless call is unusable; drop it now and decide
                        # whether to fail once the stop reason is known (a
                        # terminal finish may have just truncated it).
                        saw_nameless_call = True
                        continue

                    call_id = call.id
                    if not call_id or call_id in seen_call_ids:
                        call_id = f"call_{uuid.uuid4().hex}"

                    seen_call_ids.add(call_id)

                    parts.append(
                        ToolCallPart(
                            call_id=call_id,
                            tool_name=call.name,
                            args=call.args or {},
                            provider_details=provider_details,
                        )
                    )

                else:
                    # A part that is neither text nor a function call carries
                    # content the adapter cannot represent yet (e.g.
                    # `inline_data`, `executable_code`, or a code-execution
                    # result).  Fail loud rather than silently dropping it and
                    # returning a misleadingly-successful response.
                    raise ProviderResponseValidationError(
                        "Gemini returned an unsupported content part."
                    )

        finish_reason = candidate.finish_reason if candidate is not None else None
        stop_reason = self._map_stop_reason(
            finish_reason,
            prompt_blocked=(
                response.prompt_feedback is not None
                and response.prompt_feedback.block_reason is not None
            ),
            has_candidate=candidate is not None,
            has_tool_calls=any(isinstance(p, ToolCallPart) for p in parts),
        )

        # A nameless call on a continuable finish (`"stop"` / `"tool_use"`) is
        # malformed tool-call data from the model - the same class as
        # `MALFORMED_FUNCTION_CALL` - so surface it as the canonical `"error"`
        # stop reason, not a provider decode error.  On a terminal finish the
        # stop reason already describes the outcome and the dropped partial call
        # is just an artifact of the truncation or block, so let it pass.
        nameless_drove_error = saw_nameless_call and stop_reason in (
            "stop",
            "tool_use",
        )
        if nameless_drove_error:
            stop_reason = "error"

        if stop_reason == "error":
            # The canonical `"error"` reason drops the provider-specific cause;
            # log it so an abnormal finish stays diagnosable.
            if nameless_drove_error:
                reason = "a function call had no name"
            elif candidate is None:
                reason = "no candidate returned"
            else:
                reason = f"finish_reason={finish_reason}"
            logger.warning("Gemini finished abnormally: %s", reason)

        return ProviderResponse(
            message=AssistantMessage(
                parts=parts,
                stop_reason=stop_reason,
                provider_name=self.name,
            ),
            usage=self._map_usage(response.usage_metadata),
            raw_usage=(
                response.usage_metadata.model_dump(mode="json")
                if response.usage_metadata is not None
                else None
            ),
            response_id=response.response_id,
            model=response.model_version,
            provider_name=self.name,
            warnings=warnings,
        )

    async def aclose(self) -> None:
        """Close the underlying SDK client when this provider owns it.

        Closes **both** httpx pools the `Client` holds: `aio.aclose()` releases
        the async pool (the one this adapter uses) and `close()` the sync pool -
        the Gemini SDK creates both and each method closes only its own.

        No-op when the client was supplied by the caller via `client=` - its
        lifecycle belongs to whoever passed it in.  Safe to call more than once:
        both close methods (and the httpx pools they delegate to) are
        idempotent.
        """

        if self._owns_client:
            await self._client.aio.aclose()
            self._client.close()

    @staticmethod
    def _map_usage(
        usage: types.GenerateContentResponseUsageMetadata | None,
    ) -> Usage | None:
        """Map Gemini's usage metadata to the canonical `Usage`, or `None` when
        the response carried no usage metadata.

        - `input_tokens`: `prompt_token_count` plus
          `tool_use_prompt_token_count` (tokens from tool-execution results fed
          back to the model as input, reported separately by Gemini), folded
          together into a true input total.  Cached tokens are a subset of the
          prompt count, not a separate addend.
        - `output_tokens`: `candidates_token_count` plus `thoughts_token_count`
          - Gemini reports thinking tokens *outside* the candidates count, so
          they are folded back in to keep `output_tokens` a true total.
        - `reasoning_tokens`: `thoughts_token_count` - Gemini itemizes thinking,
          so this is a concrete number (`0` when the turn did no thinking),
          not unknown.
        - `cache_read_tokens`: `cached_content_token_count`.
        - `cache_write_tokens`: `0` - Gemini creates a cache via a separate API
          call, so a `generate_content` response reports no cache writes.

        avior's `total_tokens` is derived (`input + output`) and cross-checked
        against Gemini's own `total_token_count` (a mismatch is logged below).
        """

        if usage is None:
            return None

        candidates = usage.candidates_token_count or 0
        thoughts = usage.thoughts_token_count or 0
        prompt = usage.prompt_token_count or 0
        tool_use_prompt = usage.tool_use_prompt_token_count or 0
        mapped = Usage(
            input_tokens=prompt + tool_use_prompt,
            output_tokens=candidates + thoughts,
            cache_read_tokens=usage.cached_content_token_count or 0,
            cache_write_tokens=0,
            reasoning_tokens=thoughts,
        )

        # Gemini reports its own total, which should match the total our mapped
        # `Usage` derives.  If they ever diverge (e.g. Google changing which
        # sub-counts the total includes), warn rather than silently shipping a
        # total that disagrees with the provider (whose own number stays in
        # `raw_usage`).
        reported_total = usage.total_token_count
        if reported_total is not None and reported_total != mapped.total_tokens:
            logger.warning(
                "Gemini reported total_token_count=%d != avior's derived total "
                "%d; avior reports the derived total (provider's stays in "
                "raw_usage).",
                reported_total,
                mapped.total_tokens,
            )

        return mapped

    @staticmethod
    def _map_stop_reason(
        finish_reason: types.FinishReason | None,
        *,
        prompt_blocked: bool,
        has_candidate: bool,
        has_tool_calls: bool,
    ) -> StopReason:
        """Map Gemini's `finish_reason` to canonical `StopReason`.

        Terminal finish reasons are resolved first, so an abnormal finish wins
        over a tool request.  Gemini has no tool-use finish reason - a tool
        request finishes with `STOP` - so a tool call is inferred from the
        presence of `function_call` parts, but only on an otherwise-normal
        finish.  That keeps a truncated, blocked, or malformed response from
        being reported as `"tool_use"` and executed.  A prompt blocked before
        generation yields no candidate (so no `finish_reason`); `prompt_blocked`
        carries that case.

        Precedence (first match wins):

        - prompt blocked -> `"content_filter"` (moderation blocked the prompt).
        - no candidate at all (and not a prompt block) -> `"error"`: an empty
          response with no candidate is abnormal, not a normal empty stop.
        - `MAX_TOKENS` -> `"max_tokens"` (output truncated at the cap).
        - safety / recitation block (`SAFETY`, `PROHIBITED_CONTENT`,
          `BLOCKLIST`, `SPII`, `RECITATION`, and the `IMAGE_*` variants) ->
          `"content_filter"` (moderation blocked the response).
        - `MALFORMED_FUNCTION_CALL` / `UNEXPECTED_TOOL_CALL` (no usable tool
          call), `LANGUAGE` (unsupported language), `NO_IMAGE` -> `"error"`.
        - `STOP` / `None` -> `"tool_use"` if `function_call` parts are present,
          else `"stop"` (the only continuable finishes).
        - `OTHER` / `FINISH_REASON_UNSPECIFIED` / `IMAGE_OTHER` -> `"error"`:
          not a clean stop, so surfaced loudly rather than as a successful
          response.
        """

        if prompt_blocked:
            return "content_filter"

        if not has_candidate:
            return "error"

        match finish_reason:
            case types.FinishReason.MAX_TOKENS:
                return "max_tokens"

            case (
                types.FinishReason.SAFETY
                | types.FinishReason.PROHIBITED_CONTENT
                | types.FinishReason.BLOCKLIST
                | types.FinishReason.SPII
                | types.FinishReason.RECITATION
                | types.FinishReason.IMAGE_SAFETY
                | types.FinishReason.IMAGE_PROHIBITED_CONTENT
                | types.FinishReason.IMAGE_RECITATION
            ):
                return "content_filter"

            case (
                types.FinishReason.MALFORMED_FUNCTION_CALL
                | types.FinishReason.UNEXPECTED_TOOL_CALL
                | types.FinishReason.LANGUAGE
                | types.FinishReason.NO_IMAGE
            ):
                return "error"

            case None | types.FinishReason.STOP:
                # The only continuable finishes.  Gemini reports a tool request
                # with `STOP`, so detect it here from the function-call parts.
                return "tool_use" if has_tool_calls else "stop"

            case (
                types.FinishReason.OTHER
                | types.FinishReason.FINISH_REASON_UNSPECIFIED
                | types.FinishReason.IMAGE_OTHER
            ):
                # Not a clean stop, so surface as `"error"` rather than a
                # silently-successful empty/partial response.
                return "error"

            case _:
                # Every known `FinishReason` is handled above; a value reaching
                # here is one a newer `google-genai` added.  Fail loud (and
                # trip the static exhaustiveness check) so it gets an explicit
                # mapping instead of silently bucketing as `"error"`.
                assert_never(finish_reason)

    def _resolve_thinking(
        self,
        settings: ModelSettings,
        warnings: list[RunWarning],
    ) -> types.ThinkingConfig | None:
        """Map the thinking settings to Gemini's native config.

        Returns the `thinking_config` config field, or `None` when none should
        be sent.

        The raw `gemini` provider options take precedence over the portable
        `thinking` setting; `GeminiProviderOptions` documents how they combine.
        Without them, the portable setting maps to the model's native shape -
        see `_portable_thinking`.
        """

        options = resolve_provider_options(
            settings,
            self.name,
            _GEMINI_OPTIONS_ADAPTER,
        )
        raw_config = options.get("thinking_config")
        if raw_config is not None:
            return types.ThinkingConfig.model_validate(raw_config)
        else:
            return self._portable_thinking(settings, warnings)

    def _portable_thinking(
        self,
        settings: ModelSettings,
        warnings: list[RunWarning],
    ) -> types.ThinkingConfig | None:
        """Map the portable `thinking` level to a native config for `settings`.

        Returns the `thinking_config` field (or `None`), appending an
        `UnsupportedSettingRunWarning` when a request is dropped.  `complete`
        documents the mapping and the drop conditions.
        """

        thinking = settings.thinking
        if thinking is None:
            return None

        support = _thinking_support(settings.model)
        if support is None:
            # The model is not a recognized thinking model.  A request to
            # enable (`True` / a level) is dropped and warned; the reason is
            # honest for both a model that genuinely does not think and one
            # avior does not recognize, pointing to the `gemini` provider
            # options rather than claiming the model cannot think.  Disabling
            # (`False`) is a harmless no-op.
            if thinking is not False:
                warnings.append(
                    self._thinking_dropped(
                        settings,
                        "the model is not a recognized thinking model; if it "
                        "does think, configure thinking via the `gemini` "
                        "provider options",
                    )
                )
            return None

        shape, mode = support
        if thinking is False:
            match mode:
                case "off_by_default" | "on_by_default":
                    match shape:
                        case "budget":
                            return types.ThinkingConfig(thinking_budget=0)
                        case "level":
                            # The level dialect has no full off; `MINIMAL` is
                            # its lowest setting.
                            return types.ThinkingConfig(
                                thinking_level=types.ThinkingLevel.MINIMAL
                            )
                        case _:
                            assert_never(shape)
                case "always_on":
                    warnings.append(
                        self._thinking_dropped(
                            settings,
                            "the model's thinking is always on and cannot be disabled",
                        )
                    )
                    return None
                case _:
                    assert_never(mode)

        elif thinking is True:
            match mode:
                case "off_by_default":
                    # An `off_by_default` model needs an explicit config to
                    # start thinking.  The budget dialect has a native
                    # model-chosen depth (`-1`); the level dialect does not,
                    # so a moderate default level stands in.
                    match shape:
                        case "budget":
                            return types.ThinkingConfig(thinking_budget=-1)
                        case "level":
                            return types.ThinkingConfig(thinking_level=_DEFAULT_LEVEL)
                        case _:
                            assert_never(shape)
                case "on_by_default" | "always_on":
                    # The model already thinks at its own default depth.
                    return None
                case _:
                    assert_never(mode)

        else:
            match shape:
                case "budget":
                    return types.ThinkingConfig(
                        thinking_budget=_THINKING_BUDGET_TOKENS[thinking]
                    )
                case "level":
                    return types.ThinkingConfig(
                        thinking_level=_THINKING_LEVELS[thinking]
                    )
                case _:
                    assert_never(shape)

    def _thinking_dropped(
        self,
        settings: ModelSettings,
        reason: str | None = None,
    ) -> UnsupportedSettingRunWarning:
        """Build the warning for a `thinking` request that was dropped.

        `reason` is an optional standalone explanation of why the request could
        not be honored; it is `None` when the generic message already says
        enough.
        """

        return UnsupportedSettingRunWarning(
            setting_name="thinking",
            setting_value=settings.thinking,
            reason=reason,
            provider=self.name,
            model=settings.model,
        )

    @staticmethod
    def _to_function_declaration(
        tool: Tool[Any, Any, Any],
    ) -> types.FunctionDeclaration:
        """Convert an avior `Tool` to a Gemini function declaration.

        A tool's parameters are described by a JSON Schema.  The Gemini SDK
        accepts those parameters in two forms:

        - `parameters`: its own `Schema` type - a restricted dialect based on
          a subset of the OpenAPI schema object;
        - `parameters_json_schema`: a standard JSON Schema, taken as-is.

        avior uses `parameters_json_schema`, passing the `args_model` schema
        through unchanged; targeting `Schema` would instead require rewriting it
        (remapping types, dropping unsupported keywords, inlining `$ref`s).
        """

        return types.FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters_json_schema=tool.args_model.model_json_schema(),
        )

    def _to_wire(
        self,
        message: Message,
        call_names: dict[str, str],
    ) -> types.Content | None:
        """Convert an avior `Message` to a Gemini `Content`, or `None`.

        Maps each message type to Gemini's wire shape:

        - `UserMessage` -> a `"user"` turn of text parts.
        - `AssistantMessage` -> a `"model"` turn; text parts become text parts,
          tool calls become `function_call` parts, and a reasoning step whose
          `provider_details` carry a thought signature becomes a thought part.
          Each part's signature is echoed back unchanged when this provider
          produced the turn, and dropped otherwise - see `_signature_bytes`.
          A turn's first `function_call` part is stamped with
          `_SKIP_SIGNATURE_VALIDATOR` when it carries no signature to echo,
          so a turn that legitimately has none still passes the Gemini API's
          replay validation; later `function_call` parts are accepted
          unsigned.
        - `ToolMessage` -> a `"user"` turn of `function_response` parts (Gemini
          carries tool results in the user role).

          Each result's tool name is looked up in `call_names` by `call_id`,
          and its `call_id` is also set as the `function_response` `id`; the
          assistant turn's `function_call` carries that same `id`, so Gemini
          pairs each result with its call.

          A result whose `call_id` matches no call carries no name, which
          makes the request invalid.  Gemini rejects it with an HTTP 400 that
          surfaces as `ProviderHTTPError` - the same outcome the other
          adapters produce for an orphaned tool result.

        Returns `None` for an assistant turn whose parts all drop out (only
        reasoning steps, none of them echoable): it would serialize to an empty
        turn, which carries nothing for the model, so it is omitted from the
        request.
        """

        match message:
            case UserMessage():
                return types.Content(
                    role="user",
                    parts=[types.Part(text=p.text) for p in message.parts],
                )

            case AssistantMessage():
                asst_parts: list[types.Part] = []
                first_function_call = True
                for part in message.parts:
                    signature = self._signature_bytes(message, part)
                    match part:
                        case TextPart():
                            asst_parts.append(
                                types.Part(
                                    text=part.text,
                                    thought_signature=signature,
                                )
                            )
                        case ToolCallPart():
                            if signature is None and first_function_call:
                                logger.debug(
                                    "Stamping the signature-skip placeholder "
                                    "on function call %s (turn provider: %s).",
                                    part.call_id,
                                    message.provider_name,
                                )
                                signature = _SKIP_SIGNATURE_VALIDATOR
                            first_function_call = False
                            asst_parts.append(
                                types.Part(
                                    function_call=types.FunctionCall(
                                        id=part.call_id,
                                        name=part.tool_name,
                                        args=part.args,
                                    ),
                                    thought_signature=signature,
                                )
                            )
                        case ThinkingPart():
                            # A reasoning step is echoed only for its
                            # signature; one with no signature to echo carries
                            # nothing the Gemini API needs back, so it is
                            # dropped.
                            if signature is not None:
                                asst_parts.append(
                                    types.Part(
                                        text=part.content,
                                        thought=True,
                                        thought_signature=signature,
                                    )
                                )
                        case _:
                            assert_never(part)

                # A turn left empty after dropping non-echoable reasoning
                # steps carries nothing to send; omit it.
                if not asst_parts:
                    return None

                return types.Content(role="model", parts=asst_parts)

            case ToolMessage():
                tool_parts: list[types.Part] = []
                for p in message.parts:
                    # An unmatched `call_id` leaves the name unset on purpose.
                    # Gemini rejects an empty name with a 400, but silently
                    # accepts any non-empty name for a result with no matching
                    # call - feeding the model an orphaned result.  A missing
                    # name must fail loudly, so never substitute a placeholder.
                    name = call_names.get(p.call_id)

                    # Gemini's documented convention: an `"output"` / `"error"`
                    # key names the function output / error; any other key would
                    # be treated as the whole response object instead.
                    match p.result.status:
                        case "ok":
                            key = "output"
                        case "error":
                            key = "error"
                        case _:
                            assert_never(p.result.status)

                    tool_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=p.call_id,
                                name=name,
                                response={key: p.result.content},
                            )
                        )
                    )

                return types.Content(role="user", parts=tool_parts)

            case _:
                assert_never(message)

    def _signature_bytes(
        self,
        message: AssistantMessage,
        part: AssistantPart,
    ) -> bytes | None:
        """Return the thought signature to echo for `part`, or `None`.

        A thought signature round-trips only to the provider that produced
        it: the token is provider-specific, and the Gemini API verifies it on
        replay.  Returns `None` - dropping the signature - when the turn came
        from a different provider, or the part carries no signature.

        `provider_details` stores the signature as base64 text; this decodes
        it back to the `bytes` the Gemini SDK expects.

        Raises:
            AviorUsageError: The stored signature is not valid base64 text -
                the transcript was corrupted after the signature was stored.
        """

        if message.provider_name != self.name:
            return None

        details = part.provider_details or {}
        signature = details.get("thought_signature")
        if signature is None or signature == "":
            return None
        if not isinstance(signature, str):
            raise AviorUsageError(
                "Invalid `thought_signature` in a part's `provider_details`: "
                "the value is not a string, so the transcript was corrupted "
                "after the signature was stored."
            )

        try:
            # `b64decode` raises `binascii.Error` for malformed base64 and a
            # plain `ValueError` for a non-ASCII string; both are
            # `ValueError`s.
            return base64.b64decode(signature, validate=True)
        except ValueError as e:
            raise AviorUsageError(
                "Invalid `thought_signature` in a part's `provider_details`: "
                "the value is not valid base64, so the transcript was "
                "corrupted after the signature was stored."
            ) from e
