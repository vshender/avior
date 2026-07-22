"""Native OpenAI Responses provider adapter.

Wraps `openai.AsyncOpenAI` and implements `Provider` against OpenAI's Responses
API.

Stateless wire: `store=False` is always passed and `previous_response_id` is
not used.  avior treats the conversation transcript as user-owned; no
server-side state is created.

Install via the optional extra: `pip install avior[openai]`.
"""

import json
import logging
from collections.abc import Sequence
from typing import Any, Literal, TypedDict, assert_never

from pydantic import ConfigDict, JsonValue, TypeAdapter

try:
    import openai
    from openai import AsyncOpenAI
    from openai._types import Omit, omit
    from openai.types.responses import (
        EasyInputMessageParam,
        FunctionToolParam,
        Response,
        ResponseFunctionToolCall,
        ResponseFunctionToolCallParam,
        ResponseIncludable,
        ResponseInputItemParam,
        ResponseInputParam,
        ResponseOutputMessage,
        ResponseOutputRefusal,
        ResponseOutputText,
        ResponseReasoningItem,
        ResponseReasoningItemParam,
        ResponseUsage,
    )
    from openai.types.responses.response_input_item_param import FunctionCallOutput
    from openai.types.shared_params import Reasoning
except ImportError as e:
    raise ImportError(
        "The `openai` package is required to use `avior.providers.openai_responses`. "
        "Install with: pip install avior[openai]"
    ) from e

from avior.core.exceptions import (
    ProviderConnectionError,
    ProviderError,
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


type _ReasoningMode = Literal["off_by_default", "on_by_default", "always_on"]
"""How a model treats reasoning.

- `"off_by_default"` - the model does not reason unless an effort level turns
  reasoning on; `effort="none"` keeps it off.
- `"on_by_default"` - the model reasons unless `effort="none"` turns reasoning
  off.
- `"always_on"` - the model reasons on every response and cannot be turned off,
  so a request to disable it is dropped and the model keeps reasoning.
"""


_DEFAULT_EFFORT: Literal["medium"] = "medium"
"""The effort that `thinking=True` selects on an `off_by_default` model.

An `off_by_default` model needs an explicit effort to start reasoning.  `medium`
is a moderate default: a middle level that every `off_by_default` model accepts,
so `thinking=True` turns reasoning on without committing to the lightest or
deepest setting.
"""

_DEFAULT_TEMPERATURE = 1
"""The only `temperature` OpenAI accepts while reasoning is active.

OpenAI rejects any other `temperature` on a request with reasoning active.
OpenAI's `temperature` parameter defaults to this value, so sending it in a
request is the same as omitting it.
"""

_REASONING_RULES: list[tuple[str | None, str | None, _ReasoningMode | None]] = [
    (None, "chat", None),  # a `-chat` variant never reasons
    (None, "pro", "always_on"),  # a `-pro` variant always reasons
    # The one `-codex` that can disable reasoning:
    ("gpt-5.3", "codex", "off_by_default"),
    (None, "codex", "always_on"),  # other `-codex` variants always reason
    (None, "codex-max", "always_on"),
    (None, "codex-mini", "always_on"),
    ("gpt-5", None, "always_on"),  # base `gpt-5` always reasons
    ("gpt-5", "mini", "always_on"),
    ("gpt-5", "nano", "always_on"),
    ("gpt-5.1", None, "off_by_default"),
    ("gpt-5.2", None, "off_by_default"),
    ("gpt-5.4", None, "off_by_default"),
    ("gpt-5.4", "mini", "off_by_default"),
    ("gpt-5.4", "nano", "off_by_default"),
    ("gpt-5.5", None, "on_by_default"),
    ("gpt-5.6", None, "on_by_default"),
    ("gpt-5.6", "sol", "on_by_default"),
    ("gpt-5.6", "terra", "on_by_default"),
    ("gpt-5.6", "luna", "on_by_default"),
    ("o1", None, "always_on"),
    ("o3", None, "always_on"),
    ("o3", "mini", "always_on"),
    ("o4", "mini", "always_on"),
]
"""Ordered rules mapping a model id to its `_ReasoningMode` or to `None`;
first match wins.

Each rule is `(prefix, variant, mode)`:

- `prefix` - the leading model id, a family like `o3` or a version like
  `gpt-5.1`; `None` matches any family.
- `variant` - an optional qualifier such as `mini` or `codex`; `None`
  matches a family with no variant.  A variant is matched in full, so
  `codex-max` needs its own rule rather than folding into `(None, "codex")`.
- `mode` - the `_ReasoningMode` a matching model takes, or `None` for a
  model that is known not to reason.

A model matches when its id decomposes into the `prefix`, the `variant`, and a
version tail (see `_rule_matches`).  So `(None, "codex")` is any `-codex` model,
and `("gpt-5.1", None)` is `gpt-5.1` and its snapshots.

The modes are seeded from probing the live OpenAI API, not derived from version
numbers: `-codex` / `-pro` / `-chat` cut across versions irregularly
(`gpt-5.3-codex` is the lone `-codex` that opts in), so this is a maintained
table, not a formula.  A variant avior has not listed (`gpt-5.4-cyber` and the
like) matches no rule and falls through to no mode; such a model can instead
be driven through an explicit `effort` in the raw `openai` provider options.

Order matters only where one rule is a special case of another: the `gpt-5.3`
`-codex` opt-in comes before the generic `-codex`.  The rest are mutually
exclusive, so their order is free.
"""


def _reasoning_mode(model: str) -> _ReasoningMode | None:
    """Return how `model` treats reasoning, or `None` if avior does not treat
    it as reasoning: either a known non-reasoning model or one it does not
    recognize.

    Applies `_REASONING_RULES` in order and returns the first match, or `None`
    when no rule matches.
    """

    for prefix, variant, mode in _REASONING_RULES:
        if _rule_matches(model, prefix, variant):
            return mode

    return None


def _reasons_by_default(model: str) -> bool:
    """Whether avior classifies `model` as reasoning when no effort is
    requested."""

    mode = _reasoning_mode(model)
    match mode:
        case "on_by_default" | "always_on":
            return True
        case None | "off_by_default":
            return False
        case _:
            assert_never(mode)


def _rule_matches(model: str, prefix: str | None, variant: str | None) -> bool:
    """Check whether `model` matches a `(prefix, variant)` rule.

    A rule names a `prefix` (a model family or version) and an optional
    `variant` (a qualifier such as `mini` or `codex`).  `model` matches when it
    decomposes, in order, into:

    - `prefix` - `model` must equal it or start with `prefix` + `-` (so `gpt-5`
      matches `gpt-5` and `gpt-5-mini` but not `gpt-5.1`, which uses `.`).  A
      `None` prefix matches any family.  The prefix normally marks where the
      variant starts, so without one the variant is searched for from the end of
      `model`.
    - `variant` - the part after the prefix, matched in full: `(None, "codex")`
      does not match `gpt-5.1-codex-max`.  A `None` variant requires nothing
      after the prefix but a version tail.
    - a version tail - what may trail the variant: nothing, `latest`, or a dated
      snapshot (`2025-11-13`).

    A rule must have a prefix or a variant; `(None, None)` matches nothing.
    """

    # Prefix: strip it from the front.  With no prefix the family length is
    # unknown, so find where the variant begins by searching from the end.
    if prefix is not None:
        if model == prefix:
            rest = ""
        elif model.startswith(prefix + "-"):
            rest = model[len(prefix) + 1 :]
        else:
            return False
    elif variant is not None:
        index = model.rfind("-" + variant)
        if index == -1:
            return False
        rest = model[index + 1 :]
    else:
        # No prefix and no variant to match on; with nothing to identify a
        # model, nothing matches.
        return False

    # Variant: consume it from the front of `rest`, leaving the version tail.
    if variant is None:
        version = rest
    elif rest == variant:
        version = ""
    elif rest.startswith(variant + "-"):
        version = rest[len(variant) + 1 :]
    else:
        return False

    # Version tail: what remains must be empty, the `latest` alias, or a dated
    # snapshot whose hyphen-separated parts are all digits (`2025-11-13`).
    return (
        version == ""
        or version == "latest"
        or all(part.isdigit() for part in version.split("-"))
    )


class OpenAIProviderOptions(TypedDict, total=False):
    """OpenAI-specific `provider_options["openai"]` settings.

    A raw OpenAI reasoning config, for control that the portable
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

    reasoning: Reasoning
    """A raw OpenAI reasoning config, passed through in place of the portable
    setting.  It bypasses the portable mapping, so it still drives reasoning on
    a model avior does not yet classify, and reaches settings the portable
    setting does not - notably `summary`, to ask for a human-readable reasoning
    summary.
    avior validates its shape against the installed OpenAI SDK types before
    sending, so it must match that version's config, not a newer one.
    """


_OPENAI_OPTIONS_ADAPTER = TypeAdapter(OpenAIProviderOptions)
"""`TypeAdapter` for the `provider_options["openai"]` slice."""


class OpenAIResponsesProvider(Provider):
    """Async adapter to OpenAI's Responses API.

    Translates avior's canonical `Message` shape to and from OpenAI Responses
    input/output items.  Exceptions from the OpenAI SDK are translated to
    avior's provider-agnostic hierarchy (`ProviderError` and subclasses), with
    the original exception preserved as `__cause__`.
    """

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the provider.

        Args:
            client: A pre-built `AsyncOpenAI` instance.  Takes precedence over
                `api_key` if both are supplied.  Lifecycle stays with the
                caller; `aclose` will not close it.
            api_key: API key for a freshly constructed `AsyncOpenAI`.  If both
                `client` and `api_key` are `None`, `AsyncOpenAI` reads
                `OPENAI_API_KEY` from the environment.  A self-constructed
                client is closed by `aclose`.
        """

        super().__init__()
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = AsyncOpenAI(api_key=api_key)
            self._owns_client = True

    @property
    def name(self) -> str:
        """The provider's canonical name."""

        return "openai"

    def model_capabilities(self, model: str) -> ModelCapabilities:
        """Report what `model` supports.

        Reports `supports_thinking=True` for a recognized reasoning model - one
        that `_reasoning_mode` maps to a mode - and the conservative default
        otherwise.
        """

        return ModelCapabilities(supports_thinking=_reasoning_mode(model) is not None)

    async def complete(
        self,
        messages: Sequence[Message],
        settings: ModelSettings,
        *,
        tools: Sequence[Tool[Any, Any, Any]] = (),
        system_prompt: str | None = None,
    ) -> ProviderResponse:
        """Send the conversation to OpenAI Responses and return the response.

        `store=False` is always passed (stateless wire; no server-side history).
        When reasoning is active for the request,
        `include=["reasoning.encrypted_content"]` is requested so the reasoning
        items can be replayed before their tool calls on the next turn.

        The portable `settings` map to OpenAI's request as follows:

        - `max_tokens` - sent as `max_output_tokens`, only when explicitly set
          on `settings`; otherwise the model's own default applies.
        - `temperature` - forwarded only when explicitly set on `settings`.
          A value other than `1` is dropped, with an
          `UnsupportedSettingRunWarning`, when reasoning is active for the
          request - OpenAI rejects such a value.
        - `thinking` - the portable setting maps to a `reasoning` config chosen
          by the model's reasoning mode (see `_ReasoningMode`):

          - a level (`low` / `medium` / `high`) becomes `reasoning.effort`;
          - `True` sends the default effort on an `off_by_default` model and
            leaves a model that already reasons by default at its own depth;
          - `False` sends `reasoning.effort="none"` on an `off_by_default` or
            `on_by_default` model.

          A request the model cannot honor is dropped, with an
          `UnsupportedSettingRunWarning` on the response: enabling reasoning on
          a model avior does not treat as reasoning, or disabling it on an
          `always_on` model.

          `provider_options["openai"]` overrides this mapping; see
          `OpenAIProviderOptions`.

        Args:
            messages: Conversation transcript (user / assistant / tool turns).
            settings: Per-call invocation settings.
            tools: Tools to offer the model.  Each is sent as a Responses
                function tool with its arguments JSON schema as `parameters`
                (`strict=False`: the schema guides the model but is not
                grammar-enforced).  `function_call` items in the response are
                parsed back into `ToolCallPart`s.
            system_prompt: The system prompt, sent in Responses' top-level
                `instructions` parameter (which OpenAI folds to the `developer`
                role for reasoning models), or `None` for no system prompt.

        Returns:
            A `ProviderResponse` wrapping the assistant message together with
            the call metadata.

        Raises:
            AviorUsageError: The `openai` `provider_options` slice is invalid
                (an unknown key or a value of the wrong type), raised before
                the request is sent.
            ProviderHTTPError: The provider returned a 4xx or 5xx HTTP response.
                `status_code` carries the wire status.
            ProviderResponseValidationError: The provider returned a successful
                response that could not be decoded (typically an outdated
                `openai` package), or that carries an output item the adapter
                does not support.
            ProviderConnectionError: Network-level failure (DNS / TCP / TLS /
                timeout) - no HTTP response was received.
            ProviderError: Any other unexpected failure from the OpenAI SDK.

        Errors translated from an OpenAI SDK exception preserve it as
        `__cause__`; validation errors avior detects in an otherwise successful
        response are raised directly.
        """

        logger.debug("complete: model=%s, messages=%d", settings.model, len(messages))

        warnings: list[RunWarning] = []
        reasoning_param = self._resolve_reasoning(settings, warnings)

        # An explicit `effort` decides whether reasoning is active for the
        # request: a level turns reasoning on; `"none"` turns it off.  Through
        # a raw config in provider options, an effort can turn reasoning on
        # even for a model avior does not recognize by name.  Without an
        # explicit effort - the portable mapping produced no config, or a raw
        # config carries no `effort` - the model's default, as avior
        # classifies it, applies.  An encrypted reasoning item is requested
        # and replayed only when reasoning is active.
        effort = (
            None if isinstance(reasoning_param, Omit) else reasoning_param.get("effort")
        )
        if effort is not None:
            reasoning_active = effort != "none"
        else:
            reasoning_active = _reasons_by_default(settings.model)

        # A single avior message can expand to several Responses input items
        # (an assistant turn with tool calls becomes a `message` item plus one
        # `function_call` item per call; a tool turn becomes one or more
        # `function_call_output` items), so the wire input is flat-mapped.
        wire_input: ResponseInputParam = []
        for m in messages:
            wire_input.extend(self._to_wire(m, reasoning_active=reasoning_active))

        # Encrypted reasoning content is returned only when asked for, and only
        # when reasoning is active for the request.
        include_param: list[ResponseIncludable] | Omit = (
            ["reasoning.encrypted_content"] if reasoning_active else omit
        )
        instructions_param: str | Omit = (
            system_prompt if system_prompt is not None else omit
        )
        temperature_param = self._resolve_temperature(
            settings,
            reasoning_active,
            warnings,
        )
        max_output_tokens_param: int | Omit = (
            settings.max_tokens if settings.max_tokens is not None else omit
        )
        tools_param: list[FunctionToolParam] | Omit = (
            [self._to_tool_param(t) for t in tools] if tools else omit
        )

        try:
            response = await self._client.responses.create(
                input=wire_input,
                instructions=instructions_param,
                model=settings.model,
                max_output_tokens=max_output_tokens_param,
                temperature=temperature_param,
                reasoning=reasoning_param,
                include=include_param,
                tools=tools_param,
                store=False,
            )
        except openai.APIStatusError as e:
            raise ProviderHTTPError(str(e), status_code=e.status_code) from e
        except openai.APIResponseValidationError as e:
            raise ProviderResponseValidationError(str(e)) from e
        except openai.APIConnectionError as e:
            raise ProviderConnectionError(str(e)) from e
        except openai.OpenAIError as e:
            raise ProviderError(str(e)) from e

        # An incomplete response (`status == "incomplete"`) may be truncated
        # mid-output, so a half-finished `function_call` can carry invalid-JSON
        # arguments.  Skip tool-call decoding - the call may be garbage, and
        # decoding it raises on partial JSON.  Text cannot raise, so it is
        # always collected.
        incomplete = response.status == "incomplete"

        # Text and tool calls are collected in output order into `parts`.
        # Refusals are kept in a separate list so that a refusal can override
        # `parts` (see below).
        parts: list[AssistantPart] = []
        refusal_parts: list[AssistantPart] = []
        for item in response.output:
            if isinstance(item, ResponseOutputMessage):
                for content in item.content:
                    match content:
                        case ResponseOutputText():
                            parts.append(TextPart(text=content.text))
                        case ResponseOutputRefusal():
                            refusal_parts.append(TextPart(text=content.refusal))
                        case _:
                            assert_never(content)

            elif isinstance(item, ResponseFunctionToolCall):
                if not incomplete:
                    parts.append(self._to_tool_call_part(item))

            elif isinstance(item, ResponseReasoningItem):
                parts.append(self._to_thinking_part(item))

            else:
                # An output item avior cannot represent yet (built-in tool
                # calls / results, etc.).  Fail loud rather than silently
                # dropping it and returning a misleading success.
                raise ProviderResponseValidationError(
                    "OpenAI returned an unsupported output item: "
                    f"{type(item).__name__}."
                )

        # When the response contains both text and a refusal (rare in non-
        # streaming), the refusal wins: it is the authoritative final word, and
        # the partial text is dropped.
        final_parts = refusal_parts or parts

        stop_reason = self._map_stop_reason(
            response,
            has_refusal=bool(refusal_parts),
            has_tool_call=any(isinstance(p, ToolCallPart) for p in parts),
        )
        if stop_reason == "error":
            # The canonical `"error"` reason drops the provider-specific cause;
            # log the status (and the provider's error message when present) so
            # an abnormal finish stays diagnosable.
            detail: str = response.status or "unknown status"
            if response.error is not None:
                detail = f"{detail}: {response.error.message}"
            logger.warning("OpenAI finished abnormally: %s", detail)

        raw_usage = (
            response.usage.model_dump(mode="json")
            if response.usage is not None
            else None
        )

        return ProviderResponse(
            message=AssistantMessage(
                parts=final_parts,
                stop_reason=stop_reason,
                provider_name=self.name,
            ),
            usage=self._map_usage(response.usage),
            raw_usage=raw_usage,
            response_id=response.id,
            model=response.model,
            provider_name=self.name,
            warnings=warnings,
        )

    def _resolve_reasoning(
        self,
        settings: ModelSettings,
        warnings: list[RunWarning],
    ) -> Reasoning | Omit:
        """Map the thinking settings to OpenAI's `reasoning` request parameter.

        Returns the `reasoning` parameter, or `omit` when it should not be sent.

        The raw `openai` provider options take precedence over the portable
        `thinking` setting; `OpenAIProviderOptions` documents how they combine.
        Without them, the portable setting maps to a `reasoning` config - see
        `_portable_reasoning`.
        """

        options = resolve_provider_options(
            settings,
            self.name,
            _OPENAI_OPTIONS_ADAPTER,
        )
        raw_reasoning = options.get("reasoning")
        if raw_reasoning is not None:
            return raw_reasoning
        else:
            return self._portable_reasoning(settings, warnings)

    def _portable_reasoning(
        self,
        settings: ModelSettings,
        warnings: list[RunWarning],
    ) -> Reasoning | Omit:
        """Map the portable `thinking` level to a `reasoning` config.

        Returns the `reasoning` parameter (or `omit`), appending an
        `UnsupportedSettingRunWarning` when a request is dropped.  `complete`
        documents the mapping and the drop conditions.
        """

        thinking = settings.thinking
        if thinking is None:
            return omit

        mode = _reasoning_mode(settings.model)
        if mode is None:
            # The model is not a recognized reasoning model.  A request to
            # enable (`True` / a level) is dropped and warned; the reason is
            # honest for both a model that genuinely does not reason (`gpt-4o`)
            # and one avior does not recognize, pointing to the `openai`
            # provider options rather than claiming the model cannot reason.
            # Disabling (`False`) is a harmless no-op.
            if thinking is not False:
                warnings.append(
                    self._thinking_dropped(
                        settings,
                        "the model is not a recognized reasoning model; if it "
                        "does reason, configure reasoning via the `openai` "
                        "provider options",
                    )
                )
            return omit

        elif thinking is False:
            match mode:
                case "off_by_default" | "on_by_default":
                    return Reasoning(effort="none")
                case "always_on":
                    # Reasoning cannot be turned off.
                    warnings.append(
                        self._thinking_dropped(
                            settings,
                            "the model always reasons and cannot be disabled",
                        )
                    )
                    return omit
                case _:
                    assert_never(mode)

        elif thinking is True:
            match mode:
                case "off_by_default":
                    # An `off_by_default` model needs an explicit effort to
                    # start reasoning.
                    return Reasoning(effort=_DEFAULT_EFFORT)
                case "on_by_default" | "always_on":
                    # The model already reasons at its own default depth.
                    return omit
                case _:
                    assert_never(mode)

        else:
            return Reasoning(effort=thinking)

    def _resolve_temperature(
        self,
        settings: ModelSettings,
        reasoning_active: bool,
        warnings: list[RunWarning],
    ) -> float | Omit:
        """Map `temperature`, dropping a value OpenAI would reject.

        OpenAI rejects a `temperature` other than `_DEFAULT_TEMPERATURE` when
        reasoning is active for the request (the `reasoning_active` argument).
        Such a value is dropped with an `UnsupportedSettingRunWarning`; the
        default `1` is forwarded and an unset value leaves the parameter unsent.
        """

        temperature = settings.temperature
        if temperature is None:
            return omit
        if temperature == _DEFAULT_TEMPERATURE:
            return temperature

        if reasoning_active:
            warnings.append(
                self._sampling_dropped(
                    settings,
                    "a custom temperature is not accepted while reasoning is active",
                )
            )
            return omit

        return temperature

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

    def _sampling_dropped(
        self,
        settings: ModelSettings,
        reason: str,
    ) -> UnsupportedSettingRunWarning:
        """Build the warning for a `temperature` that was dropped."""

        return UnsupportedSettingRunWarning(
            setting_name="temperature",
            setting_value=settings.temperature,
            reason=reason,
            provider=self.name,
            model=settings.model,
        )

    async def aclose(self) -> None:
        """Close the underlying SDK client when this provider owns it.

        No-op when the client was supplied by the caller via `client=` - its
        lifecycle belongs to whoever passed it in.  Safe to call more than once:
        `AsyncOpenAI.close` (and the httpx pool it delegates to) is itself
        idempotent.
        """

        if self._owns_client:
            await self._client.close()

    @staticmethod
    def _map_usage(usage: ResponseUsage | None) -> Usage | None:
        """Map OpenAI's `ResponseUsage` to the canonical `Usage`, or `None` when
        the response carries no usage (e.g. some incomplete responses).

        OpenAI's totals already include their sub-slices, so they map directly:

        - `input_tokens` / `output_tokens`: used as-is.
        - `cache_read_tokens`: from `input_tokens_details.cached_tokens`.
        - `cache_write_tokens`: `0` - OpenAI has no cache-write counter.
        - `reasoning_tokens`: from `output_tokens_details.reasoning_tokens`.

        avior's `total_tokens` is then derived (`input + output`) and equals
        OpenAI's own reported `total_tokens` (a mismatch is logged below).
        """

        if usage is None:
            return None

        mapped = Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.input_tokens_details.cached_tokens,
            reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
        )

        # OpenAI reports its own total, which should match the total our mapped
        # `Usage` derives.  If they ever diverge, warn rather than silently
        # shipping a total that disagrees with the provider (whose own number
        # stays in `raw_usage`).
        if usage.total_tokens != mapped.total_tokens:
            logger.warning(
                "OpenAI reported total_tokens=%d != avior's derived total %d; "
                "avior reports the derived total (provider's stays in raw_usage).",
                usage.total_tokens,
                mapped.total_tokens,
            )

        return mapped

    @staticmethod
    def _map_stop_reason(
        response: Response,
        *,
        has_refusal: bool,
        has_tool_call: bool,
    ) -> StopReason:
        """Map OpenAI Responses signals to a canonical `StopReason`.

        Resolved from `response.status` first, then - for a normal completion -
        from the response content.

        Status:

        - `"incomplete"`: resolved from `incomplete_details.reason`:
            - `"max_output_tokens"` -> `"max_tokens"` (truncated at the cap);
            - `"content_filter"` -> `"content_filter"` (blocked).

          A reason the SDK left unset falls through to the content
          classification below (the field is typed `Optional`, though OpenAI
          sets it in practice).
        - `"failed"` / `"cancelled"` / `"queued"` / `"in_progress"` ->
          `"error"`.  A terminal failure (`"failed"` / `"cancelled"`) or a
          non-terminal status a non-streaming `create` should not return
          (`"queued"` / `"in_progress"`): no usable result, surfaced rather
          than passed off as a successful empty stop.
        - `"completed"` / `None`: classified from the content below.

        Content (for a completed or reason-less incomplete response):

        - a `ResponseOutputRefusal` part -> `"refusal"` (the model declined);
        - one or more `function_call` items -> `"tool_use"` (the Responses API
          has no dedicated stop-reason field; their presence is the signal);
        - otherwise -> `"stop"`.

        Every `Response.status` is handled; an unknown value (added by a newer
        `openai`) trips `assert_never`.
        """

        match response.status:
            case "incomplete":
                details = response.incomplete_details
                if details is not None:
                    match details.reason:
                        case "max_output_tokens":
                            return "max_tokens"
                        case "content_filter":
                            return "content_filter"
                        case None:
                            pass
                        case _:
                            assert_never(details.reason)

            case "failed" | "cancelled" | "queued" | "in_progress":
                return "error"

            case "completed" | None:
                pass

            case _:
                assert_never(response.status)

        if has_refusal:
            return "refusal"

        if has_tool_call:
            return "tool_use"

        return "stop"

    @staticmethod
    def _to_tool_param(tool: Tool[Any, Any, Any]) -> FunctionToolParam:
        """Convert an avior `Tool` to a Responses function-tool definition.

        The tool's `args_model` JSON schema is sent as-is with `strict=False`:
        the schema guides the model but is not grammar-enforced, so the raw
        Pydantic schema (optional fields, defaults, open objects) is accepted
        unchanged.  This keeps the adapter symmetric with the Anthropic one;
        strict mode would require a lossy schema rewrite and is a separate
        opt-in.
        """

        return FunctionToolParam(
            type="function",
            name=tool.name,
            description=tool.description,
            parameters=tool.args_model.model_json_schema(),
            strict=False,
        )

    @staticmethod
    def _to_tool_call_part(item: ResponseFunctionToolCall) -> ToolCallPart:
        """Decode a Responses `function_call` output item into a `ToolCallPart`.

        The Responses API carries call arguments as a JSON string; it is parsed
        into the `dict` that `ToolCallPart.args` expects (an empty string maps
        to `{}`).  `call_id` (not `id`) is kept so the matching tool result can
        be correlated back on the next request.

        The item `id` is deliberately not kept, so `_to_wire` replays every
        `function_call` without an id.  OpenAI checks that a reasoning item
        precedes its tool calls only for items that carry an id; a replay
        without ids therefore passes that check even when a tool call has no
        reasoning item before it - `_to_reasoning_item_param` drops the item
        for a turn from another provider, for a request whose reasoning is
        off, or for a part with no token to echo.  The cost: OpenAI cannot
        flag a broken pairing that avior itself produced.
        """

        try:
            args: dict[str, Any] = json.loads(item.arguments) if item.arguments else {}
        except json.JSONDecodeError as e:
            raise ProviderResponseValidationError(
                f"OpenAI returned tool-call arguments that are not valid JSON: {e}"
            ) from e

        return ToolCallPart(
            call_id=item.call_id,
            tool_name=item.name,
            args=args,
        )

    @staticmethod
    def _to_thinking_part(item: ResponseReasoningItem) -> ThinkingPart:
        """Decode a Responses reasoning item into a `ThinkingPart`.

        The summary entries (if any) join into the readable content.  The
        reasoning item's `id` and `encrypted_content` are kept in
        `provider_details` so the item can be replayed before its tool call on
        the next request, which OpenAI requires under stateless operation.
        `encrypted_content` is present only when it was requested and the model
        supports it; it is stored when present.
        """

        content = "".join(summary.text for summary in item.summary)

        provider_details: dict[str, JsonValue] = {"reasoning_id": item.id}
        if item.encrypted_content is not None:
            provider_details["encrypted_content"] = item.encrypted_content

        return ThinkingPart(content=content, provider_details=provider_details)

    def _to_wire(
        self,
        message: Message,
        *,
        reasoning_active: bool,
    ) -> list[ResponseInputItemParam]:
        """Convert an avior `Message` to Responses input items.

        Returns a list because, unlike a chat-style wire format, the Responses
        API carries reasoning, tool calls, and tool results as their own
        top-level items rather than nested in a message:

        - `UserMessage` -> a single `user` message item.
        - `AssistantMessage` -> its parts, in order, each mapped to a wire item:

          - a text part -> a `message` item;
          - a reasoning step -> a `reasoning` item;
          - a tool call -> a `function_call` item.

          A reasoning item is echoed back only to the same provider, and only
          when reasoning is active for the request (`reasoning_active` is true).
        - `ToolMessage` -> one `function_call_output` item per result, matched
          to its call by `call_id`.  The Responses API has no error flag on a
          tool output, so an error result is sent as its text content (the
          status distinction is carried only in that text).
        """

        match message:
            case UserMessage():
                return [
                    EasyInputMessageParam(
                        role="user",
                        type="message",
                        content=message.text or "",
                    )
                ]

            case AssistantMessage():
                items: list[ResponseInputItemParam] = []

                # Emit the parts in order so a reasoning item keeps its place
                # immediately before the item it informed.
                for part in message.parts:
                    match part:
                        case TextPart():
                            items.append(
                                EasyInputMessageParam(
                                    role="assistant",
                                    type="message",
                                    content=part.text,
                                )
                            )

                        case ThinkingPart():
                            reasoning_item = self._to_reasoning_item_param(
                                message,
                                part,
                                reasoning_active=reasoning_active,
                            )
                            if reasoning_item is not None:
                                items.append(reasoning_item)

                        case ToolCallPart():
                            # No item `id` is sent: `_to_tool_call_part` does
                            # not keep the id, and its docstring explains why
                            # that is deliberate.
                            items.append(
                                ResponseFunctionToolCallParam(
                                    type="function_call",
                                    call_id=part.call_id,
                                    name=part.tool_name,
                                    arguments=json.dumps(part.args),
                                )
                            )

                        case _:
                            assert_never(part)

                return items

            case ToolMessage():
                return [
                    FunctionCallOutput(
                        type="function_call_output",
                        call_id=p.call_id,
                        output=p.result.content,
                    )
                    for p in message.parts
                ]

            case _:
                assert_never(message)

    def _to_reasoning_item_param(
        self,
        message: AssistantMessage,
        part: ThinkingPart,
        *,
        reasoning_active: bool,
    ) -> ResponseReasoningItemParam | None:
        """Build the wire reasoning item to echo a reasoning step, or `None`.

        A reasoning item round-trips only to the provider that produced it and
        only when reasoning is active for the request: the token is provider-
        and model-specific, and OpenAI rejects a foreign one.  Returns `None` -
        dropping the part - when the turn came from a different provider,
        reasoning is not active for the request (`reasoning_active` is false),
        or the part carries no token to echo.
        """

        if message.provider_name != self.name:
            return None
        if not reasoning_active:
            return None

        details = part.provider_details or {}
        reasoning_id = details.get("reasoning_id")
        encrypted_content = details.get("encrypted_content")
        if not isinstance(reasoning_id, str) or not isinstance(encrypted_content, str):
            return None

        return ResponseReasoningItemParam(
            id=reasoning_id,
            type="reasoning",
            summary=[],
            encrypted_content=encrypted_content,
        )
