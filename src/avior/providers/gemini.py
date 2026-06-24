"""Native Gemini provider adapter.

Wraps `google.genai.Client` and implements `Provider` against the Gemini API
(`generate_content`).

Install via the optional extra: `pip install avior[gemini]`.
"""

import logging
import uuid
from collections.abc import Callable, Sequence
from typing import Any, assert_never

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
    ToolCallPart,
    ToolMessage,
    UserMessage,
)
from avior.core.provider import ModelSettings, Provider, ProviderResponse
from avior.core.tools import Tool
from avior.core.usage import Usage

logger = logging.getLogger(__name__)


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

    Known limitation - thought signatures: thinking-enabled Gemini models (e.g.
    `gemini-2.5-flash`) attach a `thought_signature` to function-call parts that
    the API expects sent back unchanged on the next request, especially for
    multi-step tool use.  The canonical IR has no slot for it yet, so the
    signature is dropped on the round trip.  A single tool round-trip (one call,
    its result, then a final answer) still works, but a chain of several tool
    calls across requests on these models may be rejected.
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

    async def complete(
        self,
        messages: Sequence[Message],
        settings: ModelSettings,
        *,
        tools: Sequence[Tool[Any, Any, Any]] = (),
        system_prompt: str | None = None,
    ) -> ProviderResponse:
        """Send the conversation to Gemini and return the response.

        `temperature` and `max_output_tokens` are forwarded only when
        explicitly set on `settings`.

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
        wire_messages = [self._to_wire(m, call_names) for m in messages]

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
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=settings.max_tokens,
            temperature=settings.temperature,
            tools=tools_param,
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
                # Thought summaries (only present when thinking is configured to
                # include them) are not the answer; skip them from the text.
                if part.thought:
                    continue

                if part.text is not None:
                    parts.append(TextPart(text=part.text))

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
            message=AssistantMessage(parts=parts, stop_reason=stop_reason),
            usage=self._map_usage(response.usage_metadata),
            raw_usage=(
                response.usage_metadata.model_dump(mode="json")
                if response.usage_metadata is not None
                else None
            ),
            response_id=response.response_id,
            model=response.model_version,
            provider_name="gemini",
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

    @staticmethod
    def _to_wire(message: Message, call_names: dict[str, str]) -> types.Content:
        """Convert an avior `Message` to a Gemini `Content`.

        Maps each message type to Gemini's wire shape:

        - `UserMessage` -> a `"user"` turn of text parts.
        - `AssistantMessage` -> a `"model"` turn; text parts become text parts
          and tool calls become `function_call` parts.
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
        """

        match message:
            case UserMessage():
                return types.Content(
                    role="user",
                    parts=[types.Part(text=p.text) for p in message.parts],
                )

            case AssistantMessage():
                asst_parts: list[types.Part] = []
                for part in message.parts:
                    match part:
                        case TextPart():
                            asst_parts.append(types.Part(text=part.text))
                        case ToolCallPart():
                            asst_parts.append(
                                types.Part(
                                    function_call=types.FunctionCall(
                                        id=part.call_id,
                                        name=part.tool_name,
                                        args=part.args,
                                    )
                                )
                            )
                        case _:
                            assert_never(part)

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
