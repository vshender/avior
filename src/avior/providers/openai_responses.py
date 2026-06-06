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
from typing import Any, assert_never

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
        ResponseInputItemParam,
        ResponseInputParam,
        ResponseOutputMessage,
        ResponseOutputRefusal,
        ResponseOutputText,
        ResponseUsage,
    )
    from openai.types.responses.response_input_item_param import FunctionCallOutput
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
    SystemMessage,
    TextPart,
    ToolCallPart,
    ToolMessage,
    UserMessage,
)
from avior.core.provider import ModelSettings, Provider, ProviderResponse
from avior.core.tools import Tool
from avior.core.usage import Usage

logger = logging.getLogger(__name__)


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

    async def complete(
        self,
        messages: Sequence[Message],
        settings: ModelSettings,
        tools: Sequence[Tool[Any, Any]] = (),
    ) -> ProviderResponse:
        """Send `messages` to OpenAI Responses API and return the assistant's
        response.

        `store=False` is always passed (stateless wire; no server-side
        history).  `temperature` and `max_output_tokens` are forwarded only
        when explicitly set on `settings`.

        Args:
            messages: Conversation transcript.  `SystemMessage`s are lifted
                out of the transcript and joined (newline-separated) into
                OpenAI Responses' top-level `instructions` parameter (see
                `_extract_instructions` for the rationale of this choice over
                inline `role='system'` items).  Relative order is preserved
                within each group (system messages keep their order in the
                joined string; non-system messages keep their order in
                `input`), but the interleaving between the two groups is lost.
            settings: Per-call invocation settings.
            tools: Tools to offer the model.  Each is sent as a Responses
                function tool with its arguments JSON schema as `parameters`
                (`strict=False`: the schema guides the model but is not
                grammar-enforced).  `function_call` items in the response are
                parsed back into `ToolCallPart`s.

        Returns:
            A `ProviderResponse` wrapping the assistant message together with
            the call metadata.

        Raises:
            ProviderHTTPError: The provider returned a 4xx or 5xx HTTP response.
                `status_code` carries the wire status.
            ProviderResponseValidationError: The provider returned a successful
                response whose body could not be decoded (typically indicates
                an outdated `openai` package).
            ProviderConnectionError: Network-level failure (DNS / TCP / TLS /
                timeout) - no HTTP response was received.
            ProviderError: Any other unexpected failure from the OpenAI SDK,
                preserved as `__cause__`.
        """

        logger.debug("complete: model=%s, messages=%d", settings.model, len(messages))

        instructions, conversation = self._extract_instructions(messages)

        # A single avior message can expand to several Responses input items
        # (an assistant turn with tool calls becomes a `message` item plus one
        # `function_call` item per call; a tool turn becomes one or more
        # `function_call_output` items), so the wire input is flat-mapped.
        wire_input: ResponseInputParam = []
        for m in conversation:
            wire_input.extend(self._to_wire(m))

        instructions_param: str | Omit = (
            instructions if instructions is not None else omit
        )
        temperature_param: float | Omit = (
            settings.temperature if settings.temperature is not None else omit
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

            elif isinstance(item, ResponseFunctionToolCall):
                if not incomplete:
                    parts.append(self._to_tool_call_part(item))

        # When the response contains both text and a refusal (rare in non-
        # streaming), the refusal wins: it is the authoritative final word, and
        # the partial text is dropped.
        final_parts = refusal_parts or parts

        stop_reason = self._map_stop_reason(
            response,
            has_refusal=bool(refusal_parts),
            has_tool_call=any(isinstance(p, ToolCallPart) for p in parts),
        )
        raw_usage = (
            response.usage.model_dump(mode="json")
            if response.usage is not None
            else None
        )

        return ProviderResponse(
            message=AssistantMessage(parts=final_parts, stop_reason=stop_reason),
            usage=self._map_usage(response.usage),
            raw_usage=raw_usage,
            response_id=response.id,
            model=response.model,
            provider_name="openai",
        )

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

        Channels are checked in order; the first match wins:

        1. `status == "incomplete"` with `incomplete_details.reason`:
           - `"max_output_tokens"` -> `"max_tokens"` (truncated at the cap);
           - `"content_filter"` -> `"content_filter"` (blocked).

           An incomplete response whose `reason` the SDK left unset falls
           through (the field is typed `Optional`, though OpenAI sets it in
           practice).
        2. A `ResponseOutputRefusal` part on a completed response ->
           `"refusal"` (the model itself declined).
        3. One or more `function_call` items -> `"tool_use"`.  The Responses
           API has no dedicated stop-reason field for this; their presence is
           the signal.
        4. Anything else -> `"stop"`: a normal completion.  The caller then
           decides what to do from `parts` alone.
        """

        if response.status == "incomplete":
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

        if has_refusal:
            return "refusal"

        if has_tool_call:
            return "tool_use"

        return "stop"

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
    def _to_tool_param(tool: Tool[Any, Any]) -> FunctionToolParam:
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
    def _extract_instructions(
        messages: Sequence[Message],
    ) -> tuple[str | None, list[UserMessage | AssistantMessage | ToolMessage]]:
        """Pull all `SystemMessage`s out of the conversation.

        System content is lifted to the top-level `instructions` parameter
        rather than passed inline as `role='system'` items in `input` (the API
        supports both shapes).  The top-level choice
        - keeps prompt-cache prefixes stable
        - lets OpenAI fold `instructions` to the `developer` role automatically
          for reasoning models (no per-model branching in the adapter)

        `instructions` accepts a single string only, so multiple system
        messages are collected (newline-separated) into one string.  Empty
        system messages are skipped.

        Returns `(instructions, rest)`; `instructions` is `None` when no
        non-empty system message is present.
        """

        texts: list[str] = []
        rest: list[UserMessage | AssistantMessage | ToolMessage] = []
        for msg in messages:
            match msg:
                case SystemMessage():
                    if msg.text:
                        texts.append(msg.text)
                case UserMessage() | AssistantMessage() | ToolMessage():
                    rest.append(msg)
                case _:
                    assert_never(msg)

        return ("\n\n".join(texts) if texts else None), rest

    @staticmethod
    def _to_wire(
        message: UserMessage | AssistantMessage | ToolMessage,
    ) -> list[ResponseInputItemParam]:
        """Convert an avior non-system `Message` to Responses input items.

        `SystemMessage`s are not accepted - they are extracted into the
        top-level `instructions` parameter by `_extract_instructions`
        upstream.

        Returns a list because, unlike a chat-style wire format, the Responses
        API carries tool calls and tool results as their own top-level items
        rather than nested in a message:

        - `UserMessage` -> a single `user` message item.
        - `AssistantMessage` -> a single `message` item with the assistant's
          text (omitted when the turn is tool calls only), followed by one
          `function_call` item per tool call in order.  The tool calls keep
          their order, but the text is always hoisted ahead of them.
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

                if message.text:
                    items.append(
                        EasyInputMessageParam(
                            role="assistant",
                            type="message",
                            content=message.text,
                        )
                    )

                for part in message.parts:
                    if isinstance(part, ToolCallPart):
                        items.append(
                            ResponseFunctionToolCallParam(
                                type="function_call",
                                call_id=part.call_id,
                                name=part.tool_name,
                                arguments=json.dumps(part.args),
                            )
                        )

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
