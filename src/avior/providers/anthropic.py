"""Native Anthropic provider adapter.

Wraps `anthropic.AsyncAnthropic` and implements `Provider` against Anthropic's
Messages API.

Install via the optional extra: `pip install avior[anthropic]`.
"""

import logging
from collections.abc import Sequence
from typing import Any, assert_never, cast

try:
    import anthropic
    from anthropic import AsyncAnthropic, Omit, omit
    from anthropic.types import Message as AnthropicMessage
    from anthropic.types import (
        MessageParam,
        RedactedThinkingBlock,
        RedactedThinkingBlockParam,
        TextBlock,
        TextBlockParam,
        ThinkingBlock,
        ThinkingBlockParam,
        ToolParam,
        ToolResultBlockParam,
        ToolUseBlock,
        ToolUseBlockParam,
    )
    from anthropic.types import Usage as AnthropicUsage
except ImportError as e:
    raise ImportError(
        "The `anthropic` package is required to use `avior.providers.anthropic`. "
        "Install with: pip install avior[anthropic]"
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
from avior.core.provider import ModelSettings, Provider, ProviderResponse
from avior.core.tools import Tool
from avior.core.usage import Usage

logger = logging.getLogger(__name__)

# The Anthropic SDK refuses a non-streaming request it estimates could run past
# a 10-minute limit, assuming a rate of 128_000 output tokens per hour.  Scaling
# that rate to the 10-minute limit gives the largest value it serves without
# streaming: 128_000 * 10 // 60 = 21333.  These mirror literals inside the
# Anthropic SDK's non-streaming guard, which does not expose them as reusable
# constants; if that ceiling ever drops below this, the guard raises (surfaced
# as `ProviderError`) rather than failing silently.
_NONSTREAMING_LIMIT_MINUTES = 10
_MINUTES_PER_HOUR = 60
_TOKENS_PER_HOUR = 128_000

_MAX_NONSTREAMING_TOKENS = (
    _TOKENS_PER_HOUR * _NONSTREAMING_LIMIT_MINUTES // _MINUTES_PER_HOUR
)
"""`max_tokens` used when `ModelSettings.max_tokens` is `None`.

Anthropic's Messages API requires `max_tokens`, so an unset value defaults to
the largest output the Anthropic SDK serves without streaming.  This is below
the model's true maximum for models that can emit more with streaming.
"""


class AnthropicProvider(Provider):
    """Async adapter to Anthropic's Messages API.

    Translates avior's canonical `Message` shape to and from Anthropic's wire
    format.  Exceptions from the Anthropic SDK are translated to avior's
    provider-agnostic hierarchy (`ProviderError` and subclasses), with the
    original exception preserved as `__cause__`.
    """

    def __init__(
        self,
        *,
        client: AsyncAnthropic | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the provider.

        Args:
            client: A pre-built `AsyncAnthropic` instance.  Takes precedence
                over `api_key` if both are supplied.  Lifecycle stays with the
                caller; `aclose` will not close it.
            api_key: API key for a freshly constructed `AsyncAnthropic`.  If
                both `client` and `api_key` are `None`, `AsyncAnthropic` reads
                `ANTHROPIC_API_KEY` from the environment.  A self-constructed
                client is closed by `aclose`.
        """

        super().__init__()
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = AsyncAnthropic(api_key=api_key)
            self._owns_client = True

    @property
    def name(self) -> str:
        """The provider's canonical name."""

        return "anthropic"

    async def complete(
        self,
        messages: Sequence[Message],
        settings: ModelSettings,
        *,
        tools: Sequence[Tool[Any, Any, Any]] = (),
        system_prompt: str | None = None,
    ) -> ProviderResponse:
        """Send the conversation to Claude and return the response.

        `max_tokens` falls back to the model's maximum non-streaming output when
        `settings.max_tokens is None`; `temperature` is forwarded only when
        explicitly set on `settings`.

        Args:
            messages: Conversation transcript (user / assistant / tool turns).
            settings: Per-call invocation settings.
            tools: Tools to offer the model.  Each is sent as an Anthropic tool
                with its arguments JSON schema as `input_schema`; `tool_use`
                blocks in the response are parsed back into `ToolCallPart`s.
            system_prompt: The system prompt, sent as a text block in
                Anthropic's top-level `system` parameter, or `None` for no
                system prompt.  Pass `None`, not a blank string - Anthropic
                rejects an empty or whitespace-only text block.

        Returns:
            A `ProviderResponse` wrapping the assistant message together with
            the call metadata.

        Raises:
            ProviderHTTPError: The provider returned a 4xx or 5xx HTTP response.
                `status_code` carries the wire status.
            ProviderResponseValidationError: The provider returned a successful
                response that could not be decoded (typically an outdated
                `anthropic` package), or whose decoded content avior cannot
                accept as a finished assistant turn (for example an unsupported
                content block, or a turn Anthropic paused for continuation).
            ProviderConnectionError: Network-level failure (DNS / TCP / TLS /
                timeout) - no HTTP response was received.
            ProviderError: A request the Anthropic SDK refuses to send without
                streaming (a large `max_tokens` risking the 10-minute
                non-streaming limit), or any other unexpected failure from the
                Anthropic SDK.

        Errors translated from an Anthropic SDK exception preserve it as
        `__cause__`; validation errors avior detects in an otherwise successful
        response are raised directly.
        """

        logger.debug("complete: model=%s, messages=%d", settings.model, len(messages))

        wire_messages = [
            wire_message
            for m in messages
            if (wire_message := self._to_wire(m)) is not None
        ]

        system_param: list[TextBlockParam] | Omit = (
            [TextBlockParam(type="text", text=system_prompt)]
            if system_prompt is not None
            else omit
        )
        temperature_param: float | Omit = (
            settings.temperature if settings.temperature is not None else omit
        )
        max_tokens = (
            settings.max_tokens
            if settings.max_tokens is not None
            else _MAX_NONSTREAMING_TOKENS
        )
        tools_param: list[ToolParam] | Omit = (
            [self._to_tool_param(t) for t in tools] if tools else omit
        )

        try:
            response = await self._client.messages.create(
                messages=wire_messages,
                system=system_param,
                model=settings.model,
                max_tokens=max_tokens,
                temperature=temperature_param,
                tools=tools_param,
            )
        except anthropic.APIStatusError as e:
            raise ProviderHTTPError(str(e), status_code=e.status_code) from e
        except anthropic.APIResponseValidationError as e:
            raise ProviderResponseValidationError(str(e)) from e
        except anthropic.APIConnectionError as e:
            raise ProviderConnectionError(str(e)) from e
        except ValueError as e:
            # The Anthropic SDK guards client-side: a non-streaming request
            # whose `max_tokens` risks exceeding the 10-minute limit raises a
            # plain `ValueError` (not an `AnthropicError`) before any network
            # call.  Translate it into the provider hierarchy with an actionable
            # message rather than leaking a raw `ValueError`.  Re-raise any
            # other `ValueError`.
            if "Streaming is required" in str(e):
                raise ProviderError(
                    "Anthropic requires streaming for this request:  "
                    f"max_tokens ({max_tokens}) risks exceeding the 10-minute "
                    "non-streaming limit, and avior has no streaming support "
                    "yet.  Lower max_tokens."
                ) from e
            else:
                raise
        except anthropic.AnthropicError as e:
            raise ProviderError(str(e)) from e

        parts: list[AssistantPart] = []
        for block in response.content:
            if isinstance(block, TextBlock):
                parts.append(TextPart(text=block.text))

            elif isinstance(block, ToolUseBlock):
                parts.append(
                    ToolCallPart(
                        call_id=block.id,
                        tool_name=block.name,
                        args=cast(dict[str, Any], block.input),
                    )
                )

            elif isinstance(block, ThinkingBlock):
                # The `signature` is kept in `provider_details` so it can be
                # echoed back unchanged on a later turn (Anthropic checks it on
                # replay).
                parts.append(
                    ThinkingPart(
                        content=block.thinking,
                        provider_details={"signature": block.signature},
                    )
                )

            elif isinstance(block, RedactedThinkingBlock):
                # Encrypted reasoning with no readable text; keep the opaque
                # blob to echo back, with empty `content`.
                parts.append(
                    ThinkingPart(
                        content="",
                        provider_details={"redacted_data": block.data},
                    )
                )

            else:
                # A block avior cannot represent yet (server tool use / results,
                # container uploads, ...).  avior does not enable the features
                # that produce these, so failing loud beats silently dropping
                # content and returning a misleading success.
                raise ProviderResponseValidationError(
                    "Anthropic returned an unsupported content block: "
                    f"{type(block).__name__}."
                )

        stop_reason = self._map_stop_reason(response)

        # `stop_reason="tool_use"` with no decoded tool call would hand the
        # `Runner` an empty turn that reads as a final answer; surface it
        # instead.
        if stop_reason == "tool_use" and not any(
            isinstance(p, ToolCallPart) for p in parts
        ):
            raise ProviderResponseValidationError(
                "Anthropic reported stop_reason='tool_use' but decoded no tool call."
            )

        return ProviderResponse(
            message=AssistantMessage(
                parts=parts,
                stop_reason=stop_reason,
                provider_name=self.name,
            ),
            usage=self._map_usage(response.usage),
            raw_usage=response.usage.model_dump(mode="json"),
            response_id=response.id,
            model=response.model,
            provider_name=self.name,
        )

    async def aclose(self) -> None:
        """Close the underlying SDK client when this provider owns it.

        No-op when the client was supplied by the caller via `client=` - its
        lifecycle belongs to whoever passed it in.  Safe to call more than once:
        `AsyncAnthropic.close` (and the httpx pool it delegates to) is itself
        idempotent.
        """

        if self._owns_client:
            await self._client.close()

    @staticmethod
    def _map_usage(usage: AnthropicUsage) -> Usage:
        """Map `AnthropicUsage` to the canonical `Usage`.

        - `input_tokens`: Anthropic reports the *non-cached* input only, with
          cache reads / creation separate and additional; avior's `input_tokens`
          includes its cache sub-slices, so they are folded back in.
        - `cache_read_tokens` / `cache_write_tokens`: Anthropic's cache reads /
          cache creation; `None` (unused cache) is coalesced to `0`.
        - `reasoning_tokens`: stays `None` - Anthropic does not itemize
          extended thinking out of `output_tokens`, so the count is genuinely
          unknown, not `0`.

        The provider-native usage (e.g. `server_tool_use`, cache-write TTL
        split) is preserved on `ProviderResponse.raw_usage`.
        """

        cache_read = usage.cache_read_input_tokens or 0
        cache_write = usage.cache_creation_input_tokens or 0
        return Usage(
            input_tokens=usage.input_tokens + cache_read + cache_write,
            output_tokens=usage.output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )

    @staticmethod
    def _map_stop_reason(response: AnthropicMessage) -> StopReason:
        """Map Anthropic's `stop_reason` to canonical `StopReason`.

        - `"tool_use"` -> `"tool_use"` (the model requested tool calls).
        - `"max_tokens"` -> `"max_tokens"` (output truncated at the cap).
        - `"refusal"` -> `"refusal"` (the model itself declined).
        - `"end_turn"` / `"stop_sequence"` / `None` -> `"stop"` (normal
          completion).

        `"pause_turn"` is rejected: it marks a turn Anthropic paused mid-flight
        (long-running server tools), to be resumed by sending the partial
        assistant content back unchanged.  avior has no continuation path, so
        treating it as a normal stop would surface a half-finished turn as the
        final answer.

        Every known `stop_reason` is handled; an unknown value (added by a
        newer `anthropic`) trips `assert_never`, both statically and at runtime,
        so it gets an explicit mapping instead of a silent default.
        """

        match response.stop_reason:
            case "tool_use":
                return "tool_use"
            case "max_tokens":
                return "max_tokens"
            case "refusal":
                return "refusal"
            case "end_turn" | "stop_sequence" | None:
                return "stop"
            case "pause_turn":
                raise ProviderResponseValidationError(
                    "Anthropic paused the turn (stop_reason='pause_turn'), "
                    "which requires resuming the request to finish.  avior does "
                    "not support continuation yet."
                )
            case _:
                assert_never(response.stop_reason)

    @staticmethod
    def _to_tool_param(tool: Tool[Any, Any, Any]) -> ToolParam:
        """Convert an avior `Tool` to an Anthropic tool definition."""

        return ToolParam(
            name=tool.name,
            description=tool.description,
            input_schema=tool.args_model.model_json_schema(),
        )

    def _to_wire(self, message: Message) -> MessageParam | None:
        """Convert an avior `Message` to an Anthropic `MessageParam`, or `None`.

        Maps each message type to Anthropic's wire shape:

        - `UserMessage` -> a `user` turn of text blocks.
        - `AssistantMessage` -> an `assistant` turn; text parts become text
          blocks and tool calls become `tool_use` blocks.  A reasoning step is
          echoed back unchanged as a `thinking` / `redacted_thinking` block when
          this provider produced the turn, and dropped otherwise.
        - `ToolMessage` -> a `user` turn of `tool_result` blocks (Anthropic
          carries tool results in the user role).

        Returns `None` for an assistant turn whose parts all drop out (only
        reasoning steps, none of them echoable): it would serialize to an empty
        turn, which Anthropic rejects, so it is omitted from the request.
        """

        match message:
            case UserMessage():
                user_content: list[TextBlockParam] = [
                    TextBlockParam(type="text", text=p.text) for p in message.parts
                ]
                return MessageParam(role="user", content=user_content)

            case AssistantMessage():
                asst_content: list[
                    TextBlockParam
                    | ToolUseBlockParam
                    | ThinkingBlockParam
                    | RedactedThinkingBlockParam
                ] = []
                for part in message.parts:
                    match part:
                        case TextPart():
                            asst_content.append(
                                TextBlockParam(type="text", text=part.text)
                            )
                        case ToolCallPart():
                            # Cast our `dict[str, JsonValue]` to the SDK's wider
                            # `dict[str, object]`: `dict` is invariant in its
                            # value, and every `JsonValue` is a Python `object`.
                            asst_content.append(
                                ToolUseBlockParam(
                                    type="tool_use",
                                    id=part.call_id,
                                    name=part.tool_name,
                                    input=cast(dict[str, object], part.args),
                                )
                            )
                        case ThinkingPart():
                            block = self._to_thinking_block_param(message, part)
                            if block is not None:
                                asst_content.append(block)
                        case _:
                            assert_never(part)

                # A turn left empty after dropping non-echoable reasoning steps
                # carries nothing to send; omit it, since Anthropic rejects an
                # empty turn.
                if not asst_content:
                    return None

                return MessageParam(role="assistant", content=asst_content)

            case ToolMessage():
                tool_content: list[ToolResultBlockParam] = [
                    ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=p.call_id,
                        content=p.result.content,
                        is_error=p.result.status == "error",
                    )
                    for p in message.parts
                ]
                return MessageParam(role="user", content=tool_content)

            case _:
                assert_never(message)

    def _to_thinking_block_param(
        self,
        message: AssistantMessage,
        part: ThinkingPart,
    ) -> ThinkingBlockParam | RedactedThinkingBlockParam | None:
        """Build the wire block to echo a reasoning step, or `None` to drop it.

        A reasoning block round-trips only to the provider that produced it: the
        opaque token is provider-specific, and Anthropic rejects a foreign or
        modified block.  Returns `None` - dropping the part - when the turn came
        from a different provider, or carries no token to echo.
        """

        if message.provider_name != self.name:
            return None

        details = part.provider_details or {}
        if "redacted_data" in details:
            return RedactedThinkingBlockParam(
                type="redacted_thinking",
                data=cast(str, details["redacted_data"]),
            )
        elif "signature" in details:
            return ThinkingBlockParam(
                type="thinking",
                thinking=part.content,
                signature=cast(str, details["signature"]),
            )
        else:
            return None
