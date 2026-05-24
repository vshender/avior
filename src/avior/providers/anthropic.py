"""Native Anthropic provider adapter.

Wraps `anthropic.AsyncAnthropic` and implements `Provider` against Anthropic's
Messages API.

Install via the optional extra: `pip install avior[anthropic]`.
"""

import logging
from typing import Literal, assert_never

try:
    import anthropic
    from anthropic import AsyncAnthropic, Omit, omit
    from anthropic.types import Message as AnthropicMessage
    from anthropic.types import MessageParam, TextBlock, TextBlockParam
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
from avior.core.messages import Message, StopReason, TextPart
from avior.core.provider import ModelSettings, Provider

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 4096
"""Fallback `max_tokens` when `ModelSettings.max_tokens` is `None`."""

type _AnthropicRole = Literal["user", "assistant"]
"""Anthropic's per-message role union.

Anthropic's Messages API places system instructions in a top-level `system`
parameter, so the per-message `role` is narrower than avior's canonical `Role`
(which includes `"system"`).
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

    async def complete(
        self,
        messages: list[Message],
        settings: ModelSettings,
    ) -> Message:
        """Send `messages` to Claude and return the assistant's response.

        `max_tokens` falls back to 4096 when `settings.max_tokens is None`;
        `temperature` is forwarded only if explicitly set on `settings`.

        Args:
            messages: Conversation transcript.  `system`-role messages are
                lifted out of the transcript and passed as separate text blocks
                in Anthropic's top-level `system` parameter; relative order
                among non-`system` messages is preserved, but `system`-message
                positioning is not (Anthropic's wire format does not support
                per-position system instructions).
            settings: Per-call invocation settings.

        Returns:
            The assistant response message.

        Raises:
            ProviderHTTPError: The provider returned a 4xx or 5xx HTTP response.
                `status_code` carries the wire status.
            ProviderResponseValidationError: The provider returned a successful
                response whose body could not be decoded (typically indicates
                an outdated `anthropic` package).
            ProviderConnectionError: Network-level failure (DNS / TCP / TLS /
                timeout) - no HTTP response was received.
            ProviderError: Any other unexpected failure from the Anthropic
                SDK, preserved as `__cause__`.
        """

        logger.debug("complete: model=%s, messages=%d", settings.model, len(messages))

        system, conversation = self._extract_system(messages)
        wire_messages = [self._to_wire(m) for m in conversation]

        system_param: list[TextBlockParam] | Omit = (
            system if system is not None else omit
        )
        temperature_param: float | Omit = (
            settings.temperature if settings.temperature is not None else omit
        )
        max_tokens = (
            settings.max_tokens
            if settings.max_tokens is not None
            else _DEFAULT_MAX_TOKENS
        )

        try:
            response = await self._client.messages.create(
                messages=wire_messages,
                system=system_param,
                model=settings.model,
                max_tokens=max_tokens,
                temperature=temperature_param,
            )
        except anthropic.APIStatusError as e:
            raise ProviderHTTPError(str(e), status_code=e.status_code) from e
        except anthropic.APIResponseValidationError as e:
            raise ProviderResponseValidationError(str(e)) from e
        except anthropic.APIConnectionError as e:
            raise ProviderConnectionError(str(e)) from e
        except anthropic.AnthropicError as e:
            raise ProviderError(str(e)) from e

        parts: list[TextPart] = [
            TextPart(text=block.text)
            for block in response.content
            if isinstance(block, TextBlock)
        ]
        stop_reason = self._map_stop_reason(response)
        return Message(role="assistant", parts=parts, stop_reason=stop_reason)

    @staticmethod
    def _map_stop_reason(response: AnthropicMessage) -> StopReason:
        """Map Anthropic's `stop_reason` to canonical `StopReason`.

        - `"max_tokens"` -> `"max_tokens"` (output truncated at the cap).
        - `"refusal"` -> `"refusal"` (the model itself declined; the
          refusal text is present in `content` and lands in `parts`).
        - Anything else (`"end_turn"`, `"stop_sequence"`, `"tool_use"`,
          `"pause_turn"`, `None`) -> `"stop"`; the orchestrator treats the
          response as normal completion and decides next steps from `parts`.
        """

        match response.stop_reason:
            case "max_tokens":
                return "max_tokens"
            case "refusal":
                return "refusal"
            case _:
                return "stop"

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
    def _extract_system(
        messages: list[Message],
    ) -> tuple[list[TextBlockParam] | None, list[Message]]:
        """Pull all `system` messages out of the conversation.

        Anthropic's Messages API does not accept `system` messages in the
        `messages` array; the canonical IR allows them anywhere, so the adapter
        collects them as separate `TextBlockParam`s for Anthropic's top-level
        `system` parameter.  Empty `system` messages are skipped.

        Returns `(blocks, rest)`; `blocks` is `None` when no non-empty `system`
        message is present.
        """

        system_blocks: list[TextBlockParam] = []
        rest: list[Message] = []
        for msg in messages:
            if msg.role == "system":
                if msg.text:
                    system_blocks.append(TextBlockParam(type="text", text=msg.text))
            else:
                rest.append(msg)

        return (system_blocks or None), rest

    @staticmethod
    def _to_wire(message: Message) -> MessageParam:
        """Convert an avior `Message` to an Anthropic `MessageParam`.

        System messages are not accepted - they are extracted into the
        top-level `system` parameter by `_extract_system` upstream.
        """

        assert message.role != "system", (
            "System messages must be filtered out by `_extract_system` upstream."
        )

        match message.role:
            case "user" | "assistant":
                role: _AnthropicRole = message.role
            case _:
                assert_never(message.role)

        content: list[TextBlockParam] = [
            TextBlockParam(type="text", text=p.text) for p in message.parts
        ]
        return MessageParam(role=role, content=content)
