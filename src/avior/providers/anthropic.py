"""Native Anthropic provider adapter.

Wraps `anthropic.AsyncAnthropic` and implements the `Provider` protocol against
Anthropic's Messages API.

Install via the optional extra: `pip install avior[anthropic]`.
"""

from typing import Literal, assert_never

try:
    import anthropic
    from anthropic import AsyncAnthropic, Omit, omit
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
from avior.core.messages import Message, TextPart
from avior.core.provider import ModelSettings

_DEFAULT_MAX_TOKENS = 4096
"""Fallback `max_tokens` when `ModelSettings.max_tokens` is `None`."""

type _AnthropicRole = Literal["user", "assistant"]
"""Anthropic's per-message role union.

Anthropic's Messages API places system instructions in a top-level `system`
parameter, so the per-message `role` is narrower than avior's canonical `Role`
(which includes `"system"`).
"""


class AnthropicProvider:
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
                over `api_key` if both are supplied.
            api_key: API key for a freshly constructed `AsyncAnthropic`.  If
                both `client` and `api_key` are `None`, `AsyncAnthropic` reads
                `ANTHROPIC_API_KEY` from the environment.
        """

        self._client = client if client is not None else AsyncAnthropic(api_key=api_key)

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
        return Message(role="assistant", parts=parts)

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
