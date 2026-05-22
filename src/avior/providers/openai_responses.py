"""Native OpenAI Responses provider adapter.

Wraps `openai.AsyncOpenAI` and implements the `Provider` protocol against
OpenAI's Responses API.

Stateless wire: `store=False` is always passed and `previous_response_id` is
not used.  avior treats the conversation transcript as user-owned; no
server-side state is created.

Install via the optional extra: `pip install avior[openai]`.
"""

import logging
from typing import Literal, assert_never

try:
    import openai
    from openai import AsyncOpenAI
    from openai._types import Omit, omit
    from openai.types.responses import (
        EasyInputMessageParam,
        ResponseInputParam,
        ResponseOutputMessage,
        ResponseOutputText,
    )
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
from avior.core.messages import Message, TextPart
from avior.core.provider import ModelSettings

logger = logging.getLogger(__name__)

type _OpenAIRole = Literal["user", "assistant"]
"""OpenAI Responses' per-message role union.

OpenAI Responses' API places system instructions in a top-level `instructions`
parameter, so the per-message `role` is narrower than avior's canonical `Role`
(which includes `"system"`).
"""


class OpenAIResponsesProvider:
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
                `api_key` if both are supplied.
            api_key: API key for a freshly constructed `AsyncOpenAI`.  If both
                `client` and `api_key` are `None`, `AsyncOpenAI` reads
                `OPENAI_API_KEY` from the environment.
        """

        self._client = client if client is not None else AsyncOpenAI(api_key=api_key)

    async def complete(
        self,
        messages: list[Message],
        settings: ModelSettings,
    ) -> Message:
        """Send `messages` to OpenAI Responses API and return the assistant's
        response.

        `store=False` is always passed (stateless wire; no server-side
        history).  `temperature` and `max_output_tokens` are forwarded only
        when explicitly set on `settings`.

        Args:
            messages: Conversation transcript.  `system`-role messages are
                lifted out of the transcript and joined (newline-separated)
                into OpenAI Responses' top-level `instructions` parameter
                (see `_extract_instructions` for the rationale of this choice
                over inline `role='system'` items).  Relative order among
                non-`system` messages is preserved.
            settings: Per-call invocation settings.

        Returns:
            The assistant response message.

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
        wire_input: ResponseInputParam = [self._to_wire(m) for m in conversation]

        instructions_param: str | Omit = (
            instructions if instructions is not None else omit
        )
        temperature_param: float | Omit = (
            settings.temperature if settings.temperature is not None else omit
        )
        max_output_tokens_param: int | Omit = (
            settings.max_tokens if settings.max_tokens is not None else omit
        )

        try:
            response = await self._client.responses.create(
                input=wire_input,
                instructions=instructions_param,
                model=settings.model,
                max_output_tokens=max_output_tokens_param,
                temperature=temperature_param,
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

        parts: list[TextPart] = []
        for item in response.output:
            if isinstance(item, ResponseOutputMessage):
                for content in item.content:
                    if isinstance(content, ResponseOutputText):
                        parts.append(TextPart(text=content.text))
        return Message(role="assistant", parts=parts)

    @staticmethod
    def _extract_instructions(
        messages: list[Message],
    ) -> tuple[str | None, list[Message]]:
        """Pull all `system` messages out of the conversation.

        System content is lifted to the top-level `instructions` parameter
        rather than passed inline as `role='system'` items in `input` (the API
        supports both shapes).  The top-level choice
        - keeps prompt-cache prefixes stable
        - lets OpenAI fold `instructions` to the `developer` role automatically
          for reasoning models (no per-model branching in the adapter)

        `instructions` accepts a single string only, so multiple `system`
        messages are collected (newline-separated) into one string.  Empty
        `system` messages are skipped.

        Returns `(instructions, rest)`; `instructions` is `None` when no
        non-empty `system` message is present.
        """

        texts: list[str] = []
        rest: list[Message] = []
        for msg in messages:
            if msg.role == "system":
                if msg.text:
                    texts.append(msg.text)
            else:
                rest.append(msg)

        return ("\n\n".join(texts) if texts else None), rest

    @staticmethod
    def _to_wire(message: Message) -> EasyInputMessageParam:
        """Convert an avior `Message` to an OpenAI Responses input item.

        System messages are not accepted - they are extracted into the
        top-level `instructions` parameter by `_extract_instructions`
        upstream.
        """

        assert message.role != "system", (
            "System messages must be filtered out by `_extract_instructions` upstream."
        )

        match message.role:
            case "user" | "assistant":
                role: _OpenAIRole = message.role
            case _:
                assert_never(message.role)

        return EasyInputMessageParam(
            role=role,
            type="message",
            content=message.text or "",
        )
