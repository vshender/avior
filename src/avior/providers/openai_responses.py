"""Native OpenAI Responses provider adapter.

Wraps `openai.AsyncOpenAI` and implements `Provider` against OpenAI's Responses
API.

Stateless wire: `store=False` is always passed and `previous_response_id` is
not used.  avior treats the conversation transcript as user-owned; no
server-side state is created.

Install via the optional extra: `pip install avior[openai]`.
"""

import logging
from collections.abc import Sequence
from typing import Literal, assert_never

try:
    import openai
    from openai import AsyncOpenAI
    from openai._types import Omit, omit
    from openai.types.responses import (
        EasyInputMessageParam,
        Response,
        ResponseInputParam,
        ResponseOutputMessage,
        ResponseOutputRefusal,
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
from avior.core.messages import (
    AssistantMessage,
    Message,
    StopReason,
    SystemMessage,
    TextPart,
    UserMessage,
)
from avior.core.provider import ModelSettings, Provider

logger = logging.getLogger(__name__)

type _OpenAIRole = Literal["user", "assistant"]
"""OpenAI Responses' per-message role union.

OpenAI Responses' API places system instructions in a top-level `instructions`
parameter, so the per-message `role` is narrower than avior's canonical
`Message` (which also admits `SystemMessage`).
"""


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
    ) -> AssistantMessage:
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

        text_parts: list[TextPart] = []
        refusal_parts: list[TextPart] = []
        for item in response.output:
            if isinstance(item, ResponseOutputMessage):
                for content in item.content:
                    match content:
                        case ResponseOutputText():
                            text_parts.append(TextPart(text=content.text))
                        case ResponseOutputRefusal():
                            refusal_parts.append(TextPart(text=content.refusal))

        # When the response interleaves text and refusal (rare in non-
        # streaming), refusal wins: the refusal is the authoritative final
        # word and the partial text attempt is dropped.  A `RefusalPart`
        # type would let us keep both; deferred until parts gain richer
        # discriminator support.
        parts = refusal_parts or text_parts
        stop_reason = self._map_stop_reason(response, has_refusal=bool(refusal_parts))
        return AssistantMessage(parts=parts, stop_reason=stop_reason)

    @staticmethod
    def _map_stop_reason(response: Response, *, has_refusal: bool) -> StopReason:
        """Map OpenAI Responses signals to canonical `StopReason`.

        Two channels are checked, in order:

        1. `status == "incomplete"` with `incomplete_details.reason` -
           `"max_output_tokens"` -> `"max_tokens"`, `"content_filter"` ->
           `"content_filter"`.  Other reasons (or missing details) fall
           through to (2).
        2. A `ResponseOutputRefusal` content part on an otherwise completed
           response - the model itself declined to answer.

        Anything else maps to `"stop"`; the orchestrator treats the response as
        a normal completion and decides what to do from `parts` alone.
        """

        if response.status == "incomplete":
            details = response.incomplete_details
            if details is not None:
                match details.reason:
                    case "max_output_tokens":
                        return "max_tokens"
                    case "content_filter":
                        return "content_filter"
                    case _:
                        pass

        if has_refusal:
            return "refusal"

        return "stop"

    async def aclose(self) -> None:
        """Close the underlying SDK client when this provider owns it.

        No-op when the client was supplied by the caller via `client=` - its
        lifecycle belongs to whoever passed it in.  Safe to call more than once:
        `AsyncOpenAI.close` (and the httpx pool it delegates to)is itself
        idempotent.
        """

        if self._owns_client:
            await self._client.close()

    @staticmethod
    def _extract_instructions(
        messages: Sequence[Message],
    ) -> tuple[str | None, list[UserMessage | AssistantMessage]]:
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
        rest: list[UserMessage | AssistantMessage] = []
        for msg in messages:
            match msg:
                case SystemMessage():
                    if msg.text:
                        texts.append(msg.text)
                case UserMessage() | AssistantMessage():
                    rest.append(msg)
                case _:
                    assert_never(msg)

        return ("\n\n".join(texts) if texts else None), rest

    @staticmethod
    def _to_wire(message: UserMessage | AssistantMessage) -> EasyInputMessageParam:
        """Convert an avior non-system `Message` to an OpenAI Responses input
        item.

        `SystemMessage`s are not accepted - they are extracted into the
        top-level `instructions` parameter by `_extract_instructions`
        upstream.
        """

        match message:
            case UserMessage():
                role: _OpenAIRole = "user"
            case AssistantMessage():
                role = "assistant"
            case _:
                assert_never(message)

        return EasyInputMessageParam(
            role=role,
            type="message",
            content=message.text or "",
        )
