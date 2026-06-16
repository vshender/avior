"""LLM provider abstraction."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from types import TracebackType
from typing import Any, Self

from pydantic import BaseModel, ConfigDict

from avior.core.messages import AssistantMessage, Message
from avior.core.tools import Tool
from avior.core.usage import Usage


class ModelSettings(BaseModel):
    """Per-call model invocation settings."""

    model_config = ConfigDict(frozen=True)

    model: str
    """The model to use (a provider-specific identifier)."""

    temperature: float | None = None
    """Sampling temperature; `None` uses the provider's default."""

    max_tokens: int | None = None
    """Maximum output tokens; `None` uses the provider's default."""


class ProviderResponse(BaseModel):
    """Result of a single `Provider.complete` call.

    Wraps the assistant message together with the call metadata.  This
    metadata lives *beside* the message rather than inside it so that
    `AssistantMessage` stays a clean transcript primitive: replay and test
    transcripts carry no call-specific bookkeeping.

    All metadata fields are optional; a provider populates what it can.
    """

    model_config = ConfigDict(frozen=True)

    message: AssistantMessage
    """The assistant message produced by the call."""

    usage: Usage | None = None
    """Normalized token usage for the call, or `None` if the provider reported
    none.
    """

    raw_usage: dict[str, Any] | None = None
    """The provider's own usage payload as JSON-like data.  Per-call provenance
    kept beside the normalized `usage`, for debugging / audit and for cost
    tooling that prefers provider-native numbers.
    """

    response_id: str | None = None
    """The model provider's id for this response.  For correlating with
    provider-side logs and traces.
    """

    model: str | None = None
    """The model the provider reports having served, which may differ from the
    requested `ModelSettings.model` (alias resolution, fallback).
    """

    provider_name: str | None = None
    """Name of the provider that produced this response."""


class Provider(ABC):
    """Adapter to an LLM service.

    Stateless wrapper around an SDK client.  Subclasses convert the canonical
    `Message` / `Part` shape into the provider's wire format and back.
    Implementations should be safe to share across concurrent runs.

    Lifecycle: release resources held by the provider with either `await
    provider.aclose()` or `async with provider: ...`.  Nested `async with` on
    the same provider is supported; resources are released only on the
    outermost exit.  After `aclose` has completed the provider is no longer
    usable; construct a fresh one instead.

    Subclasses implement `complete` and `aclose`; the async context manager
    dunder methods are provided here and call `aclose` on the outermost exit.
    """

    def __init__(self) -> None:
        """Initialize the lifecycle bookkeeping.

        Subclasses must call `super().__init__()` so that nested `async with`
        works correctly.
        """

        self._entered_count = 0

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[Message],
        settings: ModelSettings,
        *,
        tools: Sequence[Tool[Any, Any, Any]] = (),
        system_prompt: str | None = None,
    ) -> ProviderResponse:
        """Send the conversation to the model and return its response.

        Args:
            messages: Conversation transcript to send to the model.  Carries
                only `user`, `assistant`, and `tool` turns - never a system
                turn.
            settings: Per-call invocation settings.
            tools: Tools to offer the model.  The adapter sends each tool's
                name, description, and arguments JSON schema, and parses any
                tool calls in the response into `ToolCallPart`s on the assistant
                message.
            system_prompt: The system prompt for this call, or `None` for no
                system prompt.  Pass `None` to omit it - a blank or
                whitespace-only string is not a valid stand-in for `None`, and
                how a provider treats one is provider-defined.  The adapter
                sends it the way its API expects - for example a top-level
                system field, or a system-role message.  It is separate from
                `messages` because the system prompt is run configuration, not a
                conversational turn.

        Returns:
            A `ProviderResponse` carrying the assistant message and the
            call metadata.
        """

    @abstractmethod
    async def aclose(self) -> None:
        """Release any resources held by the provider.

        Subclasses must implement this to be idempotent: calling `aclose`
        more than once must be safe.
        """

    async def __aenter__(self) -> Self:
        """Enter the provider as an async context manager.

        Returns the provider itself.
        """

        self._entered_count += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the async context manager block.

        `aclose` runs only on the outermost exit.
        """

        self._entered_count -= 1
        if self._entered_count == 0:
            await self.aclose()
