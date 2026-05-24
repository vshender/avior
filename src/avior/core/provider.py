"""LLM provider abstraction."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from types import TracebackType
from typing import Self

from pydantic import BaseModel, ConfigDict

from avior.core.messages import AssistantMessage, Message


class ModelSettings(BaseModel):
    """Per-call model invocation settings.

    `model` is the only required field.  `temperature` and `max_tokens` are
    optional with `None` meaning "use the provider's default for this model";
    each adapter applies its own fallback if it requires a value.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    temperature: float | None = None
    max_tokens: int | None = None


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
    ) -> AssistantMessage:
        """Send `messages` to the model and return its response message.

        Args:
            messages: Conversation transcript to send to the model.
            settings: Per-call invocation settings.

        Returns:
            The model's response as an `AssistantMessage`.
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
