"""LLM provider protocol."""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from avior.core.messages import Message


class ModelSettings(BaseModel):
    """Per-call model invocation settings.

    `model` is the only required field. `temperature` and `max_tokens` are
    optional with `None` meaning "use the provider's default for this model";
    each adapter applies its own fallback if it requires a value.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    temperature: float | None = None
    max_tokens: int | None = None


@runtime_checkable
class Provider(Protocol):
    """Adapter to an LLM service.

    Stateless wrapper around an SDK client. Implementations convert the
    canonical `Message` / `Part` shape into the provider's wire format and back.
    Implementations should be safe to share across concurrent runs (the
    underlying SDK client is reused).
    """

    async def complete(
        self,
        messages: list[Message],
        settings: ModelSettings,
    ) -> Message:
        """Send `messages` to the model and return its response message.

        Args:
            messages: Conversation transcript to send to the model.
            settings: Per-call invocation settings.

        Returns:
            The model's response as a `Message` with `role="assistant"`.
        """

        ...
