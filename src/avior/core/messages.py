"""Conversation transcript primitives.

The avior canonical message format uses a three-role shape (`system`, `user`,
`assistant`) modeled after Anthropic.  Each role is its own class so that fields
meaningful only on certain roles (such as `stop_reason` on assistant turns) live
exactly where they apply, making invalid states unrepresentable.  Provider
adapters are responsible for translating between this canonical form and the
wire shape of the underlying API.
"""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

type StopReason = Literal["stop", "max_tokens", "content_filter", "refusal"]
"""Canonical reason a model stopped producing output.

Normalized across providers so the orchestrator can apply a uniform policy
without branching on vendor specifics:

- `"stop"` - normal completion (end-of-turn, stop sequence, tool use).
- `"max_tokens"` - hit the configured token budget; output likely truncated.
- `"content_filter"` - the provider's server-side moderation filter blocked
  the response (a classifier layered between the model and the caller that
  screens generated output and zeroes it out on policy violation).
- `"refusal"` - the model itself declined to answer (the refusal text is present
  in `parts`).  Distinct from `"content_filter"`: the model produced output
  explaining why it refused, the response is "successful" at the transport level
  but not the requested answer.
"""


class TextPart(BaseModel):
    """A plain text content part of a message."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["text"] = "text"
    text: str


type Part = TextPart
"""A typed content part of a message."""


class SystemMessage(BaseModel):
    """A `system`-role turn carrying instructions to the model."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["system"] = "system"
    parts: list[TextPart]

    @classmethod
    def from_text(cls, text: str) -> Self:
        """Construct a system message with a single `TextPart`."""

        return cls(parts=[TextPart(text=text)])

    @property
    def text(self) -> str | None:
        """Concatenated text of all parts, or `None` if there are no parts."""

        return "".join(p.text for p in self.parts) if self.parts else None


class UserMessage(BaseModel):
    """A `user`-role turn carrying the caller's input to the model."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["user"] = "user"
    parts: list[TextPart]

    @classmethod
    def from_text(cls, text: str) -> Self:
        """Construct a user message with a single `TextPart`."""

        return cls(parts=[TextPart(text=text)])

    @property
    def text(self) -> str | None:
        """Concatenated text of all parts, or `None` if there are no parts."""

        return "".join(p.text for p in self.parts) if self.parts else None


class AssistantMessage(BaseModel):
    """An `assistant`-role turn produced by a model.

    Carries `stop_reason` describing why the model stopped (see `StopReason`).
    `Provider` adapters always set `stop_reason` when constructing an assistant
    message from a provider response.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["assistant"] = "assistant"
    parts: list[TextPart]
    stop_reason: StopReason

    @property
    def text(self) -> str | None:
        """Concatenated text of all parts, or `None` if there are no parts."""

        return "".join(p.text for p in self.parts) if self.parts else None


type Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage,
    Field(discriminator="kind"),
]
"""A single turn in the conversation transcript.  Discriminated on `kind`."""
