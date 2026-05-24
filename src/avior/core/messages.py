"""Conversation transcript primitives.

The avior canonical message format uses a three-role shape (`system`, `user`,
`assistant`) modeled after Anthropic. Provider adapters are responsible for
translating between this canonical form and the wire shape of the underlying
API.
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict

type Role = Literal["system", "user", "assistant"]
"""The role of a `Message` in a conversation."""

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

Set by `Provider` adapters on every assistant message they return.  User- or
system-constructed messages leave it as `None`.
"""


class TextPart(BaseModel):
    """A plain text content part of a `Message`."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["text"] = "text"
    text: str


type Part = TextPart
"""A typed content part of a `Message`."""


class Message(BaseModel):
    """A single turn in the conversation transcript.

    A `Message` has a role and a list of typed `Part`s.  Assistant-role messages
    produced by a provider also carry a `stop_reason` describing why the model
    stopped (see `StopReason`).

    Convenience constructors `Message.system`, `Message.user`, and
    `Message.assistant` cover the common single-`TextPart` case used by simple
    agents.
    """

    model_config = ConfigDict(frozen=True)

    role: Role
    parts: list[Part]
    stop_reason: StopReason | None = None

    @classmethod
    def system(cls, text: str) -> Self:
        """Construct a `system`-role message with a single `TextPart`."""

        return cls(role="system", parts=[TextPart(text=text)])

    @classmethod
    def user(cls, text: str) -> Self:
        """Construct a `user`-role message with a single `TextPart`."""

        return cls(role="user", parts=[TextPart(text=text)])

    @classmethod
    def assistant(cls, text: str) -> Self:
        """Construct an `assistant`-role message with a single `TextPart`."""

        return cls(role="assistant", parts=[TextPart(text=text)])

    @property
    def text(self) -> str | None:
        """Concatenated text of all parts, or `None` if there are no parts."""

        if not self.parts:
            return None

        return "".join(p.text for p in self.parts)
