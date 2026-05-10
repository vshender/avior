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


class TextPart(BaseModel):
    """A plain text content part of a `Message`."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["text"] = "text"
    text: str


type Part = TextPart
"""A typed content part of a `Message`."""


class Message(BaseModel):
    """A single turn in the conversation transcript.

    A `Message` has a role and a list of typed `Part`s. Convenience constructors
    `Message.system`, `Message.user`, and `Message.assistant` cover the common
    single-`TextPart` case used by simple agents.
    """

    model_config = ConfigDict(frozen=True)

    role: Role
    parts: list[Part]

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
