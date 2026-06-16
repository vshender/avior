"""Conversation transcript primitives.

A message is one turn in the conversation, tagged by its `kind` - the
conversation role.  `user` and `assistant` are the standard chat roles; `tool`
carries tool-call results.  There is no system role: the system prompt is run
configuration passed to the provider separately, not a turn in the transcript.
Each kind is its own class so that fields meaningful only on certain kinds (such
as `stop_reason` on assistant turns) live exactly where they apply, making
invalid states unrepresentable.

Provider adapters translate between this canonical form and the wire shape of
the underlying API.
"""

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

type StopReason = Literal["stop", "tool_use", "max_tokens", "content_filter", "refusal"]
"""Canonical reason a model stopped producing output.

Normalized across providers so the orchestrator can apply a uniform policy
without branching on vendor specifics:

- `"stop"` - normal completion (end-of-turn, stop sequence).
- `"tool_use"` - the model wants to call one or more tools; the requested calls
  are present in `parts` as `ToolCallPart`s, which the orchestrator dispatches
  before continuing the run.
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
    """The text content."""


class ToolCallPart(BaseModel):
    """A request from the LLM to call a tool, part of an assistant turn."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["tool_call"] = "tool_call"

    call_id: str
    """The ID that correlates this call with its eventual `ToolResultPart`."""

    tool_name: str
    """The name of the tool the LLM wants to call."""

    args: dict[str, Any]
    """The raw arguments object the LLM produced."""


class ToolResultOk(BaseModel):
    """A successful tool call's result."""

    model_config = ConfigDict(frozen=True)

    status: Literal["ok"] = "ok"

    content: str
    """The tool call's result, rendered as text for the model."""


class ToolResultError(BaseModel):
    """A failed tool call - the tool was missing, its arguments were invalid, or
    `execute` raised.
    """

    model_config = ConfigDict(frozen=True)

    status: Literal["error"] = "error"

    content: str
    """The error message, rendered as text for the model."""


type ToolResult = Annotated[
    ToolResultOk | ToolResultError,
    Field(discriminator="status"),
]
"""The outcome of a single tool call: `ToolResultOk` or `ToolResultError`."""


class ToolResultPart(BaseModel):
    """A tool's result, returned to the LLM in a `ToolMessage`."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["tool_result"] = "tool_result"

    call_id: str
    """The `ToolCallPart.call_id` this result answers."""

    result: ToolResult
    """The call's outcome."""


type AssistantPart = Annotated[
    TextPart | ToolCallPart,
    Field(discriminator="kind"),
]
"""A content part of an assistant turn: text, or a request to call a tool."""

type Part = Annotated[
    TextPart | ToolCallPart | ToolResultPart,
    Field(discriminator="kind"),
]
"""Any typed content part of a message.  Discriminated on `kind`."""


class UserMessage(BaseModel):
    """A `user`-role turn carrying the caller's input to the model."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["user"] = "user"

    parts: list[TextPart]
    """The caller's input, as text parts."""

    @classmethod
    def from_text(cls, text: str) -> Self:
        """Construct a user message with a single `TextPart`."""

        return cls(parts=[TextPart(text=text)])

    @property
    def text(self) -> str | None:
        """Concatenated text of all parts, or `None` if there are no parts."""

        return "".join(p.text for p in self.parts) if self.parts else None


class AssistantMessage(BaseModel):
    """An `assistant`-role turn produced by a model."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["assistant"] = "assistant"

    parts: list[AssistantPart]
    """The assistant's content: text and tool-call parts."""

    stop_reason: StopReason
    """Why the model stopped (see `StopReason`); always set by `Provider`
    adapters when building the message from a provider response.
    """

    @property
    def text(self) -> str | None:
        """Concatenated text of the message's `TextPart`s, or `None` if none.

        Tool-call parts are ignored, so this is the assistant's natural-language
        text alongside any tool requests.
        """

        texts = [p.text for p in self.parts if isinstance(p, TextPart)]
        return "".join(texts) if texts else None


class ToolMessage(BaseModel):
    """A turn carrying tool-call results back to the model."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["tool"] = "tool"

    parts: list[ToolResultPart]
    """The tool-call results carried by this turn."""

    @property
    def text(self) -> str | None:
        """Always `None`: a tool turn carries structured results, not text.

        Present so every `Message` exposes `text` uniformly; read `parts` for
        the results.
        """

        return None


type Message = Annotated[
    UserMessage | AssistantMessage | ToolMessage,
    Field(discriminator="kind"),
]
"""A single turn in the conversation transcript.  Discriminated on `kind`.

The transcript carries no system role: the system prompt is run configuration
passed to the provider separately, not a conversational turn.
"""
