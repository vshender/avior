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

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue

type StopReason = Literal[
    "stop",
    "tool_use",
    "max_tokens",
    "content_filter",
    "refusal",
    "error",
]
"""Canonical reason a model stopped producing output.

Normalized across providers so the orchestrator can apply a uniform policy
without branching on vendor specifics:

- `"stop"` - normal completion (end-of-turn, stop sequence).
- `"tool_use"` - the model wants to call one or more tools; the requested calls
  are present in `parts` as `ToolCallPart`s, which the orchestrator dispatches
  before continuing the run.
- `"max_tokens"` - hit the configured token budget; output likely truncated.
- `"content_filter"` - the provider's server-side moderation filter blocked
  the exchange (a classifier layered around the model that screens content and
  zeroes it out on policy violation).  This can block either the prompt before
  generation or the generated response.
- `"refusal"` - the model itself declined to answer (the refusal text is present
  in `parts`).  Distinct from `"content_filter"`: the model produced output
  explaining why it refused, the response is "successful" at the transport level
  but not the requested answer.
- `"error"` - the model terminated abnormally without a usable response (e.g. it
  tried to call a tool but produced a malformed or unexpected call).  The
  transport call succeeded, so this is a model/run failure, not a transport one;
  the orchestrator aborts the run.
"""


class TextPart(BaseModel):
    """A plain text content part of a message."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["text"] = "text"

    text: str
    """The text content."""


class ToolCallPart(BaseModel):
    """A request from the LLM to call a tool, part of an assistant turn."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    kind: Literal["tool_call"] = "tool_call"

    call_id: str
    """The ID that correlates this call with its eventual `ToolResultPart`."""

    tool_name: str
    """The name of the tool the LLM wants to call."""

    args: dict[str, JsonValue]
    """The raw arguments object the LLM produced."""

    provider_details: dict[str, JsonValue] | None = None
    """Opaque provider data the model expects to receive back unchanged on a
    later turn.  Carries the provider's round-trip token for the call - for
    example a Gemini `thought_signature`, which the Gemini API checks on replay
    to keep a multi-step tool chain valid.

    A provider sets it on the calls it produces; the turn's `provider_name`
    records the owner, so the data is sent back only to that same provider.  A
    different provider drops it, since the token is provider-specific and
    non-portable.

    Despite the message being frozen, the dict's contents are not - do not
    mutate them in place.
    """


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

    @classmethod
    def ok(cls, call_id: str, content: str) -> Self:
        """Build a successful result for the call `call_id`."""

        return cls(call_id=call_id, result=ToolResultOk(content=content))

    @classmethod
    def error(cls, call_id: str, message: str) -> Self:
        """Build an error result for the call `call_id`."""

        return cls(call_id=call_id, result=ToolResultError(content=message))


class ThinkingPart(BaseModel):
    """A reasoning step a model emitted, part of an assistant turn.

    Models that reason expose it as its own content block (Anthropic thinking,
    OpenAI reasoning items, Gemini thought parts).  A block can be text-less: a
    provider may return only the opaque round-trip token, with no summary.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    kind: Literal["thinking"] = "thinking"

    content: str
    """The reasoning summary text; may be empty."""

    provider_details: dict[str, JsonValue] | None = None
    """Opaque provider data the model expects to receive back unchanged on a
    later turn.  Carries the provider's round-trip token for the reasoning
    step - for example an Anthropic thinking block's `signature` or a redacted
    block's `data`, an OpenAI reasoning item's `encrypted_content`, or a Gemini
    `thought_signature`.

    A provider sets it on the blocks it produces; the turn's `provider_name`
    records the owner, so the data is sent back only to that same provider.  A
    different provider drops it, since the token is provider-specific and
    non-portable.

    Despite the message being frozen, the dict's contents are not - do not
    mutate them in place.
    """


type AssistantPart = Annotated[
    TextPart | ToolCallPart | ThinkingPart,
    Field(discriminator="kind"),
]
"""A content part of an assistant turn: text, a request to call a tool, or a
reasoning step.
"""

type Part = Annotated[
    TextPart | ToolCallPart | ThinkingPart | ToolResultPart,
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
    """The assistant's content parts."""

    stop_reason: StopReason
    """Why the model stopped (see `StopReason`); always set by `Provider`
    adapters when building the message from a provider response.
    """

    provider_name: str | None = None
    """Name of the provider that produced this turn, or `None` for a turn built
    by hand rather than by a provider.  A turn replayed from a stored transcript
    keeps the provider name it was produced with.

    It records which provider's opaque part data the turn carries, so that data
    is sent back only to that same provider and never replayed to a different
    one.
    """

    @property
    def text(self) -> str | None:
        """Concatenated text of the message's `TextPart`s, or `None` if none.

        Non-text parts are ignored, so this is the assistant's natural-language
        text alongside any tool requests or reasoning.
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
