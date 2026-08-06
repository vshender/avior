"""Result of an agent run.

`RunResult` is returned by `Runner.run`.  It is the public record of what the
run produced and the information needed to inspect or continue it.
"""

from pydantic import BaseModel, ConfigDict, Field

from avior.core.messages import Message
from avior.core.provider import ProviderResponse
from avior.core.usage import Usage
from avior.core.warnings import RunWarning


class RunResult(BaseModel):
    """The outcome of a single `Runner.run`."""

    model_config = ConfigDict(frozen=True)

    output: str
    """The text of the assistant's final response."""

    messages: list[Message]
    """Conversation transcript for this run.

    This list contains the input messages followed by the messages produced by
    this run.  It carries no system prompt: `agent.instructions` is run
    configuration passed to the provider separately, re-applied on each call.
    """

    new_message_index: int = Field(ge=0)
    """Index where this run's new messages start.

    `messages[:new_message_index]` is the input passed to the run.
    `messages[new_message_index:]` is what the run produced.  The index is
    stored so this split survives serialization.
    """

    usage: Usage | None = None
    """Normalized token usage for the whole run, or `None` if the provider
    reported none.
    """

    warnings: list[RunWarning] = Field(default_factory=list)
    """Non-fatal problems found during the run, in the order they occurred."""

    provider_responses: list[ProviderResponse] = Field(default_factory=list)
    """The `ProviderResponse` of every model call the run made, in call order.

    The per-call records behind the aggregated fields: `usage` sums the usage
    the entries report (`None` when no entry reports usage), and `warnings`
    concatenates their warnings in the same order.  Kept for tooling that
    needs per-call evidence rather than run totals.

    Each entry wraps the same assistant message that appears in `messages`,
    so a serialized result carries every assistant message the run produced
    twice - the cost of keeping the per-call records complete.
    """

    @property
    def new_messages(self) -> list[Message]:
        """Messages produced by this run, excluding the input messages."""

        return self.messages[self.new_message_index :]
