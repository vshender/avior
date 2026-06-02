"""Result of an agent run.

`RunResult` is what `Runner.run` returns: the run's final output together with
the run-level metadata accumulated while producing it.
"""

from pydantic import BaseModel, ConfigDict

from avior.core.usage import Usage


class RunResult(BaseModel):
    """The outcome of a single `Runner.run`."""

    model_config = ConfigDict(frozen=True)

    output: str
    """The assistant's final text response, or `""` if it produced no text."""

    usage: Usage | None = None
    """Normalized token usage for the whole run, or `None` if the provider
    reported none.
    """
