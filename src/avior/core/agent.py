"""Agent definition."""

from dataclasses import dataclass

from avior.core.provider import ModelSettings, Provider


@dataclass(frozen=True, kw_only=True)
class Agent:
    """Agent definition.

    Holds the static configuration that `Runner` uses to drive a conversation.
    """

    provider: Provider
    instructions: str
    model_settings: ModelSettings
