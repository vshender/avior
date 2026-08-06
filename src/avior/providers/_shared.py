"""Helpers shared by the provider adapters.

Holds the SDK-agnostic pieces that every adapter needs verbatim.  Anything
that touches a provider SDK's own types stays in that provider's module -
the adapters remain the only place that knows SDK types.
"""

from avior.core.exceptions import EmptyInputError
from avior.core.messages import UserMessage
from avior.core.provider import ModelSettings
from avior.core.warnings import UnsupportedSettingRunWarning


def thinking_dropped(
    provider: str,
    settings: ModelSettings,
    reason: str,
) -> UnsupportedSettingRunWarning:
    """Build the warning for a `thinking` request that was dropped.

    Args:
        provider: The dropping provider's canonical name (`Provider.name`).
        settings: The settings whose `thinking` request was dropped.
        reason: A standalone explanation of why the request could not be
            honored.
    """

    return UnsupportedSettingRunWarning(
        setting_name="thinking",
        setting_value=settings.thinking,
        reason=reason,
        provider=provider,
        model=settings.model,
    )


def sampling_dropped(
    provider: str,
    settings: ModelSettings,
    reason: str,
) -> UnsupportedSettingRunWarning:
    """Build the warning for a `temperature` that was dropped.

    Args:
        provider: The dropping provider's canonical name (`Provider.name`).
        settings: The settings whose `temperature` was dropped.
        reason: A standalone explanation of why the value was dropped.
    """

    return UnsupportedSettingRunWarning(
        setting_name="temperature",
        setting_value=settings.temperature,
        reason=reason,
        provider=provider,
        model=settings.model,
    )


def reject_empty_user_turn(message: UserMessage) -> None:
    """Reject a user turn that carries no content.

    Args:
        message: The user turn about to be encoded into a request.

    Raises:
        EmptyInputError: The message is an empty user turn
            (`UserMessage.is_empty`).  There is no content to encode, so the
            message is rejected as a caller mistake before the request is
            sent.
    """

    if message.is_empty:
        raise EmptyInputError(
            "The transcript has a user message with no content to send."
        )
