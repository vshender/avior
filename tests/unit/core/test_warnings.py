"""Tests for `avior.core.warnings`."""

from avior.core.warnings import UnsupportedSettingRunWarning


def test_unsupported_setting_warning_message_truncates_a_long_value() -> None:
    """`message` truncates a long setting value to keep the log bounded."""

    # GIVEN a warning whose setting value is long
    long_value = "x" * 200
    warning = UnsupportedSettingRunWarning(
        setting_name="thinking",
        setting_value=long_value,
        provider="anthropic",
        model="claude-x",
    )

    # WHEN the message is read
    message = warning.message

    # THEN the long value is shortened with an ellipsis marker
    assert "..." in message
    assert long_value not in message


def test_unsupported_setting_warning_message_includes_the_reason() -> None:
    """`message` includes `reason` when one is given."""

    # GIVEN a warning carrying a reason
    warning = UnsupportedSettingRunWarning(
        setting_name="thinking",
        setting_value="high",
        provider="anthropic",
        model="claude-x",
        reason="the thinking budget does not fit max_tokens",
    )

    # WHEN the message is read
    message = warning.message

    # THEN the reason appears in it
    assert "the thinking budget does not fit max_tokens" in message
