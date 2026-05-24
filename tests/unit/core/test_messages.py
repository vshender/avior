"""Tests for `avior.core.messages`."""

from avior.core.messages import (
    AssistantMessage,
    SystemMessage,
    TextPart,
    UserMessage,
)

# SystemMessage tests
# -----------------------------------------------------------------------------


def test_system_message_from_text_constructs_single_text_part_message() -> None:
    """`SystemMessage.from_text` builds a message with a single `TextPart`."""

    # GIVEN a text string
    text = "you are helpful"

    # WHEN `SystemMessage.from_text` is called with that text
    message = SystemMessage.from_text(text)

    # THEN the message has a single `TextPart` carrying the text
    assert message.parts == [TextPart(text=text)]


def test_system_message_text_returns_text_for_single_part() -> None:
    """`SystemMessage.text` returns the text of a single `TextPart`."""

    # GIVEN a system message with a single `TextPart`
    message = SystemMessage(parts=[TextPart(text="hello")])

    # WHEN the `text` property is read
    result = message.text

    # THEN it equals the original part's text
    assert result == "hello"


def test_system_message_text_concatenates_multiple_parts_in_order() -> None:
    """`SystemMessage.text` concatenates multiple `TextPart`s in order."""

    # GIVEN a system message whose parts are two `TextPart`s
    message = SystemMessage(parts=[TextPart(text="hello "), TextPart(text="world")])

    # WHEN the `text` property is read
    result = message.text

    # THEN the result is the parts' texts concatenated in order
    assert result == "hello world"


def test_system_message_text_returns_none_for_empty_parts() -> None:
    """`SystemMessage.text` returns `None` when the message has no parts."""

    # GIVEN a system message with an empty parts list
    message = SystemMessage(parts=[])

    # WHEN the `text` property is read
    result = message.text

    # THEN the result is `None`
    assert result is None


# UserMessage tests
# -----------------------------------------------------------------------------


def test_user_message_from_text_constructs_single_text_part_message() -> None:
    """`UserMessage.from_text` builds a message with a single `TextPart`."""

    # GIVEN a text string
    text = "hello"

    # WHEN `UserMessage.from_text` is called with that text
    message = UserMessage.from_text(text)

    # THEN the message has a single `TextPart` carrying the text
    assert message.parts == [TextPart(text=text)]


def test_user_message_text_returns_text_for_single_part() -> None:
    """`UserMessage.text` returns the text of a single `TextPart`."""

    # GIVEN a user message with a single `TextPart`
    message = UserMessage(parts=[TextPart(text="hello")])

    # WHEN the `text` property is read
    result = message.text

    # THEN it equals the original part's text
    assert result == "hello"


def test_user_message_text_concatenates_multiple_parts_in_order() -> None:
    """`UserMessage.text` concatenates multiple `TextPart`s in order."""

    # GIVEN a user message whose parts are two `TextPart`s
    message = UserMessage(parts=[TextPart(text="hello "), TextPart(text="world")])

    # WHEN the `text` property is read
    result = message.text

    # THEN the result is the parts' texts concatenated in order
    assert result == "hello world"


def test_user_message_text_returns_none_for_empty_parts() -> None:
    """`UserMessage.text` returns `None` when the message has no parts."""

    # GIVEN a user message with an empty parts list
    message = UserMessage(parts=[])

    # WHEN the `text` property is read
    result = message.text

    # THEN the result is `None`
    assert result is None


# AssistantMessage tests
# -----------------------------------------------------------------------------


def test_assistant_message_text_returns_text_for_single_part() -> None:
    """`AssistantMessage.text` returns the text of a single `TextPart`."""

    # GIVEN an assistant message with a single `TextPart`
    message = AssistantMessage(parts=[TextPart(text="hello")], stop_reason="stop")

    # WHEN the `text` property is read
    result = message.text

    # THEN it equals the original part's text
    assert result == "hello"


def test_assistant_message_text_concatenates_multiple_parts_in_order() -> None:
    """`AssistantMessage.text` concatenates multiple `TextPart`s in order."""

    # GIVEN an assistant message whose parts are two `TextPart`s
    message = AssistantMessage(
        parts=[TextPart(text="hello "), TextPart(text="world")],
        stop_reason="stop",
    )

    # WHEN the `text` property is read
    result = message.text

    # THEN the result is the parts' texts concatenated in order
    assert result == "hello world"


def test_assistant_message_text_returns_none_for_empty_parts() -> None:
    """`AssistantMessage.text` returns `None` when the message has no parts."""

    # GIVEN an assistant message with an empty parts list
    message = AssistantMessage(parts=[], stop_reason="stop")

    # WHEN the `text` property is read
    result = message.text

    # THEN the result is `None`
    assert result is None
