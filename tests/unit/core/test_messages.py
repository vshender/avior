"""Tests for `avior.core.messages`."""

from avior.core.messages import Message, TextPart


def test_message_system_constructs_system_role_with_text_part() -> None:
    """`Message.system` builds a system-role message with `TextPart`."""

    # GIVEN a system instruction text
    instruction = "you are a helpful assistant"

    # WHEN `Message.system` is called with that text
    message = Message.system(instruction)

    # THEN the message has role `system` and one `TextPart` with the text
    assert message.role == "system"
    assert message.parts == [TextPart(text=instruction)]


def test_message_user_constructs_user_role_with_text_part() -> None:
    """`Message.user` builds a user-role message with `TextPart`."""

    # GIVEN a user prompt text
    prompt = "hello"

    # WHEN `Message.user` is called with that text
    message = Message.user(prompt)

    # THEN the message has role `user` and one `TextPart` with the text
    assert message.role == "user"
    assert message.parts == [TextPart(text=prompt)]


def test_message_assistant_constructs_assistant_role_with_text_part() -> None:
    """`Message.assistant` builds an assistant-role message with `TextPart`."""

    # GIVEN an assistant response text
    response = "hi there"

    # WHEN `Message.assistant` is called with that text
    message = Message.assistant(response)

    # THEN the message has role `assistant` and one `TextPart` with the text
    assert message.role == "assistant"
    assert message.parts == [TextPart(text=response)]


def test_message_text_returns_text_for_single_part() -> None:
    """`Message.text` returns the text of a single `TextPart`."""

    # GIVEN a message with a single `TextPart`
    message = Message.user("hello")

    # WHEN the `text` property is read
    result = message.text

    # THEN it equals the original part's text
    assert result == "hello"


def test_message_text_concatenates_multiple_parts_in_order() -> None:
    """`Message.text` concatenates the text of multiple `TextPart`s in order."""

    # GIVEN a message whose parts are two `TextPart`s
    message = Message(
        role="user",
        parts=[TextPart(text="hello "), TextPart(text="world")],
    )

    # WHEN the `text` property is read
    result = message.text

    # THEN the result is the parts' texts concatenated in order
    assert result == "hello world"


def test_message_text_returns_none_for_empty_parts() -> None:
    """`Message.text` returns `None` when the message has no parts."""

    # GIVEN a message with an empty parts list
    message = Message(role="user", parts=[])

    # WHEN the `text` property is read
    result = message.text

    # THEN the result is `None`
    assert result is None
