"""Tests for `avior.core.messages`."""

import pytest
from pydantic import ValidationError

from avior.core.messages import (
    AssistantMessage,
    AssistantPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    UserMessage,
)

# `UserMessage` tests
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


def test_user_message_rejects_part_with_provider_details() -> None:
    """Constructing a user turn fails when a part carries `provider_details`:
    a user turn is not produced by a provider, so the data has no owner.
    """

    # GIVEN a text part carrying `provider_details`
    part = TextPart(text="hi", provider_details={"thought_signature": "c2ln"})

    # WHEN a user message is constructed from it
    # THEN construction is rejected
    with pytest.raises(ValidationError, match="provider_details"):
        UserMessage(parts=[part])


# `AssistantMessage` tests
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


def test_assistant_message_text_ignores_thinking_parts() -> None:
    """`AssistantMessage.text` skips `ThinkingPart`s, returning only text."""

    # GIVEN an assistant message with a thinking part and a text part
    message = AssistantMessage(
        parts=[ThinkingPart(content="reasoning"), TextPart(text="answer")],
        stop_reason="stop",
    )

    # WHEN the `text` property is read
    result = message.text

    # THEN only the text part contributes
    assert result == "answer"


@pytest.mark.parametrize(
    "part",
    [
        TextPart(text="answer", provider_details={"thought_signature": "c2ln"}),
        ToolCallPart(
            call_id="call_1",
            tool_name="get_weather",
            args={},
            provider_details={"thought_signature": "c2ln"},
        ),
        ThinkingPart(content="hmm", provider_details={"signature": "sig"}),
    ],
    ids=["text", "tool_call", "thinking"],
)
def test_assistant_message_rejects_provider_details_without_provider_name(
    part: AssistantPart,
) -> None:
    """Constructing an assistant turn whose part carries `provider_details`
    fails when the turn does not name the provider that owns the data.
    """

    # GIVEN a part carrying `provider_details`
    # (parametrized as `part`)

    # WHEN an assistant message with no `provider_name` is constructed from it
    # THEN construction is rejected, pointing at the missing owner
    with pytest.raises(ValidationError, match="provider_name"):
        AssistantMessage(parts=[part], stop_reason="stop")


def test_assistant_message_accepts_provider_details_with_provider_name() -> None:
    """An assistant turn whose part carries `provider_details` is accepted when
    the turn names the provider that owns the data.
    """

    # GIVEN a part carrying `provider_details`
    part = ThinkingPart(content="hmm", provider_details={"signature": "sig"})

    # WHEN an assistant message naming its provider is constructed from it
    message = AssistantMessage(
        parts=[part],
        stop_reason="stop",
        provider_name="anthropic",
    )

    # THEN construction succeeds and the part is kept
    assert message.parts == [part]
