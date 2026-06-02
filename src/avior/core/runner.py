"""Agent runner."""

from collections.abc import Sequence

from avior.core.agent import Agent
from avior.core.exceptions import (
    ContentFilterError,
    MaxTokensExceededError,
    ModelRefusalError,
)
from avior.core.messages import Message, SystemMessage, UserMessage
from avior.core.result import RunResult


class Runner:
    """Static-method orchestrator for `Agent` execution."""

    @staticmethod
    async def run(agent: Agent, input: str | Sequence[Message]) -> RunResult:
        """Run `agent` on `input` and return the run result.

        Args:
            agent: The configured agent to drive.
            input: The conversation to send to the model:

                - a `str` is converted to one user message;
                - a sequence of messages continues an existing conversation, for
                  example a previous run's `RunResult.messages`.

                Before calling the provider, `Runner.run` prepends a
                `SystemMessage` from `agent.instructions` to the message list.
                Provider adapters then collect all `SystemMessage`s from that
                list.  For example, an adapter may move their text into a
                provider-level `instructions` or `system` field.

        Returns:
            A `RunResult` with the assistant's final text, the conversation
            transcript, and the run's token usage.

        Raises:
            ContentFilterError: An external content filter blocked the response.
            MaxTokensExceededError: Output was truncated by the token budget.
            ModelRefusalError: The model itself declined to answer.
        """

        input_messages: list[Message] = (
            [UserMessage.from_text(input)] if isinstance(input, str) else list(input)
        )
        messages: list[Message] = [
            SystemMessage.from_text(agent.instructions),
            *input_messages,
        ]
        response = await agent.provider.complete(messages, agent.model_settings)
        message = response.message

        match message.stop_reason:
            case "content_filter":
                raise ContentFilterError(
                    "Response was blocked by the provider's content filter."
                )
            case "max_tokens":
                configured = agent.model_settings.max_tokens
                detail = (
                    f"Model hit max_tokens budget ({configured}) before completing."
                    if configured is not None
                    else (
                        "Model hit the provider's default token cap before "
                        "completing.  Set `ModelSettings.max_tokens` explicitly "
                        "to raise the limit."
                    )
                )
                raise MaxTokensExceededError(detail)
            case "refusal":
                raise ModelRefusalError(message.text or "")
            case "stop":
                pass

        return RunResult(
            output=message.text or "",
            messages=[*input_messages, message],
            new_message_index=len(input_messages),
            usage=response.usage,
        )
