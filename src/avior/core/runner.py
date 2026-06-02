"""Agent runner."""

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
    async def run(agent: Agent, input: str) -> RunResult:
        """Run `agent` on `input` and return the run result.

        Args:
            agent: The configured agent to drive.
            input: The user prompt sent to the model.

        Returns:
            A `RunResult` carrying the assistant's final text (`output`, the
            empty string if the response has no text parts) and the run's token
            usage.

        Raises:
            ContentFilterError: An external content filter blocked the response.
            MaxTokensExceededError: Output was truncated by the token budget.
            ModelRefusalError: The model itself declined to answer.
        """

        messages: list[Message] = [
            SystemMessage.from_text(agent.instructions),
            UserMessage.from_text(input),
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

        return RunResult(output=message.text or "", usage=response.usage)
