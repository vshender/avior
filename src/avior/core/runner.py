"""Agent runner."""

from avior.core.agent import Agent
from avior.core.exceptions import (
    ContentFilterError,
    MaxTokensExceededError,
    ModelRefusalError,
)
from avior.core.messages import Message


class Runner:
    """Static-method orchestrator for `Agent` execution."""

    @staticmethod
    async def run(agent: Agent, input: str) -> str:
        """Run `agent` on `input` and return the assistant's text response.

        Args:
            agent: The configured agent to drive.
            input: The user prompt sent to the model.

        Returns:
            The concatenated text of the assistant's response, or an empty
            string if the response has no text parts.

        Raises:
            ContentFilterError: An external content filter blocked the response.
            MaxTokensExceededError: Output was truncated by the token budget.
            ModelRefusalError: The model itself declined to answer.
        """

        messages = [
            Message.system(agent.instructions),
            Message.user(input),
        ]
        response = await agent.provider.complete(messages, agent.model_settings)

        match response.stop_reason:
            case "content_filter":
                raise ContentFilterError(
                    "Response was blocked by the provider's content filter."
                )
            case "max_tokens":
                configured = agent.model_settings.max_tokens
                message = (
                    f"Model hit max_tokens budget ({configured}) before completing."
                    if configured is not None
                    else (
                        "Model hit the provider's default token cap before "
                        "completing.  Set `ModelSettings.max_tokens` explicitly "
                        "to raise the limit."
                    )
                )
                raise MaxTokensExceededError(message)
            case "refusal":
                raise ModelRefusalError(response.text or "")
            case _:
                pass

        return response.text or ""
