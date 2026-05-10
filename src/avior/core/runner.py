"""Agent runner."""

from avior.core.agent import Agent
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
        """

        messages = [
            Message.system(agent.instructions),
            Message.user(input),
        ]
        response = await agent.provider.complete(messages, agent.model_settings)
        return response.text or ""
