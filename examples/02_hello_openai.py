"""Hello-world agent against OpenAI's Responses API.

The same agent as `01_hello_anthropic.py`; only the provider handed to the
runner changes.  Requires `OPENAI_API_KEY` in the environment.

Run with: `uv run python examples/02_hello_openai.py`
"""

import asyncio

from avior.core import Agent, ModelSettings, Runner
from avior.providers.openai_responses import OpenAIResponsesProvider


async def main() -> None:
    agent = Agent(
        instructions="You are a concise assistant.  Reply in one sentence.",
        model_settings=ModelSettings(model="gpt-5-nano"),
    )

    # `async with` owns the provider's lifecycle; the runner only borrows it.
    async with OpenAIResponsesProvider() as provider:
        runner = Runner(provider=provider)
        result = await runner.run(agent, "Say hello to avior.")
        print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
