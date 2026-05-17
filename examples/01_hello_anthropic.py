"""Hello-world agent against Anthropic's Claude.

The smallest possible avior program: one agent, one prompt, one response.
Requires `ANTHROPIC_API_KEY` in the environment.

Run with: `uv run python examples/01_hello_anthropic.py`
"""

import asyncio

from avior.core import Agent, ModelSettings, Runner
from avior.providers.anthropic import AnthropicProvider


async def main() -> None:
    agent = Agent(
        provider=AnthropicProvider(),
        instructions="You are a concise assistant.  Reply in one sentence.",
        model_settings=ModelSettings(model="claude-haiku-4-5-20251001"),
    )
    reply = await Runner.run(agent, "Say hello to avior.")
    print(reply)


if __name__ == "__main__":
    asyncio.run(main())
