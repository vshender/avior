"""Agent with a local tool against Anthropic's Claude.

The agent is given one tool.  The model decides to call it, avior validates the
arguments and runs `execute`, feeds the result back, and the model answers using
it.  Requires `ANTHROPIC_API_KEY` in the environment.

The tool is defined as an explicit `Tool` subclass: you set `name`,
`description`, and `args_model`, and implement `execute` yourself.  This is the
low-level form that every tool ultimately is.

Run with: `uv run python examples/03_local_tool.py`
"""

import asyncio

from pydantic import BaseModel

from avior.core import Agent, ModelSettings, Runner, Tool
from avior.providers.anthropic import AnthropicProvider


class WeatherArgs(BaseModel):
    """Arguments for the `get_weather` tool."""

    city: str


class GetWeather(Tool[WeatherArgs, str]):
    """A toy weather tool returning a canned report for any city."""

    name = "get_weather"
    description = "Get the current weather for a city."
    args_model = WeatherArgs

    async def execute(self, args: WeatherArgs) -> str:
        return f"It is 22 degrees Celsius and sunny in {args.city}."


async def main() -> None:
    agent = Agent(
        instructions="You are a helpful assistant.  Use tools when relevant.",
        model_settings=ModelSettings(model="claude-haiku-4-5-20251001"),
        tools=[GetWeather()],
    )
    # `async with` owns the provider's lifecycle; the runner only borrows it.
    async with AnthropicProvider() as provider:
        runner = Runner(provider=provider)
        result = await runner.run(agent, "What's the weather in Paris?")
        print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
