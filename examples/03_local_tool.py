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
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from avior.core import Agent, ModelSettings, RunContext, Runner, Tool
from avior.providers.anthropic import AnthropicProvider


class WeatherArgs(BaseModel):
    """Arguments for the `get_weather` tool."""

    # The LLM reads each field's description from the schema.  Two ways to set
    # one: `use_attribute_docstrings` turns the docstring under a field into its
    # description, and `Field(description=...)` sets it directly.  A field that
    # uses both keeps the `Field` one.
    model_config = ConfigDict(use_attribute_docstrings=True)

    city: str
    """The city to report the weather for."""

    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius", description="Temperature units to report in."
    )


class GetWeather(Tool[WeatherArgs, str]):
    """A toy weather tool returning a canned report for any city."""

    name = "get_weather"
    description = "Get the current weather for a city."
    args_model = WeatherArgs

    # This tool reads no dependencies, so its context is `RunContext[object]`.
    # `object` means "requires nothing", so the tool fits any agent whatever
    # its deps are.  It matches the third parameter of `Tool[WeatherArgs, str]`,
    # which is omitted here and so defaults to `object`.
    async def execute(self, ctx: RunContext[object], args: WeatherArgs) -> str:
        temperature = 22 if args.units == "celsius" else 72
        return f"It is {temperature} degrees {args.units} and sunny in {args.city}."


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
