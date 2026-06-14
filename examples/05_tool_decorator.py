"""Defining tools with the `@tool` decorator.

`03_local_tool.py` and `04_tool_with_deps.py` build tools as explicit `Tool`
subclasses - the low-level form that spells out `name`, `description`,
`args_model`, and `execute`.  `@tool` is the sugar over that form: it reads the
same pieces off an ordinary typed function.  The function name becomes the tool
name, the docstring becomes the description, and the parameters become the
arguments model that drives the schema and validation.

Two tools below show both shapes:

- `get_weather` takes plain typed parameters and no context - the function form
  of the `03` tool;
- `get_balance` takes the run context first (`RunContext[Deps]`) and reads
  dependencies through it - the function form of the `04` tool.  The context is
  recognized by its type, kept out of the schema shown to the model, and
  supplied by the runner.  Because its deps type is preserved, the tool fits a
  `deps_type=Deps` agent exactly as the class form does.

Requires `ANTHROPIC_API_KEY` in the environment.

Run with: `uv run python examples/05_tool_decorator.py`
"""

import asyncio
from dataclasses import dataclass

from avior.core import Agent, ModelSettings, RunContext, Runner, tool
from avior.providers.anthropic import AnthropicProvider


@dataclass
class Deps:
    """The run's dependencies: an in-memory map of account id to cents."""

    balances: dict[str, int]


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""

    return f"It is 22 degrees Celsius and sunny in {city}."


@tool
def get_balance(ctx: RunContext[Deps], account_id: str) -> str:
    """Get the balance, in US dollars, for an account id."""

    cents = ctx.deps.balances.get(account_id)
    if cents is None:
        return f"No account found with id {account_id!r}."
    return f"${cents / 100:.2f}"


async def main() -> None:
    agent = Agent(
        instructions="You are a helpful assistant.  Use tools when relevant.",
        model_settings=ModelSettings(model="claude-haiku-4-5-20251001"),
        tools=[get_weather, get_balance],
        deps_type=Deps,
    )
    deps = Deps(balances={"acc-001": 123_45, "acc-002": 67_00})
    # `async with` owns the provider's lifecycle; the runner only borrows it.
    async with AnthropicProvider() as provider:
        runner = Runner(provider=provider)
        result = await runner.run(
            agent,
            "What's the weather in Paris, and what is the balance of acc-001?",
            deps=deps,
        )
        print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
