"""Agent with a class-based tool that reads run dependencies.

A tool often needs resources the run owns - a database client, an HTTP session,
configuration, the identity of the current user.  These are the run's
*dependencies*.  They are not baked into the agent: the agent declares their
type with `deps_type`, the caller supplies the value per run via
`Runner.run(..., deps=...)`, and each tool reads them through `ctx.deps`.

This builds on `03_local_tool.py`.  The tool there ignored its context, so its
context type was `RunContext[object]` ("needs no dependencies").  Here the tool
reads dependencies, so it is `Tool[..., Deps]` and its context is
`RunContext[Deps]` - which is what makes `ctx.deps` a typed `Deps`.

The dependency is a tiny in-memory account store, so the model can answer a
balance question it could not know on its own.  Requires `ANTHROPIC_API_KEY` in
the environment.

Run with: `uv run python examples/04_tool_with_deps.py`
"""

import asyncio
from dataclasses import dataclass

from pydantic import BaseModel

from avior.core import Agent, ModelSettings, RunContext, Runner, Tool
from avior.providers.anthropic import AnthropicProvider


@dataclass
class Deps:
    """The run's dependencies.

    A stand-in for whatever resources tools need.  Real dependencies would hold
    a database client or an HTTP session; here it is a plain in-memory mapping
    of account id to balance in US cents.
    """

    balances: dict[str, int]


class BalanceArgs(BaseModel):
    """Arguments for the `get_balance` tool."""

    account_id: str


class GetBalance(Tool[BalanceArgs, str, Deps]):
    """Look up an account balance from the run's dependencies.

    The third type parameter, `Deps`, is what types `ctx.deps`: inside
    `execute`, `ctx.deps` is a `Deps`, so reading `ctx.deps.balances` is checked
    by the type checker.
    """

    name = "get_balance"
    description = "Get the balance, in US dollars, for an account id."
    args_model = BalanceArgs

    async def execute(self, ctx: RunContext[Deps], args: BalanceArgs) -> str:
        cents = ctx.deps.balances.get(args.account_id)
        if cents is None:
            return f"No account found with id {args.account_id!r}."
        return f"${cents / 100:.2f}"


async def main() -> None:
    agent = Agent(
        instructions="You are a banking assistant.  Use tools to look up data.",
        model_settings=ModelSettings(model="claude-haiku-4-5-20251001"),
        tools=[GetBalance()],
        deps_type=Deps,
    )
    # The dependency value is supplied per run, not baked into the agent.
    deps = Deps(balances={"acc-001": 123_45, "acc-002": 67_00})
    # `async with` owns the provider's lifecycle; the runner only borrows it.
    async with AnthropicProvider() as provider:
        runner = Runner(provider=provider)
        result = await runner.run(
            agent, "What is the balance of account acc-001?", deps=deps
        )
        print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
