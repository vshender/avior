"""One thinking configuration, three providers.

`ModelSettings.thinking` is the portable reasoning control: the same value
drives Anthropic adaptive thinking, OpenAI reasoning effort, and Gemini
thinking levels.  Beside it, `provider_options` carries raw provider-specific
settings the portable control does not cover.  Each provider reads only its
own slice, so one settings object can hold slices for several providers at
once - here, asking OpenAI and Gemini to return their reasoning summaries,
which they omit by default.  A raw slice replaces the portable mapping for its
provider, so it restates the reasoning depth; Anthropic needs no slice, since
it returns its summary by default and the portable control alone drives it.

The run asks the same question on all three providers with this one
configuration and prints each model's reasoning summary and answer.  It then
continues the Gemini transcript on Anthropic: a transcript is
provider-neutral, and the opaque reasoning artifacts a provider attaches
(signatures, encrypted reasoning) are echoed back only to the provider that
produced them and dropped for any other, so a conversation started on one
provider continues on another with thinking still on.

Requires `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and `GEMINI_API_KEY` (or
`GOOGLE_API_KEY`) in the environment.

Run with: `uv run python examples/06_cross_provider_thinking.py`
"""

import asyncio
import textwrap

from avior.core import (
    Agent,
    AssistantMessage,
    Message,
    ModelSettings,
    Runner,
    RunResult,
    ThinkingPart,
    UserMessage,
)
from avior.providers.anthropic import AnthropicProvider
from avior.providers.gemini import GeminiProvider
from avior.providers.openai_responses import OpenAIResponsesProvider

QUESTION = (
    "A farmer has chickens and rabbits: 20 heads and 56 legs in total.  "
    "How many chickens and how many rabbits?  Answer in one sentence."
)

FOLLOW_UP = (
    "The farmer sells all the rabbits and buys one goat per two rabbits "
    "sold.  How many legs are on the farm now?  Answer in one sentence."
)


def thinking_agent(model: str) -> Agent[None]:
    """Build the example agent for `model`.

    Every agent shares one thinking setup; only the model id varies.
    """

    return Agent(
        instructions="You are a careful assistant.",
        model_settings=ModelSettings(
            model=model,
            thinking="high",
            provider_options={
                "openai": {
                    "reasoning": {
                        "effort": "high",
                        "summary": "auto",
                    },
                },
                "gemini": {
                    "thinking_config": {
                        "thinking_level": "HIGH",
                        "include_thoughts": True,
                    },
                },
            },
        ),
    )


def print_run(result: RunResult) -> None:
    """Print the run's reasoning summaries and final answer."""

    for message in result.new_messages:
        if isinstance(message, AssistantMessage):
            for part in message.parts:
                if isinstance(part, ThinkingPart) and part.content:
                    thinking = part.content.rstrip()
                    print(textwrap.indent(f"[thinking]\n{thinking}", "  "))
    print(textwrap.indent(f"\n[output]\n{result.output}", "  "))
    print()


async def main() -> None:
    # `async with` owns each provider's lifecycle; the runners only borrow
    # them.
    async with (
        AnthropicProvider() as anthropic,
        OpenAIResponsesProvider() as openai,
        GeminiProvider() as gemini,
    ):
        # The same question with the same configuration on each provider.
        transcripts: dict[str, list[Message]] = {}
        for provider, model in [
            (anthropic, "claude-sonnet-4-6"),
            (openai, "gpt-5-mini"),
            (gemini, "gemini-3.6-flash"),
        ]:
            runner = Runner(provider=provider)
            result = await runner.run(thinking_agent(model), QUESTION)
            print(f"--- {model} ---")
            print_run(result)
            transcripts[provider.name] = result.messages

        # Continue the Gemini transcript on Anthropic, thinking still on.
        follow_up = [
            *transcripts[gemini.name],
            UserMessage.from_text(FOLLOW_UP),
        ]
        runner = Runner(provider=anthropic)
        result = await runner.run(thinking_agent("claude-sonnet-4-6"), follow_up)
        print("--- the Gemini transcript continued on claude-sonnet-4-6 ---")
        print_run(result)


if __name__ == "__main__":
    asyncio.run(main())
