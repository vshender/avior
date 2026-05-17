"""Integration smoke test against the real Anthropic API.

Gated by `ANTHROPIC_API_KEY`; skipped when the environment variable is unset.
Not run by the default `make test` / unit-test path - invoked separately via
`make test-integration` and a dedicated GitHub Actions workflow.
"""

import os

import pytest

from avior.core import Agent, ModelSettings, Runner
from avior.providers.anthropic import AnthropicProvider

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


async def test_runner_run_against_anthropic_returns_non_empty_text() -> None:
    """`Runner.run` against real Anthropic returns a non-empty assistant reply.

    End-to-end smoke: avior `Agent` -> `AnthropicProvider` -> `anthropic` SDK ->
    HTTP -> Anthropic Messages API -> decoded `Message`.  Asserts only on the
    transport contract (a non-empty string), not on response content.
    """

    # GIVEN an agent using the real Anthropic provider and a cheap model
    agent = Agent(
        provider=AnthropicProvider(),
        instructions="Reply with one short word.",
        model_settings=ModelSettings(
            model="claude-haiku-4-5-20251001",
            max_tokens=64,
        ),
    )

    # WHEN we run a trivial prompt
    reply = await Runner.run(agent, "Say hello.")

    # THEN we get a non-empty text response
    assert isinstance(reply, str)
    assert reply.strip() != ""
