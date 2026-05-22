"""Integration smoke test against the real OpenAI Responses API.

Gated by `OPENAI_API_KEY`; skipped when the environment variable is unset.
Not run by the default `make test` / unit-test path - invoked separately via
`make test-integration` and a dedicated GitHub Actions workflow.
"""

import os

import pytest

from avior.core import Agent, ModelSettings, Runner
from avior.providers.openai_responses import OpenAIResponsesProvider

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


async def test_runner_run_against_openai_returns_non_empty_text() -> None:
    """`Runner.run` against real OpenAI Responses returns a non-empty reply.

    End-to-end smoke: avior `Agent` -> `OpenAIResponsesProvider` -> `openai`
    SDK -> HTTP -> OpenAI Responses API -> decoded `Message`.  Asserts only on
    the transport contract (a non-empty string), not on response content.
    """

    # GIVEN an agent using the real OpenAI Responses provider and a cheap model
    agent = Agent(
        provider=OpenAIResponsesProvider(),
        instructions="Reply with one short word.",
        model_settings=ModelSettings(
            model="gpt-4.1-nano",
            max_tokens=256,
        ),
    )

    # WHEN we run a trivial prompt
    reply = await Runner.run(agent, "Say hello.")

    # THEN we get a non-empty text response
    assert isinstance(reply, str)
    assert reply.strip() != ""
