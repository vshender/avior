"""Integration smoke tests against the real OpenAI Responses API.

Gated by `OPENAI_API_KEY`; skipped when the environment variable is unset.
Not run by the default `make test` / unit-test path - invoked separately via
`make test-integration` and a dedicated GitHub Actions workflow.
"""

import os

import pytest

from avior.core import Agent, ModelSettings, Runner
from avior.core.exceptions import MaxTokensExceededError
from avior.providers.openai_responses import OpenAIResponsesProvider

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


async def test_runner_run_against_openai_returns_non_empty_text(
    openai_responses_provider: OpenAIResponsesProvider,
) -> None:
    """`Runner.run` against real OpenAI Responses returns a non-empty reply.

    End-to-end smoke: avior `Agent` -> `OpenAIResponsesProvider` -> `openai`
    SDK -> HTTP -> OpenAI Responses API -> decoded `Message`.  Asserts only on
    the transport contract (a non-empty string), not on response content.
    """

    # GIVEN an agent using the real OpenAI Responses provider and a cheap model
    agent = Agent(
        provider=openai_responses_provider,
        instructions="Reply with one short word.",
        model_settings=ModelSettings(
            model="gpt-4.1-nano",
            max_tokens=256,
        ),
    )

    # WHEN we run a trivial prompt
    result = await Runner.run(agent, "Say hello.")

    # THEN we get a non-empty text response
    assert result.output.strip() != ""


async def test_runner_run_raises_max_tokens_exceeded_against_openai(
    openai_responses_provider: OpenAIResponsesProvider,
) -> None:
    """`Runner.run` raises `MaxTokensExceededError` when the token cap is hit.

    Confirms end-to-end mapping: OpenAI Responses returns `status="incomplete"`
    with `incomplete_details.reason="max_output_tokens"` -> provider sets
    canonical `stop_reason="max_tokens"` -> Runner raises.
    """

    # GIVEN an agent with `model_settings.max_tokens` too small to complete.
    # (OpenAI Responses API enforces `max_output_tokens >= 16`; 16 is enough
    # to trigger truncation for a long-story prompt.)
    agent = Agent(
        provider=openai_responses_provider,
        instructions="Write a long story.",
        model_settings=ModelSettings(
            model="gpt-4.1-nano",
            max_tokens=16,
        ),
    )

    # WHEN `Runner.run` is invoked
    # THEN `MaxTokensExceededError` is raised
    with pytest.raises(MaxTokensExceededError):
        await Runner.run(agent, "Tell me a story.")
