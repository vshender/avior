"""Shared fixtures for integration tests."""

from collections.abc import AsyncIterator

import pytest_asyncio

from avior.providers.anthropic import AnthropicProvider
from avior.providers.openai_responses import OpenAIResponsesProvider


@pytest_asyncio.fixture
async def anthropic_provider() -> AsyncIterator[AnthropicProvider]:
    """Yield a fresh `AnthropicProvider`, closed on teardown.

    Reads `ANTHROPIC_API_KEY` from the environment.  Per-test scope: each test
    gets its own provider and its own underlying `AsyncAnthropic` client.
    """

    async with AnthropicProvider() as provider:
        yield provider


@pytest_asyncio.fixture
async def openai_responses_provider() -> AsyncIterator[OpenAIResponsesProvider]:
    """Yield a fresh `OpenAIResponsesProvider`, closed on teardown.

    Reads `OPENAI_API_KEY` from the environment.  Per-test scope: each test
    gets its own provider and its own underlying `AsyncOpenAI` client.
    """

    async with OpenAIResponsesProvider() as provider:
        yield provider
