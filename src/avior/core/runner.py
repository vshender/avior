"""Agent runner."""

import json
from collections.abc import Mapping, Sequence
from typing import Any, assert_never

from pydantic import BaseModel, ValidationError

from avior.core.agent import Agent
from avior.core.exceptions import (
    ContentFilterError,
    MaxIterationsExceeded,
    MaxTokensExceededError,
    ModelRefusalError,
)
from avior.core.messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCallPart,
    ToolMessage,
    ToolResultError,
    ToolResultOk,
    ToolResultPart,
    UserMessage,
)
from avior.core.provider import ModelSettings, Provider
from avior.core.result import RunResult
from avior.core.tools import Tool
from avior.core.usage import Usage


class Runner:
    """Orchestrator that drives `Agent` execution against a `Provider`.

    A runner holds the `Provider` that performs the model calls.  One runner can
    drive many agents, and the same agent can be driven by different runners.

    The runner *borrows* the provider: it does not own the provider's lifecycle.
    Open and close the provider yourself, typically with `async with provider:`,
    so resource ownership stays explicit and a single provider can be shared
    across runners.  The runner is therefore not itself a context manager.
    """

    def __init__(self, *, provider: Provider) -> None:
        """Build a runner that drives agents against `provider`.

        Args:
            provider: The `Provider` that performs every model call this runner
                makes.  The runner borrows it - the caller owns its lifecycle
                (see the class docstring).
        """

        self.provider = provider

    async def run(
        self,
        agent: Agent,
        input: str | Sequence[Message],
        *,
        max_iter: int | None = None,
    ) -> RunResult:
        """Run `agent` on `input` and return the run result.

        Drives the agent loop: each iteration sends the transcript to the
        runner's provider; if the model requests tool calls they are executed
        and their results appended, and the loop continues; otherwise the
        response is final.

        Args:
            agent: The configured agent to drive.
            input: The conversation to send to the model:

                - a `str` is converted to one user message;
                - a sequence of messages continues an existing conversation, for
                  example a previous run's `RunResult.messages`.

                Before each model call, `Runner.run` prepends a `SystemMessage`
                from `agent.instructions` to the message list.  Provider
                adapters then collect all `SystemMessage`s from that list.  For
                example, an adapter may move their text into a provider-level
                `instructions` or `system` field.
            max_iter: Maximum loop iterations; defaults to `agent.max_iter`.

        Returns:
            A `RunResult` with the assistant's final text, the conversation
            transcript, and the run's token usage.

        Raises:
            ContentFilterError: An external content filter blocked the response.
            MaxTokensExceededError: Output was truncated by the token budget.
            ModelRefusalError: The model itself declined to answer.
            MaxIterationsExceeded: The loop hit `max_iter` without finishing.
        """

        input_messages: list[Message] = (
            [UserMessage.from_text(input)] if isinstance(input, str) else list(input)
        )
        tools_by_name = {tool.name: tool for tool in agent.tools}
        max_iter = max_iter if max_iter is not None else agent.max_iter
        system = SystemMessage.from_text(agent.instructions)

        generated: list[Message] = []
        usages: list[Usage] = []
        for _ in range(max_iter):
            messages: list[Message] = [system, *input_messages, *generated]
            response = await self.provider.complete(
                messages, agent.model_settings, agent.tools
            )
            message = response.message
            if response.usage is not None:
                usages.append(response.usage)

            _raise_for_error_stop(message, agent.model_settings)

            generated.append(message)

            tool_calls = [p for p in message.parts if isinstance(p, ToolCallPart)]
            if not tool_calls:
                return RunResult(
                    output=message.text or "",
                    messages=[*input_messages, *generated],
                    new_message_index=len(input_messages),
                    usage=Usage.sum(usages) if usages else None,
                )

            results = [await _run_tool(tools_by_name, call) for call in tool_calls]
            generated.append(ToolMessage(parts=results))

        raise MaxIterationsExceeded(
            f"Agent did not produce a final response within max_iter={max_iter} "
            "iterations."
        )


def _raise_for_error_stop(message: AssistantMessage, settings: ModelSettings) -> None:
    """Raise the matching `AgentRunError` for a non-continuable stop reason.

    `"stop"` and `"tool_use"` continue the run; the rest abort it.
    """

    match message.stop_reason:
        case "content_filter":
            raise ContentFilterError(
                "Response was blocked by the provider's content filter."
            )

        case "max_tokens":
            configured = settings.max_tokens
            detail = (
                f"Model hit max_tokens budget ({configured}) before completing."
                if configured is not None
                else (
                    "Model hit the provider's default token cap before "
                    "completing.  Set `ModelSettings.max_tokens` explicitly "
                    "to raise the limit."
                )
            )
            raise MaxTokensExceededError(detail)

        case "refusal":
            raise ModelRefusalError(message.text or "")

        case "stop" | "tool_use":
            pass

        case _:
            assert_never(message.stop_reason)


async def _run_tool(
    tools_by_name: Mapping[str, Tool[Any, Any]],
    call: ToolCallPart,
) -> ToolResultPart:
    """Validate, run one tool call, and wrap the outcome as a `ToolResultPart`.

    A missing tool, invalid arguments, or an exception from `execute` all become
    an `error` `ToolResult` fed back to the model, rather than aborting the run.
    """

    tool = tools_by_name.get(call.tool_name)
    if tool is None:
        return _error_result(call, f"Unknown tool: {call.tool_name!r}.")

    try:
        args = tool.args_model.model_validate(call.args)
    except ValidationError as exc:
        return _error_result(call, f"Invalid arguments: {exc}")

    try:
        result = await tool.execute(args)
    except Exception as exc:  # noqa: BLE001  (tool failures are reported to the model, not raised)
        return _error_result(call, f"Tool raised an error: {exc}")

    return ToolResultPart(
        call_id=call.call_id,
        result=ToolResultOk(content=_serialize(result)),
    )


def _error_result(call: ToolCallPart, message: str) -> ToolResultPart:
    """Build an `error` `ToolResultPart` for `call`."""

    return ToolResultPart(
        call_id=call.call_id,
        result=ToolResultError(content=message),
    )


def _serialize(result: object) -> str:
    """Render a tool result as the text fed back to the model."""

    if isinstance(result, str):
        return result
    elif isinstance(result, BaseModel):
        return result.model_dump_json()
    else:
        return json.dumps(result, default=str)
