"""Agent runner."""

import json
from collections.abc import Mapping, Sequence
from typing import Any, assert_never, overload

from pydantic import BaseModel, ValidationError
from typing_extensions import TypeVar

from avior.core.agent import Agent
from avior.core.context import RunContext
from avior.core.exceptions import (
    ContentFilterError,
    EmptyInputError,
    MaxIterationsExceeded,
    MaxTokensExceededError,
    MissingDependenciesError,
    ModelRefusalError,
    OrphanedToolResultError,
    UnansweredToolCallError,
    UnexpectedModelBehaviorError,
)
from avior.core.messages import (
    AssistantMessage,
    Message,
    ToolCallPart,
    ToolMessage,
    ToolResultPart,
    UserMessage,
)
from avior.core.provider import ModelSettings, Provider
from avior.core.result import RunResult
from avior.core.tools import Tool
from avior.core.usage import Usage
from avior.core.warnings import RunWarning, WarningHandler, log_warning

# Binds the agent's deps type to the `deps` argument of `run`.  Explicit
# `TypeVar` (Python 3.12 has no `def run[Deps = None]` default syntax); a
# function-scoped type parameter, so variance does not apply.
Deps = TypeVar("Deps", default=None)

# Sentinel for an omitted `deps` argument, kept distinct from a real `deps=None`
# value: some declared deps types accept `None` (an empty `Protocol`, or `None`
# itself), so a passed `deps=None` is valid and must not read as "not passed".
_MISSING: Any = object()


class Runner:
    """Orchestrator that drives `Agent` execution against a `Provider`.

    A runner holds the `Provider` that performs the model calls.  One runner can
    drive many agents, and the same agent can be driven by different runners.

    The runner *borrows* the provider: it does not own the provider's lifecycle.
    Open and close the provider yourself, typically with `async with provider:`,
    so resource ownership stays explicit and a single provider can be shared
    across runners.  The runner is therefore not itself a context manager.
    """

    def __init__(
        self,
        *,
        provider: Provider,
        warning_handlers: Sequence[WarningHandler] | None = None,
    ) -> None:
        """Build a runner that drives agents against `provider`.

        Args:
            provider: The `Provider` that performs every model call this runner
                makes.  The runner borrows it - the caller owns its lifecycle
                (see the class docstring).
            warning_handlers: Functions that process each `RunWarning` the run
                produces, for example by logging it.  Called once per warning,
                in order; if a handler raises, the exception propagates and
                aborts the run.  `None` installs a single handler that logs each
                warning; `[]` drops warnings silently.
        """

        self.provider = provider
        self._warning_handlers: tuple[WarningHandler, ...] = (
            (log_warning,) if warning_handlers is None else tuple(warning_handlers)
        )

    # `deps` is optional for an agent that declares no dependencies, i.e. when
    # `Deps` is:
    #
    # - `None` - no `deps_type` and no tools to infer from;
    # - `object` - no `deps_type`, but tools that need no dependencies make
    #   the checker infer `Deps` as `object`.
    @overload
    async def run(
        self,
        agent: Agent[None] | Agent[object],
        input: str | Sequence[Message],
        *,
        max_iter: int | None = None,
    ) -> RunResult: ...

    # `deps` is required once the agent's `Deps` is concrete (set via
    # `deps_type` or inferred from the tools), so this overload has no default
    # for it.
    @overload
    async def run(
        self,
        agent: Agent[Deps],
        input: str | Sequence[Message],
        *,
        deps: Deps,
        max_iter: int | None = None,
    ) -> RunResult: ...

    async def run(
        self,
        agent: Agent[Any],
        input: str | Sequence[Message],
        *,
        deps: Any = _MISSING,
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

                The conversation carries no system prompt: `agent.instructions`
                goes to the provider separately on each model call, not into
                this list.
            deps: The dependencies passed to the agent's tools through their
                `RunContext`, of the agent's declared `Deps` type.  Required
                when `Deps` is a concrete type (set on the agent via
                `deps_type`, or inferred from its tools); may be omitted when
                `Deps` is `None` or `object`.

                In addition to that type-level requirement, a runtime check
                guards untyped callers: if the agent sets `deps_type` and `deps`
                is missing, the run stops at once with a clear error before any
                model call (see Raises), rather than letting a tool fail later
                on `deps=None`.  The check is presence-only - it does not verify
                `deps` matches the declared type (the type checker's job) - and
                a `Deps` inferred without `deps_type` is not covered (see
                `Agent.deps_type`).
            max_iter: Maximum loop iterations; defaults to `agent.max_iter`.

        Returns:
            A `RunResult` for the run.

        Raises:
            MissingDependenciesError: `deps_type` is set but `deps` is missing.
            EmptyInputError: The input carries no content to send.
            UnansweredToolCallError: A tool call in the input has no matching
                tool result.
            OrphanedToolResultError: A tool result references a tool call absent
                from the input.
            ContentFilterError: An external content filter blocked the response.
            MaxTokensExceededError: Output was truncated by the token budget.
            ModelRefusalError: The model itself declined to answer.
            UnexpectedModelBehaviorError: The model terminated abnormally
                without a usable response - the `"error"` stop reason, or a
                response with no content (no text and no tool calls).
            MaxIterationsExceeded: The loop hit `max_iter` without finishing.
            Exception: A warning handler that raises aborts the run; its
                exception propagates unchanged.
        """

        # `deps` is required only for a concrete dependency type.  `None`/
        # `NoneType` (no deps) and `object` (deps-agnostic) need no value, and
        # the overloads treat `Agent[None]` / `Agent[object]` as deps-optional -
        # so the guard must skip them too, or it would reject a call the type
        # checker accepted.
        #
        # It checks only that `deps` is present, never that it matches
        # `deps_type`: a runtime `isinstance` check would force every protocol
        # deps type to be `@runtime_checkable`, and even then verify only
        # attribute presence, not the structural type the checker enforces.
        #
        # "Present" is tracked by the `_MISSING` sentinel, not `deps is None`:
        # a real `deps=None` is a valid value for a deps type that accepts it.
        deps_type = agent.deps_type
        if (
            deps_type is not None
            and deps_type is not object
            and deps_type is not type(None)
            and deps is _MISSING
        ):
            # Statically `deps_type` is a class, so it has `__name__`.  But a
            # caller bypassing the type checker could pass a runtime value that
            # isn't a plain class - a union like `int | str` has no `__name__` -
            # so read the name defensively: the error message must not itself
            # raise an `AttributeError`.
            deps_name = getattr(deps_type, "__name__", str(deps_type))
            raise MissingDependenciesError(
                f"Agent declares deps_type={deps_name!r}, so "
                "`Runner.run` requires a `deps` argument."
            )
        if deps is _MISSING:
            deps = None

        input_messages: list[Message] = (
            [UserMessage.from_text(input)] if isinstance(input, str) else list(input)
        )
        self._validate_input(input_messages)
        tools_by_name = {tool.name: tool for tool in agent.tools}
        max_iter = max_iter if max_iter is not None else agent.max_iter

        # Blank instructions (empty or whitespace-only) carry no system prompt,
        # so normalize them to `None`.
        system_prompt = (
            agent.instructions
            if agent.instructions and agent.instructions.strip()
            else None
        )

        generated: list[Message] = []
        usages: list[Usage] = []
        warnings: list[RunWarning] = []
        for run_step in range(1, max_iter + 1):
            messages: list[Message] = [*input_messages, *generated]
            response = await self.provider.complete(
                messages,
                agent.model_settings,
                tools=agent.tools,
                system_prompt=system_prompt,
            )
            message = response.message

            if response.usage is not None:
                usages.append(response.usage)

            # Run warnings through the handlers before the error-stop check, so
            # a degraded response's warnings are observed even if it then fails.
            for warning in response.warnings:
                for handler in self._warning_handlers:
                    handler(warning)

            warnings.extend(response.warnings)

            self._raise_for_error_stop(message, agent.model_settings)

            if not message.parts:
                # The model produced no content at all - no text and no tool
                # calls.  That is a degenerate response, not a usable answer, so
                # surface it rather than returning an empty result that hides a
                # provider glitch.
                raise UnexpectedModelBehaviorError(
                    "The model returned an empty response with no content."
                )

            generated.append(message)

            tool_calls = [p for p in message.parts if isinstance(p, ToolCallPart)]
            if not tool_calls:
                return RunResult(
                    output=message.text or "",
                    messages=[*input_messages, *generated],
                    new_message_index=len(input_messages),
                    usage=Usage.sum(usages) if usages else None,
                    warnings=warnings,
                )

            results = [
                await _run_tool(tools_by_name, call, deps, run_step)
                for call in tool_calls
            ]
            generated.append(ToolMessage(parts=results))

        raise MaxIterationsExceeded(
            f"Agent did not produce a final response within max_iter={max_iter} "
            "iterations."
        )

    @staticmethod
    def _validate_input(messages: list[Message]) -> None:
        """Reject caller input that violates avior's input contract.

        Raised before the first model call, so a contract violation surfaces at
        the call site as a clear avior usage error rather than reaching the
        model, where the same fault might fail opaquely on one backend or be
        silently accepted as a meaningless run on another.  Each fault raises
        the matching `InvalidInputError` subclass.

        Only faults that hold regardless of the model API are checked; how each
        model API constrains transcript shape (role ordering, alternation) is
        enforced by that API and surfaced through the `Provider`, not checked
        here.
        """

        if not messages:
            raise EmptyInputError("The input has no messages to send.")

        # Check each message for content and collect the tool-call and
        # tool-result `call_id`s.
        call_ids: set[str] = set()
        result_ids: set[str] = set()
        for message in messages:
            if Runner._is_empty_message(message):
                raise EmptyInputError(
                    "The input has a message with no content to send."
                )

            if isinstance(message, AssistantMessage):
                call_ids.update(
                    part.call_id
                    for part in message.parts
                    if isinstance(part, ToolCallPart)
                )
            elif isinstance(message, ToolMessage):
                result_ids.update(part.call_id for part in message.parts)

        # Tool calls and results must pair up by `call_id`: every result answers
        # a call, and every call is answered.
        if unanswered := call_ids - result_ids:
            raise UnansweredToolCallError(
                f"Tool call {sorted(unanswered)[0]!r} has no matching tool result "
                "in the input."
            )
        if orphaned := result_ids - call_ids:
            raise OrphanedToolResultError(
                f"Tool result references call_id {sorted(orphaned)[0]!r}, which "
                "matches no tool call in the input."
            )

    @staticmethod
    def _is_empty_message(message: Message) -> bool:
        """Whether a message carries no content to send."""

        match message:
            case UserMessage():
                return not (message.text or "").strip()
            case AssistantMessage() | ToolMessage():
                return not message.parts
            case _:
                assert_never(message)

    @staticmethod
    def _raise_for_error_stop(
        message: AssistantMessage,
        settings: ModelSettings,
    ) -> None:
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
                    f"Model hit max_tokens budget ({configured}) before "
                    "completing.  Increase max_tokens or shorten the input."
                    if configured is not None
                    else (
                        "Model hit the provider's maximum output limit before "
                        "completing.  Shorten the input or split the task."
                    )
                )
                raise MaxTokensExceededError(detail)

            case "refusal":
                raise ModelRefusalError(message.text or "")

            case "error":
                raise UnexpectedModelBehaviorError(
                    "Model terminated abnormally without a usable response."
                )

            case "stop" | "tool_use":
                pass

            case _:
                assert_never(message.stop_reason)


async def _run_tool(
    tools_by_name: Mapping[str, Tool[Any, Any, Any]],
    call: ToolCallPart,
    deps: object,
    run_step: int,
) -> ToolResultPart:
    """Validate, run one tool call, and wrap the outcome as a `ToolResultPart`.

    A missing tool, invalid arguments, or an exception from `execute` all become
    an `error` `ToolResult` fed back to the model, rather than aborting the run.
    """

    tool = tools_by_name.get(call.tool_name)
    if tool is None:
        return ToolResultPart.error(call.call_id, f"Unknown tool: {call.tool_name!r}.")

    try:
        args = tool.args_model.model_validate(call.args)
    except ValidationError as exc:
        return ToolResultPart.error(call.call_id, f"Invalid arguments: {exc}")

    ctx = RunContext[Any](
        deps=deps,
        tool_name=call.tool_name,
        tool_call_id=call.call_id,
        run_step=run_step,
    )
    try:
        result = await tool.execute(ctx, args)
    except Exception as exc:  # noqa: BLE001  (tool failures are reported to the model, not raised)
        return ToolResultPart.error(call.call_id, f"Tool raised an error: {exc}")

    return ToolResultPart.ok(call.call_id, _serialize(result))


def _serialize(result: object) -> str:
    """Render a tool result as the text fed back to the model."""

    if isinstance(result, str):
        return result
    elif isinstance(result, BaseModel):
        return result.model_dump_json()
    else:
        return json.dumps(result, default=str)
