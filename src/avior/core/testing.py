"""Test doubles for `avior.core` primitives.

This module is part of the public API.  Users can import `StubProvider` to
test their own agents without making real LLM calls.

`StubProvider` is a stub-and-spy in the xUnit taxonomy: it returns
caller-programmed responses (stub) and records every invocation (spy) for
later assertion.
"""

from collections.abc import Awaitable, Callable, Sequence
from inspect import isawaitable
from typing import Any, NamedTuple, Self

from avior.core.messages import AssistantMessage, Message, TextPart
from avior.core.provider import ModelSettings, Provider, ProviderResponse
from avior.core.tools import Tool

type StubResponse = str | AssistantMessage | ProviderResponse
"""A scripted response, in one of three forms:

- A `str` is sugar for a single-`TextPart` `AssistantMessage` with
  `stop_reason="stop"`, tagged with the stub's `provider_name`.
- An `AssistantMessage` is sugar for a `ProviderResponse` that wraps it with
  no call metadata.
- A `ProviderResponse` is used as-is, so a test can script the call metadata
  it asserts on.
"""


class StubCall(NamedTuple):
    """A record of one `StubProvider` invocation - the parameters the call was
    made with.

    Appended to `.calls` in call order for later assertion.  Use field access
    (`.messages`, `.settings`, `.tools`, `.system_prompt`); tuple unpacking
    (`msgs, settings, tools, system_prompt = call`) also works.

    `messages`, `settings`, and `tools` are stored by reference, not by
    snapshot.  If the calling code mutates the same list or settings object
    after the call returns, the recorded history reflects those mutations.
    For predictable assertions, treat the recorded values as read-only
    after `complete()` returns, or construct a fresh list / settings per
    call.
    """

    messages: Sequence[Message]
    settings: ModelSettings
    tools: Sequence[Tool[Any, Any, Any]] = ()
    system_prompt: str | None = None


type StubCallable = Callable[[StubCall], StubResponse | Awaitable[StubResponse]]
"""The canonical callable form a `StubProvider` dispatches to.

Receives the `StubCall` for the invocation - the parameters the call was made
with - and returns a scripted response either synchronously or as an awaitable.
"""

type StubPredicate = Callable[[StubCall], bool]
"""A predicate over the `StubCall`, used by `from_predicates`."""


class StubProvider(Provider):
    """Programmable test double for the `Provider` abstraction.

    Three construction forms cover the common test scenarios.  All forms
    record their invocations in `.calls` for test assertions.

    The canonical callable and predicates receive the `StubCall` for the
    invocation, so dispatch can branch on any parameter the call was made with.
    A stub is usually written for one agent's scenario, so in practice that
    branching keys on the messages.

    1. **Canonical callable** - `StubProvider(lambda call: ...)` gives full
       control over the response from the whole `StubCall`.  The callable may
       be sync or async, and may return any `StubResponse`: a `str` (wrapped as
       a single-`TextPart` `AssistantMessage`), a fully-formed
       `AssistantMessage`, or a complete `ProviderResponse` (to script the call
       metadata).

       ```python
       def respond(call: StubCall) -> str:
           return f"echo: {call.messages[-1].text}"

       provider = StubProvider(respond)
       ```

    2. **Sequential canned responses** - `StubProvider.from_responses([...])`
       returns each entry in order, one per call.  Raises `AssertionError`
       once exhausted.

       ```python
       provider = StubProvider.from_responses(["first", "second"])
       ```

    3. **Predicate dispatch** - `StubProvider.from_predicates([(pred, resp),
       ...])` returns the response paired with the first matching predicate.
       Raises `AssertionError` if no predicate matches.  Each predicate is a
       `Callable[[StubCall], bool]`.

       ```python
       provider = StubProvider.from_predicates([
           (lambda call: call.messages[-1].text == "ping", "pong"),
           (lambda call: call.messages[-1].text == "hello", "hi"),
       ])
       ```

    After running an agent against the stub, inspect `provider.calls`
    to assert what was sent to the model:

       ```python
       assert len(provider.calls) == 2
       assert provider.calls[-1].messages[-1].text == "hello"
       assert provider.calls[-1].settings.model == "claude-3-5-sonnet"
       ```

    The stub never generates tool calls itself.  To exercise the Runner's
    tool loop, script a tool call (an `AssistantMessage` with a `ToolCallPart`
    and `stop_reason="tool_use"`), then the final reply:

       ```python
       provider = StubProvider.from_responses([
           AssistantMessage(
               parts=[
                   ToolCallPart(
                       call_id="c1",
                       tool_name="get_weather",
                       args={"city": "Paris"},
                   )
               ],
               stop_reason="tool_use",
           ),
           "It's sunny in Paris.",
       ])
       ```
    """

    def __init__(self, func: StubCallable) -> None:
        """Construct a stub from the canonical dispatch callable.

        Args:
            func: Receives the `StubCall` for each invocation and returns a
                `StubResponse` either directly or as an awaitable.
        """

        super().__init__()
        self._func = func
        self.calls: list[StubCall] = []

    @property
    def name(self) -> str:
        """The provider's canonical name."""

        return "stub"

    @classmethod
    def from_responses(cls, responses: Sequence[StubResponse]) -> Self:
        """Construct a stub that returns each response in order, one per call.

        Args:
            responses: Sequence of scripted responses.  `str` entries are
                wrapped as single-`TextPart` `AssistantMessage`s.

        Returns:
            A `StubProvider` that pops one response per `complete` call.
            Raises `AssertionError` if called more times than responses
            were supplied.
        """

        # Coerce to a concrete list: normalizes one-shot iterables
        # (generators, map objects) and snapshots the sequence so the
        # caller cannot append/replace entries between calls.
        snapshot = list(responses)
        index = 0

        def func(_call: StubCall) -> StubResponse:
            nonlocal index
            if index >= len(snapshot):
                raise AssertionError(
                    f"StubProvider.from_responses exhausted after {index} "
                    f"call(s); was constructed with {len(snapshot)} response(s)."
                )

            response = snapshot[index]
            index += 1
            return response

        return cls(func)

    @classmethod
    def from_predicates(
        cls,
        pairs: Sequence[tuple[StubPredicate, StubResponse]],
    ) -> Self:
        """Construct a stub that dispatches by matching the first predicate.

        Args:
            pairs: Sequence of `(predicate, response)` pairs.  Each predicate
                receives the `StubCall` (so it can match on any parameter the
                call was made with); first match wins.

        Returns:
            A `StubProvider` whose `complete` evaluates predicates in
            order and returns the paired response on first match.  Raises
            `AssertionError` if no predicate matches.
        """

        snapshot = list(pairs)

        def func(call: StubCall) -> StubResponse:
            for predicate, response in snapshot:
                if predicate(call):
                    return response

            raise AssertionError(
                "StubProvider.from_predicates: no predicate matched the "
                f"incoming conversation of {len(call.messages)} message(s)."
            )

        return cls(func)

    async def complete(
        self,
        messages: Sequence[Message],
        settings: ModelSettings,
        *,
        tools: Sequence[Tool[Any, Any, Any]] = (),
        system_prompt: str | None = None,
    ) -> ProviderResponse:
        """Record the call, dispatch, and return the scripted response.

        Args:
            messages: Conversation transcript passed to the stub.
            settings: Per-call model invocation settings.
            tools: Tools offered to the model.
            system_prompt: The system prompt for the call.

        Returns:
            The scripted response, normalized to a `ProviderResponse`.

        The call is recorded as a `StubCall` in `.calls`, and that same
        `StubCall` is passed to the dispatch callable, which may branch on any
        of its fields.  The stub never generates tool calls itself - to drive
        the Runner's tool loop, script a response `AssistantMessage` whose
        parts include a `ToolCallPart`.
        """

        call = StubCall(
            messages=messages,
            settings=settings,
            tools=tools,
            system_prompt=system_prompt,
        )
        self.calls.append(call)

        result = self._func(call)
        if isawaitable(result):
            result = await result

        return self._normalize_response(result)

    async def aclose(self) -> None:
        """No-op: the stub holds no real resources to release."""

        pass

    def _normalize_response(self, response: StubResponse) -> ProviderResponse:
        """Coerce any scripted `StubResponse` to a `ProviderResponse`.

        Args:
            response: A raw string, an `AssistantMessage`, or a fully-formed
                `ProviderResponse`.

        Returns:
            A `ProviderResponse` ready to be returned from `Provider.complete`,
            with no call metadata (usage, response id).  The message built from
            the `str` form is tagged with the stub's `provider_name`; the
            `AssistantMessage` and `ProviderResponse` forms are used as given.
        """

        if isinstance(response, str):
            message = AssistantMessage(
                parts=[TextPart(text=response)],
                stop_reason="stop",
                provider_name=self.name,
            )
            return ProviderResponse(message=message)
        elif isinstance(response, AssistantMessage):
            return ProviderResponse(message=response)
        else:
            return response
