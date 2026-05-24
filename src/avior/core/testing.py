"""Test doubles for `avior.core` primitives.

This module is part of the public API. Users can import `StubProvider` to
test their own agents without making real LLM calls.

`StubProvider` is a stub-and-spy in the xUnit taxonomy: it returns
caller-programmed responses (stub) and records every invocation (spy) for
later assertion. It is named "stub", not "fake", because it does not
implement the underlying LLM contract - it merely replays scripted output.
"""

from collections.abc import Awaitable, Callable, Sequence
from inspect import isawaitable
from typing import NamedTuple, Self

from avior.core.messages import AssistantMessage, Message, TextPart
from avior.core.provider import ModelSettings, Provider

type StubResponse = str | AssistantMessage
"""A scripted response.

A `str` is sugar for a single-`TextPart` `AssistantMessage` with
`stop_reason="stop"`.
"""

type StubCallable = Callable[
    [Sequence[Message], ModelSettings],
    StubResponse | Awaitable[StubResponse],
]
"""The canonical callable form a `StubProvider` dispatches to.

Receives the conversation and the model settings; returns a scripted
response either synchronously or as an awaitable.
"""

type StubPredicate = Callable[[Sequence[Message]], bool]
"""A predicate over the conversation, used by `from_predicates`."""


class StubCall(NamedTuple):
    """A single recorded invocation of a `StubProvider`.

    Stored in the order calls were made. Use field access (`.messages`,
    `.settings`) for clarity; tuple unpacking (`msgs, settings = call`)
    also works.

    `messages` and `settings` are stored by reference, not by snapshot.
    If the calling code mutates the same list or settings object after
    the call returns, the recorded history reflects those mutations.
    For predictable assertions, treat the recorded values as read-only
    after `complete()` returns, or construct a fresh list / settings per
    call.
    """

    messages: Sequence[Message]
    settings: ModelSettings


def _normalize_response(response: StubResponse) -> AssistantMessage:
    """Coerce a `str` to a single-`TextPart` `AssistantMessage`.

    Args:
        response: Either a raw string or an already-constructed
            `AssistantMessage`.

    Returns:
        An `AssistantMessage` ready to be returned from `Provider.complete`.
    """

    if isinstance(response, str):
        return AssistantMessage(parts=[TextPart(text=response)], stop_reason="stop")

    return response


class StubProvider(Provider):
    """Programmable test double for the `Provider` abstraction.

    Three construction forms cover the common test scenarios. All forms
    record their invocations in `.calls` for test assertions.

    1. **Canonical callable** - `StubProvider(lambda msgs, settings: ...)`
       gives full control over the response, including inspecting the
       conversation and settings. The callable may be sync or async, and
       may return either a `str` (wrapped as a single-`TextPart`
       `AssistantMessage`) or a fully-formed `AssistantMessage`.

       ```python
       def respond(messages: Sequence[Message], _: ModelSettings) -> str:
           return f"echo: {messages[-1].text}"

       provider = StubProvider(respond)
       ```

    2. **Sequential canned responses** - `StubProvider.from_responses([...])`
       returns each entry in order, one per call. Raises `AssertionError`
       once exhausted.

       ```python
       provider = StubProvider.from_responses(["first", "second"])
       ```

    3. **Predicate dispatch** - `StubProvider.from_predicates([(pred, resp),
       ...])` returns the response paired with the first matching
       predicate. Raises `AssertionError` if no predicate matches. The
       predicate signature is `Callable[[Sequence[Message]], bool]`; for
       settings-aware dispatch, use the canonical callable form instead.

       ```python
       provider = StubProvider.from_predicates([
           (lambda msgs: msgs[-1].text == "ping", "pong"),
           (lambda msgs: msgs[-1].text == "hello", "hi"),
       ])
       ```

    After running an agent against the stub, inspect `provider.calls`
    to assert what was sent to the model:

       ```python
       assert len(provider.calls) == 2
       assert provider.calls[-1].messages[-1].text == "hello"
       assert provider.calls[-1].settings.model == "claude-3-5-sonnet"
       ```
    """

    def __init__(self, func: StubCallable) -> None:
        """Construct a stub from the canonical dispatch callable.

        Args:
            func: Receives `(messages, settings)` and returns a
                `StubResponse` either directly or as an awaitable.
        """

        super().__init__()
        self._func = func
        self.calls: list[StubCall] = []

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

        def func(
            _messages: Sequence[Message],
            _settings: ModelSettings,
        ) -> AssistantMessage:
            nonlocal index
            if index >= len(snapshot):
                raise AssertionError(
                    f"StubProvider.from_responses exhausted after {index} "
                    f"call(s); was constructed with {len(snapshot)} response(s)."
                )
            response = snapshot[index]
            index += 1
            return _normalize_response(response)

        return cls(func)

    @classmethod
    def from_predicates(
        cls,
        pairs: Sequence[tuple[StubPredicate, StubResponse]],
    ) -> Self:
        """Construct a stub that dispatches by matching the first predicate.

        Predicates receive the conversation only, not the model settings.
        For settings-aware dispatch (e.g. branching on `settings.model`),
        use the canonical callable form (`StubProvider(func)`) directly.

        Args:
            pairs: Sequence of `(predicate, response)` pairs. Predicates
                receive the full message list; first match wins.

        Returns:
            A `StubProvider` whose `complete` evaluates predicates in
            order and returns the paired response on first match. Raises
            `AssertionError` if no predicate matches.
        """

        snapshot = list(pairs)

        def func(
            messages: Sequence[Message],
            _settings: ModelSettings,
        ) -> AssistantMessage:
            for predicate, response in snapshot:
                if predicate(messages):
                    return _normalize_response(response)
            raise AssertionError(
                "StubProvider.from_predicates: no predicate matched the "
                f"incoming conversation of {len(messages)} message(s)."
            )

        return cls(func)

    async def complete(
        self,
        messages: Sequence[Message],
        settings: ModelSettings,
    ) -> AssistantMessage:
        """Record the call, dispatch, and return the scripted response.

        Args:
            messages: Conversation transcript passed to the stub.
            settings: Per-call model invocation settings.

        Returns:
            The scripted `AssistantMessage` produced by the dispatch
            callable.
        """

        self.calls.append(StubCall(messages=messages, settings=settings))

        result = self._func(messages, settings)
        if isawaitable(result):
            result = await result

        return _normalize_response(result)

    async def aclose(self) -> None:
        """No-op: the stub holds no real resources to release."""

        pass
