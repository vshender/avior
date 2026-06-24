"""Exception types `avior` defines.

Every exception avior *defines* descends from `AviorError` - so `except
AviorError` scopes to avior's own errors.  It is not a catch-everything: avior
can still surface exceptions it does not define - e.g. a Pydantic
`ValidationError` from a bad `ModelSettings`, or an `ImportError` for a missing
optional provider.  Below the root, two branches split avior's own errors by
how a caller should respond:

- `AviorOperationalError` - a run or a provider call failed: an external
  condition or an outcome of the run.  A caller may catch and handle these
  (retry, raise a limit, fall back, surface to the user).  Two kinds:

  - `ProviderError` and subclasses - transport / SDK failures: the provider
    could not fulfill its contract (HTTP error, network failure, schema
    mismatch).
  - `AgentRunError` and subclasses - failures during a run other than provider
    failures (the loop never reached a final answer, the model refused, output
    was filtered).

- `AviorUsageError` - avior was used incorrectly: a bug to fix in code, not a
  condition to catch and handle.  Includes:

  - `ConfigurationError` - invalid setup of an avior object, found at
    construction (for example, two tools sharing a name).
  - `MissingDependenciesError` - a deps-typed agent was run without its `deps`.

For handling logic, catch a specific operational type, not the bare root.
"""


class AviorError(Exception):
    """Common root for every exception avior defines.

    It has two branches: `AviorOperationalError` (conditions a caller may
    handle) and `AviorUsageError` (bugs to fix in code).  Catching `AviorError`
    is not enough to handle every failure - avior also surfaces exceptions it
    does not define, such as a Pydantic `ValidationError`, which a caller may
    need to catch too.
    """


class AviorOperationalError(AviorError):
    """Base class for operational failures during avior's work.

    Running an agent or calling the provider failed - an external condition
    (the provider was down) or an outcome of the run (the loop never reached a
    final answer, the model refused, the token budget was too small).  These
    surface at runtime, and a caller may catch and handle them (retry, raise a
    limit, fall back, surface to the user) - unlike `AviorUsageError`, which
    signals a bug to fix in code.
    """


class ProviderError(AviorOperationalError):
    """Base class for all `Provider` failures.

    `Provider` implementations translate vendor-specific SDK exceptions into
    this hierarchy so caller code stays portable.  The original SDK exception
    is preserved as `__cause__`.  Subclasses cover common categories; this
    base also catches unusual SDK failures that don't fit one.
    """


class ProviderConnectionError(ProviderError):
    """Network-level failure: no HTTP response received.

    Covers DNS resolution failures, TCP/TLS handshake errors, request timeouts,
    dropped connections, and similar transport problems.  These are typically
    transient and amenable to retry with backoff, though the underlying SDK may
    have already retried internally before this error surfaced.
    """


class ProviderHTTPError(ProviderError):
    """The provider returned a 4xx or 5xx HTTP response.

    Field-discriminated: callers branch on `status_code` for retry and
    user-facing logic (e.g. `429` - adaptive backoff, `401` - surface "check
    your API key", `5xx` - retry with longer delay).
    """

    status_code: int
    """The HTTP status code returned by the provider."""

    def __init__(self, message: str, *, status_code: int) -> None:
        """Initialize with `message` and the HTTP `status_code`."""

        super().__init__(message)
        self.status_code = status_code


class ProviderResponseValidationError(ProviderError):
    """The provider returned a successful response avior could not decode or map
    into the canonical message shape.

    Two causes:

    - a schema mismatch between the provider's wire format and the provider
      SDK's response model - typically because the provider rolled out a new
      response shape the installed SDK doesn't yet understand (fix: upgrade the
      SDK);
    - the response decoded fine but carries content avior does not yet model in
      the canonical IR - a content kind the adapter has no mapping for (fix:
      extend avior to handle it).

    Either way the transport call succeeded; the failure is in turning the
    response into avior's `Message`.
    """


class AgentRunError(AviorOperationalError):
    """Base class for failures during an agent run."""


class MaxIterationsExceeded(AgentRunError):
    """The agent loop ran more iterations than `max_iter` without finishing.

    One iteration is a single LLM call plus the tool calls its response
    requested.  Hitting the cap usually means the LLM kept calling tools without
    converging on a final answer - a loop, or a `max_iter` set too low.  Raise
    the limit or inspect the tool behavior.
    """


class MaxTokensExceededError(AgentRunError):
    """The model hit the configured token budget before completing its reply.

    Surfaces when `ModelSettings.max_tokens` (or the provider's default cap) is
    reached and the response was truncated.  Typically actionable: raise
    `max_tokens` or shorten the prompt.
    """


class ContentFilterError(AgentRunError):
    """The provider's content filter blocked the exchange.

    The filter is a server-side moderation classifier run by the provider that
    screens content against policy and zeroes it out on a violation.  It can
    block either the prompt before generation or the generated response.  The
    HTTP call still succeeds at the transport level, but no usable content
    reaches the caller.  Surfaces across providers - for example OpenAI's
    `incomplete_details.reason == "content_filter"`, or Gemini's safety /
    recitation finish or a prompt blocked before generation.

    Distinct from `ModelRefusalError`: the filter is moderation infrastructure
    intervening *between* the model and the caller, not the model itself
    deciding to refuse.  Not retryable as-is; revise the prompt or relax content
    policy.
    """


class ModelRefusalError(AgentRunError):
    """The model declined to answer the request.

    Surfaces on Anthropic's `stop_reason == "refusal"` and OpenAI's
    `ResponseOutputRefusal` content part.  Distinct from `ContentFilterError`:
    the model itself decided to refuse.  Not retryable as-is; revise the prompt.

    The model's refusal text - its own explanation for declining - is preserved
    on `refusal_text` for logging, display, or programmatic inspection.
    """

    refusal_text: str
    """The model-provided refusal text.  Empty string when the response carried
    no refusal content (defensive default; should not happen in practice when
    this exception is raised)."""

    def __init__(self, refusal_text: str) -> None:
        """Initialize with the model-provided refusal text.

        The refusal text is used as the exception's string form so that
        `str(exc)` shows the model's own words.
        """

        super().__init__(refusal_text)
        self.refusal_text = refusal_text


class UnexpectedModelBehaviorError(AgentRunError):
    """The model terminated abnormally without a usable response.

    Surfaces on the canonical `"error"` stop reason - a provider-reported
    abnormal termination where the model produced neither usable content nor a
    valid tool call (for example Gemini's `MALFORMED_FUNCTION_CALL` /
    `UNEXPECTED_TOOL_CALL`: the model tried to call a tool but produced
    malformed tool-call data).  The HTTP call succeeded and the response
    decoded, so this is a run failure, not a `ProviderError`.

    Distinct from `ContentFilterError` / `ModelRefusalError`, which are specific
    deliberate outcomes; this is the catch-all for "the model misbehaved".  The
    provider-specific reason is not carried on the canonical stop reason; a
    provider may log it.
    """


class AviorUsageError(AviorError):
    """Base class for using avior incorrectly.

    Signals a bug in how avior is set up or called - a programmer error to fix
    in code, not a runtime condition to catch and handle.  Descends from
    `AviorError` so a boundary net still sees it, but handling logic should not
    catch it: fix the code.
    """


class ConfigurationError(AviorUsageError):
    """Invalid configuration of an avior object, detected at construction.

    For example, an `Agent` given two tools that share a name.
    """


class MissingDependenciesError(AviorUsageError):
    """A deps-typed agent was run without the `deps` it declared.

    `Runner.run` raises this before any model call when the agent declares a
    concrete `deps_type` but no `deps` argument is supplied.  Pass `deps`, or
    drop `deps_type` if the agent needs none; see `Agent.deps_type`.
    """
