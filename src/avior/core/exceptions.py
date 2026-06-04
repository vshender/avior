"""Exception types raised by `avior.core` and its subpackages.

Three categories sit under a common `AviorError` root:

- `ProviderError` and subclasses cover transport / SDK failures - the provider
  could not fulfill its contract (HTTP error, network failure, schema mismatch).
- `AgentRunError` and subclasses cover failures during an agent run other
  than transport-level provider failures.
- `ConfigurationError` covers invalid configuration detected when avior objects
  are constructed - a programmer error to fix in code, not a runtime condition
  to catch and handle.
"""


class AviorError(Exception):
    """Common root for every exception raised by avior."""


class ProviderError(AviorError):
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
    """The provider returned a successful response that could not be decoded.

    Indicates a schema mismatch between the provider's wire format and the SDK's
    response model - typically because the provider rolled out a new response
    shape the installed SDK doesn't yet understand.  The fix is usually to
    upgrade the SDK.
    """


class AgentRunError(AviorError):
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
    """The provider's content filter blocked the response.

    The filter is a server-side moderation classifier run by the provider
    (OpenAI's safety system, Azure OpenAI's configurable content filter) that
    screens the model's generated output against policy and zeroes it out on
    violation.  The HTTP call still succeeds at the transport level, but no
    usable content reaches the caller.  Surfaces on OpenAI's
    `incomplete_details.reason == "content_filter"`.

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


class ConfigurationError(AviorError):
    """Invalid configuration detected when an avior object is constructed.

    Covers programmer errors caught at construction time - for example, an
    `Agent` given two tools that share a name.  Unlike `ProviderError` and
    `AgentRunError`, which are runtime conditions a caller may catch and handle,
    a configuration error signals a bug in how avior is set up: fix it in code
    rather than handling it at runtime.
    """
