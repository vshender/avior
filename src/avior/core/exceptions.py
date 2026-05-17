"""Exception types raised by `avior.core` and its subpackages."""


class ProviderError(Exception):
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
