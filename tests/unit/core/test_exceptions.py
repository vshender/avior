"""Tests for `avior.core.exceptions`."""

import pickle

from avior.core.exceptions import ProviderHTTPError


def test_provider_http_error_survives_pickling() -> None:
    """`ProviderHTTPError` round-trips through pickle with its full state."""

    # GIVEN a provider HTTP error carrying a status code and an attached note
    note = "while polling the batch endpoint"
    error = ProviderHTTPError("rate limited", status_code=429)
    error.add_note(note)

    # WHEN it is round-tripped through pickle
    restored = pickle.loads(pickle.dumps(error))

    # THEN the message, the status code, and the note survive
    assert str(restored) == "rate limited"
    assert restored.status_code == 429
    assert restored.__notes__ == [note]


class _StatusError(ProviderHTTPError):
    """A stand-in for a user-defined `ProviderHTTPError` subclass."""


def test_provider_http_error_subclass_keeps_its_type_when_pickled() -> None:
    """A `ProviderHTTPError` subclass unpickles as the subclass.

    A caller's `except` clause for the subclass must keep matching after a
    pickle round-trip.
    """

    # GIVEN an error of a subclass of `ProviderHTTPError`
    error = _StatusError("throttled", status_code=429)

    # WHEN it is round-tripped through pickle
    restored = pickle.loads(pickle.dumps(error))

    # THEN the restored error keeps the subclass type and its state
    assert type(restored) is _StatusError
    assert restored.status_code == 429
