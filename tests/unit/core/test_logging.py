"""Tests for `avior.core._logging`."""

import importlib
import logging


def test_avior_root_logger_has_null_handler_attached() -> None:
    """Importing `avior.core` attaches a `NullHandler` to the `avior` logger.

    Guards the standard library-convention setup against accidental removal:
    without it, applications that do not configure logging would see
    "No handlers could be found" warnings the first time avior emits.
    """

    # GIVEN avior.core has been imported (triggers the side-effect setup)
    importlib.import_module("avior.core")

    # WHEN we inspect the top-level avior logger's handlers
    handler_types = [type(h) for h in logging.getLogger("avior").handlers]

    # THEN at least one `NullHandler` is attached
    assert logging.NullHandler in handler_types
