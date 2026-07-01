"""Warnings produced during a run, and their handlers.

A `RunWarning` records a non-fatal problem found during a run, such as a setting
the chosen model could not honor.  A `Runner`'s warning handlers process each
warning as it occurs, and every warning is also collected on
`RunResult.warnings` for inspection after the run.
"""

import logging
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue, computed_field

logger = logging.getLogger(__name__)

# Bound the requested value shown in `message` so a large raw config does not
# bloat log lines; the full value stays in `setting_value`.
_MAX_VALUE_REPR_LEN = 60


class UnsupportedSettingRunWarning(BaseModel):
    """A setting the chosen model could not honor."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    type: Literal["unsupported_setting"] = "unsupported_setting"
    """Discriminator identifying this warning kind."""

    setting_name: str
    """Name of the setting that was not honored, for example `"thinking"`."""

    setting_value: JsonValue
    """The requested value that was not honored, for example `"high"`."""

    reason: str | None = None
    """The reason the setting could not be honored, beyond the generic message -
    for example that a thinking budget did not fit the requested `max_tokens`.
    `None` when there is no detail to add.
    """

    provider: str
    """Name of the provider whose model could not honor the setting."""

    model: str
    """The model that could not honor the setting."""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def message(self) -> str:
        """Human-readable description, derived from the other fields."""

        value = repr(self.setting_value)
        if len(value) > _MAX_VALUE_REPR_LEN:
            value = value[:_MAX_VALUE_REPR_LEN] + "..."
        detail = f": {self.reason}" if self.reason is not None else ""
        return (
            f"Model {self.model!r} ({self.provider}) could not honor the "
            f"{self.setting_name!r} setting (requested {value}){detail}; "
            f"it was dropped from the request."
        )


type RunWarning = UnsupportedSettingRunWarning
"""A non-fatal problem found during a run.

Currently a single kind, `UnsupportedSettingRunWarning`.
"""


type WarningHandler = Callable[[RunWarning], None]
"""A function a `Runner` calls once per `RunWarning`.

The handler decides what to do with the warning: returning normally lets the run
continue, while raising propagates and aborts the run.
"""


def log_warning(warning: RunWarning) -> None:
    """Log `warning` at the warning level, without aborting the run.

    The default warning handler.
    """

    logger.warning("%s", warning.message)
