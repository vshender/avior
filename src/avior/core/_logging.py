"""Library logging setup.

Importing this module attaches a `NullHandler` to the top-level `avior` logger
as a side effect; `avior.core.__init__` imports it for that purpose.  Without
this, applications that do not configure logging would see "No handlers could be
found" warnings the first time avior emits a log record.

The module lives in `avior.core` (the earliest load point in this namespace
package - avior has no `avior/__init__.py`) but attaches the handler at the
library-wide `avior` level so propagation covers all subpackages from one place.

Logger hierarchy follows Python module paths:

- `avior` - root logger for the library (this module attaches the `NullHandler`
  here).
- `avior.core.runner`, ... - one per core module.
- `avior.providers.anthropic`, `avior.providers.openai_responses`, ... - one per
  provider module.

Users configure logging at any granularity, e.g.:

    import logging
    logging.getLogger("avior").setLevel(logging.DEBUG)    # all
    logging.getLogger("avior.providers.anthropic").setLevel(logging.INFO)
"""

import logging

logging.getLogger("avior").addHandler(logging.NullHandler())
