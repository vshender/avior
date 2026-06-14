# Type-level tests

The modules in this folder are checked by the static type checkers in CI
(basedpyright and mypy over `tests/`), **not run by pytest**.  They pin
type-level contracts that runtime tests cannot observe.

Two kinds of assertion:

- **Positive cases use `assert_type`** to pin the *exact* inferred type.  This
  checks more than an annotated assignment would: the construction must both be
  accepted and infer to exactly the expected type, so a widening regression
  (a parameter drifting to `Any` or `object`, a lost variance) changes the
  inferred type and fails the assertion.

- **Negative cases are assignments that must NOT type-check**, each carrying a
  suppression for both checkers - `# type: ignore[...]` for mypy and
  `# pyright: ignore[...]` for basedpyright.  Both are configured to flag a
  suppression that no longer suppresses anything (mypy's `warn_unused_ignores`,
  basedpyright's `reportUnnecessaryTypeIgnoreComment`), so if a regression
  widened the types the rejected assignment would start type-checking, its
  now-useless suppression would be reported, and CI would fail.
