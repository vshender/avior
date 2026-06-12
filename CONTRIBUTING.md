# Contributing to avior

> **Status: pre-alpha.**  This guide covers only conventions that are easy to
> trip over.  It will grow as the project does.

## Type checking

avior is checked by **both** mypy and basedpyright, each in `strict` mode, over
`src` and `tests`.  A change must pass both; the two disagree often enough that
satisfying one is not enough.

### Suppressions are split per checker

The two checkers disagree on which lines need a suppression — for example,
pydantic's `@computed_field` trips mypy but not basedpyright.  To keep a
suppression that one checker needs from reading as "unnecessary" to the other,
each checker owns its own suppression comment:

- **mypy** reads `# type: ignore[code]`.
- **basedpyright** reads `# pyright: ignore[rule]`.

This is enforced by `enableTypeIgnoreComments = false` in `pyproject.toml`, which
turns off basedpyright's default support for `# type: ignore`.  The practical
consequences:

- A bare `# type: ignore` does **not** suppress anything for basedpyright.  To
  silence a basedpyright diagnostic, you must use `# pyright: ignore[rule]`.
- When both checkers flag the same line, put both comments on it:

  ```python
  some_call()  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
  ```

- Always name the specific code/rule.  A blanket `# pyright: ignore` (no rule)
  hides future diagnostics too.

### Don't leave stale suppressions

Both checkers flag a suppression that no longer suppresses anything — mypy via
`warn_unused_ignores` (part of `strict`), basedpyright via
`reportUnnecessaryTypeIgnoreComment`.  So a suppression left behind after the
underlying error is gone fails CI.  Remove suppressions when you remove the
errors they covered.

This same machinery is load-bearing in `tests/typing/`: the negative
deps-compatibility fixtures assert that incompatible tool/agent pairs are
*rejected* by deliberately suppressing the resulting error.  If a regression
widened the types so the pair started type-checking, the now-unnecessary
suppression fails the build.  Treat those suppressions as assertions — don't
"clean them up" by deleting the lines.
