"""Token-usage accounting for model calls and runs.

`Usage` normalizes the token counts that providers report for a single model
call into one cross-provider convention, and the same type represents usage
summed across several calls: element-wise aggregation (`+` / `Usage.sum`) folds
per-call usage into one combined value.

Normalization convention
-------------------------

avior follows one fixed convention so that the same field means the same thing
regardless of provider (the provider adapter is responsible for converting):

- `input_tokens` and `output_tokens` are **totals that already include their
  sub-slices**:

  - `input_tokens` counts every input token, including any served from or
    written to the prompt cache.
  - `output_tokens` counts every generated token, including reasoning.

  A provider that reports a sub-slice *separately* (rather than inside the
  total) has it folded in by its adapter - e.g. Anthropic's `input_tokens`
  excludes cache, and Gemini's output count excludes its `thoughts`; both are
  widened to true totals.

- `cache_read_tokens`, `cache_write_tokens`, and `reasoning_tokens` are
  **sub-slices** that break those totals down.  The cache slices are parts of
  `input_tokens`; `reasoning_tokens` is part of `output_tokens`.

This makes `total_tokens` simply `input_tokens + output_tokens`, and makes
counts summable across providers without double-counting.

Zero vs unknown
---------------

All counts are non-negative.  `input_tokens` / `output_tokens` and the two
cache sub-slices are always concrete integers: a provider that uses no caching
means `0` cached tokens, so adapters coalesce a provider's "absent" cache count
to `0` (absence genuinely is zero there).

`reasoning_tokens` is the one nullable field.  `None` means the provider does
**not itemize** reasoning out of `output_tokens` - e.g. Anthropic extended
thinking is generated and billed inside `output_tokens` but not broken out, so
the count is genuinely unknown, which is different from a reported `0` (e.g.
OpenAI on a non-reasoning turn).  Providers that do itemize reasoning (OpenAI,
Gemini `thoughts`) report a number.  Aggregation preserves this: if any summed
call's reasoning is unknown, the aggregate's `reasoning_tokens` is `None` too.
"""

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, computed_field


class Usage(BaseModel):
    """Normalized token counts for a model interaction.

    Used both for a single model call (on `ProviderResponse`) and for usage
    summed across several calls; instances aggregate element-wise with `+` or
    `Usage.sum`.  See the module docstring for the normalization convention and
    the zero-vs-unknown treatment of `reasoning_tokens`.
    """

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(ge=0)
    """Total input (prompt) tokens for the call, **including** any tokens read
    from or written to the prompt cache.  Equals `(non-cached input) +
    cache_read_tokens + cache_write_tokens`.
    """

    output_tokens: int = Field(ge=0)
    """Total output (completion) tokens generated, **including** reasoning
    tokens.  Equals `(visible output) + (reasoning_tokens or 0)`.  Providers
    that report reasoning as a separate addend (e.g. Gemini `thoughts`) have it
    folded in by their adapter so this stays a true total.
    """

    cache_read_tokens: int = Field(default=0, ge=0)
    """Sub-slice of `input_tokens` served from the prompt cache.  `0` when no
    cache was read; adapters coalesce a provider's unreported cache count to `0`
    (no caching genuinely means zero cached tokens).
    """

    cache_write_tokens: int = Field(default=0, ge=0)
    """Sub-slice of `input_tokens` written to the prompt cache (cache creation).
    `0` when the provider reports no cache writes or has no separate write
    accounting (e.g. OpenAI; Gemini cache creation is a separate call).
    """

    reasoning_tokens: int | None = Field(default=None, ge=0)
    """Sub-slice of `output_tokens` spent on internal reasoning.  A number when
    the provider itemizes it (OpenAI reasoning models, Gemini `thoughts`);
    `None` when the provider does not break reasoning out of `output_tokens`
    (e.g. Anthropic extended thinking) - distinct from a reported `0`.
    """

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        """Total tokens for the call: `input_tokens + output_tokens`.

        Derived rather than stored, so it cannot drift from its components.
        Under the normalization convention this is the true total; for providers
        that report their own total (OpenAI, Gemini) it equals that number
        exactly.
        """

        return self.input_tokens + self.output_tokens

    def __add__(self, other: object) -> "Usage":
        """Aggregate two usages element-wise.

        Totals and the cache sub-slices add directly.  `reasoning_tokens` uses
        unknown-aware addition: `None` means the provider did not itemize
        reasoning (e.g. Anthropic) - an *unknown* amount, not zero.  If either
        side is unknown the result is `None`: treating it as `0` would
        understate real reasoning usage and present a partial sum as an
        authoritative total.
        """

        if not isinstance(other, Usage):
            return NotImplemented

        if self.reasoning_tokens is None or other.reasoning_tokens is None:
            reasoning_tokens = None
        else:
            reasoning_tokens = self.reasoning_tokens + other.reasoning_tokens

        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    @classmethod
    def sum(cls, usages: Iterable["Usage"]) -> "Usage":
        """Aggregate an iterable of usages into one snapshot.

        Returns a zero-token `Usage` when `usages` is empty.
        """

        total: Usage | None = None
        for usage in usages:
            total = usage if total is None else total + usage

        return total if total is not None else cls(input_tokens=0, output_tokens=0)
