"""Tests for `avior.core.usage`."""

import pytest

from avior.core.usage import Usage


def test_total_tokens_is_input_plus_output() -> None:
    """`total_tokens` is derived from `input_tokens + output_tokens`."""

    # GIVEN a usage with known input and output
    usage = Usage(input_tokens=10, output_tokens=4)

    # THEN total_tokens is their sum (no separate stored field)
    assert usage.total_tokens == 14


def test_add_sums_totals_and_cache_slices() -> None:
    """`+` sums input/output totals and the cache sub-slices directly."""

    # GIVEN two usages reporting totals and cache slices
    a = Usage(input_tokens=10, output_tokens=2, cache_read_tokens=3)
    b = Usage(input_tokens=5, output_tokens=3, cache_read_tokens=1)

    # WHEN they are added
    total = a + b

    # THEN totals and the cache slice are summed
    assert total.input_tokens == 15
    assert total.output_tokens == 5
    assert total.cache_read_tokens == 4
    assert total.total_tokens == 20


def test_add_reasoning_sums_when_both_sides_itemize() -> None:
    """`reasoning_tokens` adds when both operands report a number."""

    # GIVEN two usages that both itemize reasoning
    a = Usage(input_tokens=10, output_tokens=5, reasoning_tokens=4)
    b = Usage(input_tokens=5, output_tokens=2, reasoning_tokens=3)

    # THEN their reasoning counts add
    assert (a + b).reasoning_tokens == 7


# One usage that itemizes reasoning and one that leaves it unknown (`None`),
# for the unknown-aware aggregation cases below.
_REASONING_ITEMIZED = Usage(input_tokens=10, output_tokens=5, reasoning_tokens=4)
_REASONING_UNKNOWN = Usage(input_tokens=5, output_tokens=2, reasoning_tokens=None)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (_REASONING_ITEMIZED, _REASONING_UNKNOWN),
        (_REASONING_UNKNOWN, _REASONING_ITEMIZED),
        (_REASONING_UNKNOWN, _REASONING_UNKNOWN),
    ],
    ids=["known+unknown", "unknown+known", "unknown+unknown"],
)
def test_add_reasoning_is_unknown_if_either_side_is_unknown(
    left: Usage, right: Usage
) -> None:
    """`reasoning_tokens` is `None` (unknown) when either operand is unknown.

    `None` means the provider did not itemize reasoning (e.g. Anthropic) - an
    unknown amount - so a sum that swallowed it would present a partial total
    as authoritative.
    """

    # THEN combining with an unknown (in either order) yields unknown
    assert (left + right).reasoning_tokens is None


def test_sum_over_empty_is_zero_tokens_reasoning_none() -> None:
    """`Usage.sum([])` is zero tokens with reasoning unknown."""

    # WHEN summing an empty iterable
    total = Usage.sum([])

    # THEN counts are 0 (no calls used no tokens) and reasoning is None
    # (no call reported a reasoning count)
    assert total.input_tokens == 0
    assert total.output_tokens == 0
    assert total.cache_read_tokens == 0
    assert total.reasoning_tokens is None


def test_sum_folds_iterable() -> None:
    """`Usage.sum` folds a sequence of per-call usages into one snapshot."""

    # GIVEN three per-call usages
    usages = [
        Usage(input_tokens=10, output_tokens=1),
        Usage(input_tokens=20, output_tokens=2),
        Usage(input_tokens=30, output_tokens=3),
    ]

    # WHEN summed
    total = Usage.sum(usages)

    # THEN the result is the element-wise total
    assert total.input_tokens == 60
    assert total.output_tokens == 6


def test_sum_reasoning_sums_when_all_calls_itemize() -> None:
    """`Usage.sum` adds reasoning when every call itemizes it.

    This also guards `sum`'s fold-from-first design: had it instead folded from
    a synthetic zero `Usage` (whose `reasoning_tokens` defaults to `None`), the
    unknown-aware `+` would poison the result to `None` even when every call
    reports a number - so this must stay a real sum.
    """

    # GIVEN calls that all itemize reasoning
    usages = [
        Usage(input_tokens=10, output_tokens=2, reasoning_tokens=2),
        Usage(input_tokens=10, output_tokens=3, reasoning_tokens=3),
    ]

    # THEN their reasoning counts sum
    assert Usage.sum(usages).reasoning_tokens == 5


def test_sum_reasoning_unknown_if_any_call_unknown() -> None:
    """`Usage.sum` reasoning is `None` if any call did not itemize reasoning."""

    # GIVEN calls where one did not itemize reasoning
    usages = [
        Usage(input_tokens=10, output_tokens=2, reasoning_tokens=2),
        Usage(input_tokens=5, output_tokens=1, reasoning_tokens=None),
    ]

    # THEN the aggregate reasoning is unknown
    assert Usage.sum(usages).reasoning_tokens is None
