"""Whole-number bin limits of integer planes, in int64 where that is exact.

Bin ``i`` of ``n`` bins over ``width`` values starts at offset ``ceil(i * width / n)``
(or ``min(i, width)`` when ``width <= n``, one value per bin). Until v0.7.0 these limits,
the sums of per-value counts into bins and the bin of a value were found with loops over
every bin in Python. Here the same integers come from NumPy int64 arithmetic wherever no
step can overflow, and from the exact Python integers of v0.7.0 everywhere else
(64-bit planes with a wide range), so no count, limit or float built from them changes.
"""

from __future__ import annotations

import numpy as np

INT64_STARTS_LIMIT = 2**62
"""Bin starts are int64 when ``bin_count * width`` is below this: no product can overflow."""

EXACT_FLOAT_TOTAL = 2.0**52
"""Per-value counts are summed in int64 while their float64 total is below this.

Every partial sum is then below ``2**53``, so the float64 sums used until v0.7.0 were
exact and equal the int64 sums; beyond it (never for one plane in memory) they are kept.
"""


def integer_bin_starts(width: int, bin_count: int) -> list[int]:
    """Offsets at which bins ``0 .. bin_count`` start (the last entry is ``width``)."""
    if width <= bin_count:
        # One value per bin from the start; the remaining bins stay empty.
        return [min(i, width) for i in range(bin_count + 1)]
    return [-((-i * width) // bin_count) for i in range(bin_count + 1)]


def integer_bin_starts_int64(width: int, bin_count: int) -> np.ndarray | None:
    """Return :func:`integer_bin_starts` as an int64 array, or None if int64 could overflow.

    The same integers from the same formula: NumPy's integer floor division rounds towards
    minus infinity, as Python's does. None when ``bin_count * width`` reaches ``2**62``
    (64-bit planes with a wide range), where only Python integers are exact.
    """
    if bin_count * max(width, 1) >= INT64_STARTS_LIMIT:
        return None
    index = np.arange(bin_count + 1, dtype=np.int64)
    if width <= bin_count:
        return np.minimum(index, width)
    return -((-index * width) // bin_count)


def integer_bin_offsets(
    binning_min: int, binning_max: int, bin_count: int
) -> tuple[np.ndarray, np.ndarray] | tuple[list[int], list[int]]:
    """Return every bin's first and last value, as offsets from ``binning_min``.

    int64 arrays when both limits are Python integers (as in every histogram computed or
    read by this library) and the starts fit int64; otherwise the exact Python integers of
    v0.7.0, so that results built from them keep their values and types.
    """
    width = binning_max + 1 - binning_min
    starts = integer_bin_starts_int64(width, bin_count) if type(width) is int else None
    if starts is not None:
        return starts[:-1], starts[1:] - 1
    starts = integer_bin_starts(width, bin_count)
    return starts[:-1], [s - 1 for s in starts[1:]]


def search_starts(width: int, bin_count: int) -> np.ndarray:
    """Return the starts of bins ``0 .. bin_count - 1`` as uint64, for a sorted search."""
    starts = integer_bin_starts_int64(width, bin_count)
    if starts is None:  # a wide 64-bit range: computed as exact Python integers
        return np.array(integer_bin_starts(width, bin_count)[:-1], dtype=np.uint64)
    return starts[:-1].astype(np.uint64)


def counts_from_per_value(per_value: np.ndarray, bin_count: int) -> np.ndarray:
    """Sum per-value counts (offsets ``0 .. width - 1``) into ``bin_count`` bins over that width.

    Exact int64 sums (differences of the cumulative counts at the bin starts) while the
    total is below :data:`EXACT_FLOAT_TOTAL`; otherwise the float64 sums of v0.7.0.
    """
    width = per_value.size
    starts = integer_bin_starts_int64(width, bin_count)
    if starts is not None and float(np.sum(per_value, dtype=np.float64)) < EXACT_FLOAT_TOTAL:
        cumulative = np.zeros(width + 1, dtype=np.int64)
        np.cumsum(per_value, out=cumulative[1:])
        return cumulative[starts[1:]] - cumulative[starts[:-1]]
    starts = integer_bin_starts(width, bin_count)
    bin_of_value = np.searchsorted(np.array(starts[:-1]), np.arange(width), side="right")
    counts = np.bincount(bin_of_value - 1, weights=per_value, minlength=bin_count)
    return np.rint(counts).astype(np.int64)


def integer_bin_index(offset: int, width: int, bin_count: int) -> int:
    """Return ``bisect_right(integer_bin_starts(width, n), offset, hi=n) - 1`` in O(1).

    Exact integer arithmetic: -1 below the range. With no more values than bins, value
    ``offset`` is in bin ``offset`` and a value above the range in the last bin. Otherwise
    bin ``i`` starts at ``ceil(i * width / n)``, which is at most ``offset`` exactly when
    ``i <= offset * n / width``.
    """
    if offset < 0:
        return -1
    if width <= bin_count:
        return offset if offset < width else bin_count - 1
    return min(offset * bin_count // width, bin_count - 1)
