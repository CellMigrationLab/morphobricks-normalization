"""Invariants of the statistics: pixel blocks, bin counts, ranges and tails (contract §1.2, §6.2).

:class:`~morphobricks_normalization.histogram.PlaneHistogram` and
:class:`~morphobricks_normalization.histogram.HistogramTail` check these rules
when they are made, so a histogram computed from pixels, built by a calling
brick or read from ``plane_histograms.csv`` obeys the same ones. A statistic
that breaks a rule is refused with a ``ValueError``; nothing is clipped,
rounded or filled in.

The sums of a tail are float64 accumulations of the tail's pixel values, so
the checks that compare them allow a relative rounding slack of
``MOMENT_TOLERANCE``. A float64 plane whose values exceed about ``1e154`` can
overflow its sum of squares to infinity (and its sum beyond ``1.8e308``); such
an overflow is accepted only where the tail's range makes it possible.
"""

from __future__ import annotations

import math
import numbers
import sys
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # the histogram module imports this one
    from .histogram import HistogramTail, PlaneHistogram

INT64_MAX = 2**63 - 1
# Relative rounding slack of the float64 tail sums: summing N values in float64
# loses at most about N * 2.2e-16 relative, far below this for any plane.
MOMENT_TOLERANCE = 1e-9
# Below this float64 estimate, the int64 sum of non-negative counts cannot have
# wrapped (its rounding is far smaller than the 2**62 left before 2**63).
_SAFE_FLOAT_TOTAL = 2.0**62
# The four intensity limits of a histogram, in the order they must be in.
LIMIT_NAMES = ("pixel_min", "binning_min", "binning_max", "pixel_max")


def checked_blocks(blocks: Iterable[np.ndarray], dtype: np.dtype) -> Iterator[np.ndarray]:
    """Yield every block flattened, as values of ``dtype`` in native byte order.

    Parameters
    ----------
    blocks
        The blocks of one plane.
    dtype
        The plane's pixel type, in native byte order.

    Yields
    ------
    numpy.ndarray
        The block's pixels, one dimension. A block stored in the other byte
        order is converted by value; empty blocks are passed on unchecked.

    Raises
    ------
    ValueError
        If a non-empty block holds another pixel type: reading its bytes as
        ``dtype`` would change its values and its number of pixels.
    """
    for index, block in enumerate(blocks):
        values = np.asarray(block).ravel()
        if values.size and values.dtype.newbyteorder("=") != dtype:
            raise ValueError(
                f"Block {index} of the plane holds {values.dtype.name} pixels, but the plane's "
                f"pixel type is {dtype.name}. Give every block in the plane's pixel type; "
                "converting it would change the pixel values."
            )
        if not values.dtype.isnative:
            values = values.astype(dtype)
        yield values


def pixel_count(value: object, name: str) -> int:
    """Return ``value`` as a Python int if it is a whole number of pixels.

    Raises
    ------
    ValueError
        For booleans, fractions, text, negative numbers and numbers above
        ``2**63 - 1`` (the int64 counts of the statistics).
    """
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Integral):
        raise ValueError(f"The {name} must be a whole number of pixels; got {value!r}.")
    number = int(value)
    if not 0 <= number <= INT64_MAX:
        raise ValueError(f"The {name} must be between 0 and 2**63 - 1 pixels; got {number}.")
    return number


def exact_total(counts: np.ndarray) -> int:
    """Return the exact sum of non-negative int64 counts, as a Python int (never wrapped)."""
    if float(np.sum(counts, dtype=np.float64)) < _SAFE_FLOAT_TOTAL:
        return int(counts.sum())
    return sum(counts.tolist())


def check_tail(count: object, total: object, squares: object) -> tuple[int, float, float]:
    """Check one tail's count and sums on their own; return them as int and floats.

    Raises
    ------
    ValueError
        For a count that is not a whole number of pixels, sums of an empty tail
        that are not zero, a NaN sum, a negative sum of squares, an infinite sum
        with a finite sum of squares, or sums no set of values can have
        (``count * sum_of_squares < sum**2``).
    """
    count = pixel_count(count, "pixel count of an underflow or overflow")
    total, squares = float(total), float(squares)
    if count == 0:
        if total != 0 or squares != 0:
            raise ValueError("an empty underflow or overflow must have zero sums.")
        return count, total, squares
    if math.isnan(total) or not squares >= 0:
        raise ValueError(
            f"an underflow or overflow of {count} pixel(s) has sum {total!r} and sum of squares "
            f"{squares!r}; the sum must be a number and the sum of squares 0 or more."
        )
    if math.isinf(total) and math.isfinite(squares):
        raise ValueError(
            f"an underflow or overflow has an infinite sum {total!r} but a finite sum of "
            f"squares {squares!r}."
        )
    if math.isfinite(squares) and count * squares < total * total * (1 - MOMENT_TOLERANCE):
        raise ValueError(
            f"an underflow or overflow of {count} pixel(s) has sum {total!r} and sum of squares "
            f"{squares!r}, which no {count} values can have (their variance would be negative)."
        )
    return count, total, squares


def check_histogram(histogram: PlaneHistogram) -> None:
    """Check a :class:`~morphobricks_normalization.histogram.PlaneHistogram`'s invariants.

    The counts are a one-dimensional int64 array of ``bin_count`` non-negative
    values; the total and valid pixel counts are whole numbers with ``valid <=
    total``; the bins and tails hold exactly the valid pixels. A histogram
    without valid pixels has no limits, empty bins and empty tails; one with
    valid pixels has four finite limits ``pixel_min <= binning_min <=
    binning_max <= pixel_max``, each given (none is filled in from another), a
    limit inside its extremum exactly where a tail lies beyond it, and tails
    whose sums fit their range.

    Raises
    ------
    ValueError
        Naming the rule that is broken and the values that break it.
    """
    counts = histogram.counts
    if not isinstance(counts, np.ndarray) or counts.dtype != np.int64 or counts.ndim != 1:
        raise ValueError("The bin counts must be a one-dimensional int64 array.")
    if counts.size != histogram.bin_count:
        raise ValueError(
            f"There are {counts.size} bin counts for a bin count of {histogram.bin_count}."
        )
    if counts.size and int(counts.min()) < 0:
        raise ValueError(f"A bin count cannot be negative; got {int(counts.min())}.")
    total = pixel_count(histogram.total_pixel_count, "total pixel count")
    valid = pixel_count(histogram.valid_pixel_count, "valid pixel count")
    if valid > total:
        raise ValueError(f"The plane has {valid} valid pixels but only {total} pixels in all.")
    under, over = histogram.underflow, histogram.overflow
    in_bins = exact_total(counts)
    if in_bins + under.count + over.count != valid:
        raise ValueError(
            f"The bins hold {in_bins} pixel(s) and the tails {under.count + over.count}, but "
            f"the plane has {valid} valid pixel(s)."
        )
    limits = (
        histogram.pixel_min,
        histogram.binning_min,
        histogram.binning_max,
        histogram.pixel_max,
    )
    if valid == 0:
        if any(limit is not None for limit in limits) or not (under.is_empty and over.is_empty):
            raise ValueError("A plane without valid pixels can have no intensity range or tails.")
        return
    _check_range(limits, under, over)


def _check_range(
    limits: tuple[int | float | None, ...], under: HistogramTail, over: HistogramTail
) -> None:
    """Check the four limits of a histogram with valid pixels, and its tails against them."""
    bad = [
        name
        for name, limit in zip(LIMIT_NAMES, limits, strict=True)
        if limit is None or not math.isfinite(float(limit))
    ]
    if bad:
        raise ValueError(
            "A plane with valid pixels needs its four finite intensity limits (minimum, binning "
            f"minimum, binning maximum, maximum); {', '.join(bad)} "
            f"{'is' if len(bad) == 1 else 'are'} missing or not finite, got {list(limits)}."
        )
    pixel_min, binning_min, binning_max, pixel_max = limits
    if not pixel_min <= binning_min <= binning_max <= pixel_max:
        raise ValueError(
            "The intensity limits must be ordered minimum <= binning minimum <= binning maximum "
            f"<= maximum; got {list(limits)}."
        )
    for name, tail, extremum, limit in (
        ("underflow", under, pixel_min, binning_min),
        ("overflow", over, pixel_max, binning_max),
    ):
        if (extremum == limit) != tail.is_empty:
            raise ValueError(
                f"The binning limit {limit!r} and the extremum {extremum!r} differ exactly when "
                f"pixels lie beyond the limit, but the {name} holds {tail.count} pixel(s)."
            )
        if not tail.is_empty:
            _check_tail_range(name, tail, *sorted((float(extremum), float(limit))))


def _check_tail_range(name: str, tail: HistogramTail, low: float, high: float) -> None:
    """Check that a tail's mean and sum of squares are possible for values in ``[low, high]``."""
    slack = MOMENT_TOLERANCE * max(abs(low), abs(high))
    largest = max(low * low, high * high)
    smallest = 0.0 if low <= 0 <= high else min(low * low, high * high)
    if math.isinf(tail.sum):
        possible = (
            tail.count * max(abs(low), abs(high)) * (1 + MOMENT_TOLERANCE) >= sys.float_info.max
        )
    else:
        possible = low - slack <= tail.sum / tail.count <= high + slack
    if math.isinf(tail.sum_of_squares):
        possible = possible and tail.count * largest * (1 + MOMENT_TOLERANCE) >= sys.float_info.max
    else:
        possible = possible and (
            tail.count * smallest * (1 - MOMENT_TOLERANCE)
            <= tail.sum_of_squares
            <= tail.count * largest * (1 + MOMENT_TOLERANCE)
        )
    if not possible:
        raise ValueError(
            f"The {name} of {tail.count} pixel(s) between {low!r} and {high!r} has sum "
            f"{tail.sum!r} and sum of squares {tail.sum_of_squares!r}, which values in that "
            "range cannot have."
        )
