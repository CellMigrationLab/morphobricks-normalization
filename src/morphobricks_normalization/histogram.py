"""Per-plane intensity histograms.

Every plane is binned over its own observed range, so a plane's histogram
depends only on its own pixels and never has to be rebuilt when other images
are added to the experiment.

Integer planes
    The values ``pixel_min .. pixel_max`` (``W`` values) are split into
    ``bin_count`` bins with whole-number limits. Bin ``i`` holds the values
    ``pixel_min + ceil(i * W / n)`` up to ``pixel_min + ceil((i + 1) * W / n) - 1``,
    so bin widths differ by at most one value. When ``W <= bin_count`` the
    histogram is exact instead: bin ``i`` holds only the value ``pixel_min + i``
    and bins ``W`` onwards stay empty. The representative value of a bin is the
    midpoint of the integers it holds.

Float planes
    The finite range ``[pixel_min, pixel_max]`` is split into ``bin_count``
    equal-width bins. Bins are half-open ``[a, b)`` except the last, which also
    holds the maximum (up to floating-point rounding at bin edges). NaN and
    +/-infinity are not counted. A constant plane puts every pixel in bin 0. The
    representative value of a bin is its midpoint.

Robust range (contract §1.4)
    With ``HistogramSettings(histogram_range="robust")``, a histogram that would
    not be exact is binned between the plane's quantiles at ``robust_quantiles``
    instead of its minimum and maximum, with the same rules. The pixels below
    and above are kept as :class:`HistogramTail` counts and sums (underflow and
    overflow). Exact histograms never change.

Every pixel in a bin lies within half a bin width of its bin's representative
value, and every pixel of a tail between the tail's mean and the plane's
extremum, which bounds the error of any percentile computed from these
histograms (see :attr:`PlaneHistogram.max_representative_error`).

Quantiles for the robust range and the range check
(:func:`histogram_with_range_check`) are exact inverse-CDF order statistics of
the finite pixels, found by radix selection over the pixels' ordered bit
patterns, 16 bits per pass. The same code serves whole planes and planes read
in blocks, so the results never depend on how a plane was read.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import numpy as np

from .dtypes import check_supported, default_bin_count, is_integer, validate_bin_count
from .ranges import CENTRAL_PERCENTILES, HistogramSettings, RangeCheck

# Above this many distinct values, integer planes are binned with a search over
# bin starts instead of a full per-value count.
_PER_VALUE_COUNT_LIMIT = 2**24
# Radix selection resolves this many bits of the ordered pixel bits per pass.
_DIGIT_BITS = 16
_DIGIT_VALUES = 1 << _DIGIT_BITS
# Order statistics followed at once: two robust limits, two central percentiles.
_MAX_TARGETS = 4


@dataclass(frozen=True)
class HistogramTail:
    """Pixels outside a histogram's binning range: below it (underflow) or above it (overflow).

    Only the robust range (contract §1.4) leaves pixels outside the bins; with
    the default min-max range both tails are empty.

    Attributes
    ----------
    count
        Number of pixels.
    sum, sum_of_squares
        Sum of their values and of their squared values, in float64.
    """

    count: int = 0
    sum: float = 0.0
    sum_of_squares: float = 0.0

    @property
    def is_empty(self) -> bool:
        """True when no pixel lies in this tail."""
        return self.count == 0

    @property
    def mean(self) -> float | None:
        """Mean value of the tail's pixels, the value standing in for them (None if empty)."""
        return None if self.count == 0 else self.sum / self.count


EMPTY_TAIL = HistogramTail()


@dataclass(frozen=True, eq=False)
class PlaneHistogram:
    """Histogram of one 2D plane plus the facts needed to interpret it.

    Attributes
    ----------
    dtype
        Pixel type of the source plane.
    bin_count
        Number of bins (the same for every plane in an experiment).
    binning_min, binning_max
        Range the bins cover: the plane's finite minimum and maximum, or its
        robust limits (contract §1.4); None when the plane has no finite pixels.
    counts
        Number of pixels in each bin (int64, length ``bin_count``).
    total_pixel_count
        All pixels in the plane.
    valid_pixel_count
        Finite pixels: those counted in ``counts`` and in the two tails.
    pixel_min, pixel_max
        Exact finite minimum and maximum of the plane (None if it has no finite
        pixels). Default to ``binning_min`` and ``binning_max``.
    underflow, overflow
        The pixels below ``binning_min`` and above ``binning_max``; empty unless
        the robust range was used.
    """

    dtype: np.dtype
    bin_count: int
    binning_min: int | float | None
    binning_max: int | float | None
    counts: np.ndarray
    total_pixel_count: int
    valid_pixel_count: int
    pixel_min: int | float | None = None
    pixel_max: int | float | None = None
    underflow: HistogramTail = EMPTY_TAIL
    overflow: HistogramTail = EMPTY_TAIL

    def __post_init__(self) -> None:
        if self.pixel_min is None:
            object.__setattr__(self, "pixel_min", self.binning_min)
        if self.pixel_max is None:
            object.__setattr__(self, "pixel_max", self.binning_max)

    @property
    def is_empty(self) -> bool:
        """True when the plane has no finite pixels."""
        return self.valid_pixel_count == 0

    @property
    def non_finite_pixel_count(self) -> int:
        """Pixels ignored because they are NaN or infinite."""
        return self.total_pixel_count - self.valid_pixel_count

    @property
    def has_tails(self) -> bool:
        """True when some pixels lie outside the bins (robust range only)."""
        return not (self.underflow.is_empty and self.overflow.is_empty)

    @cached_property
    def representatives(self) -> np.ndarray:
        """Representative value of every bin (float64, length ``bin_count``)."""
        if self.is_empty:
            return np.full(self.bin_count, np.nan)
        if is_integer(self.dtype):
            first, last = self._integer_bin_limits
            return np.array(
                [self.binning_min + (a + b) / 2 for a, b in zip(first, last, strict=True)],
                dtype=np.float64,
            )
        lo, hi = float(self.binning_min), float(self.binning_max)
        half_range = hi * 0.5 - lo * 0.5
        return lo + half_range * ((2 * np.arange(self.bin_count) + 1) / self.bin_count)

    @property
    def max_representative_error(self) -> float:
        """Largest distance between any pixel and the value standing in for it.

        Integer planes: (most values in one bin - 1) / 2, so 0 for exact histograms.
        Float planes: half the bin width. Pixels of a tail are represented by
        the tail's mean, so they add the larger distance from that mean to the
        tail's ends: the plane's extremum and the binning limit. A histogram
        with tails is therefore never exact.
        """
        if self.is_empty:
            return 0.0
        if is_integer(self.dtype):
            first, last = self._integer_bin_limits
            widest = max(b - a + 1 for a, b in zip(first, last, strict=True))
            bins = (widest - 1) / 2
        else:
            lo, hi = float(self.binning_min), float(self.binning_max)
            bins = (hi * 0.5 - lo * 0.5) / self.bin_count
        if not self.has_tails:
            return bins
        errors = [bins]
        if not self.underflow.is_empty:
            mean = self.underflow.mean
            errors.append(max(mean - self.pixel_min, self.binning_min - mean))
        if not self.overflow.is_empty:
            mean = self.overflow.mean
            errors.append(max(mean - self.binning_max, self.pixel_max - mean))
        return float(max(errors))

    @property
    def is_exact(self) -> bool:
        """True when every bin holds a single value and no pixel lies outside the bins."""
        return is_integer(self.dtype) and self.max_representative_error == 0

    def integer_bin_limits(self) -> list[tuple[int, int]]:
        """Return ``(first_value, last_value)`` for every bin of an integer plane.

        Bins that hold no integer (possible when the range is narrower than the
        bin count) have ``first_value > last_value``.
        """
        if not is_integer(self.dtype):
            raise TypeError("Bin limits as whole numbers exist only for integer planes.")
        if self.is_empty:
            return []
        first, last = self._integer_bin_limits
        return [
            (self.binning_min + a, self.binning_min + b) for a, b in zip(first, last, strict=True)
        ]

    @cached_property
    def _integer_bin_limits(self) -> tuple[list[int], list[int]]:
        """Bin limits as offsets from ``binning_min`` (exact Python integers)."""
        starts = _integer_bin_starts(self.binning_max + 1 - self.binning_min, self.bin_count)
        return starts[:-1], [s - 1 for s in starts[1:]]


def compute_plane_histogram(
    plane: np.ndarray,
    bin_count: int | None = None,
    *,
    settings: HistogramSettings | None = None,
) -> PlaneHistogram:
    """Compute the histogram of one plane.

    Parameters
    ----------
    plane
        Pixel values of a single 2D plane (any shape is accepted and flattened).
    bin_count
        Number of bins. Defaults to one bin per value for 8- and 16-bit
        integers, 65,536 for wider integers and 10,000 for floats.
    settings
        The histogram range (contract §1.4); None bins from the plane's
        minimum to its maximum.

    Returns
    -------
    PlaneHistogram

    Raises
    ------
    UnsupportedDtypeError
        If the pixel type is not supported.
    ValueError
        If the bin count is not allowed for the pixel type.
    """
    values = np.asarray(plane)
    return compute_block_histogram(lambda: (values,), values.dtype, bin_count, settings=settings)


def compute_block_histogram(
    blocks: Callable[[], Iterable[np.ndarray]],
    dtype: np.dtype,
    bin_count: int | None = None,
    *,
    settings: HistogramSettings | None = None,
) -> PlaneHistogram:
    """Compute the histogram of a plane given in blocks, without joining them.

    The result is identical to :func:`compute_plane_histogram` of the whole
    plane: a first pass over the blocks finds the exact extrema, a second pass
    counts every pixel into the bins with the same formulas. Integer planes of 8
    and 16 bits need only one pass: they are counted per value first, and the
    counts are put into bins once the extrema are known. The robust range needs
    extra passes for the quantiles of wider types (contract §1.4): one for
    float32 and 32-bit integers, three for 64-bit types.

    Parameters
    ----------
    blocks
        Called once per pass; returns the blocks of the plane (arrays of any
        shape, together holding every pixel of the plane exactly once).
    dtype
        Pixel type of the plane.
    bin_count
        Number of bins; defaults as in :func:`compute_plane_histogram`.
    settings
        The histogram range; None bins from the plane's minimum to its maximum.

    Returns
    -------
    PlaneHistogram

    Raises
    ------
    UnsupportedDtypeError
        If the pixel type is not supported.
    ValueError
        If the bin count is not allowed for the pixel type.
    """
    return _measure(blocks, dtype, bin_count, settings, check=False)[0]


def histogram_with_range_check(
    blocks: Callable[[], Iterable[np.ndarray]],
    dtype: np.dtype,
    bin_count: int | None = None,
    *,
    settings: HistogramSettings | None = None,
) -> tuple[PlaneHistogram, RangeCheck | None]:
    """Compute a plane's histogram and, if it is not exact, check its range (N57, N59).

    The histogram is the one :func:`compute_block_histogram` gives. The check
    shares its passes over the blocks: with the min-max range, a float32 or
    32-bit integer plane is read no more often than for the histogram alone,
    and a 64-bit plane twice more.

    Parameters
    ----------
    blocks, dtype, bin_count, settings
        As for :func:`compute_block_histogram`; ``settings`` also holds the
        robust quantiles and the warning limit of the check.

    Returns
    -------
    histogram : PlaneHistogram
    check : RangeCheck or None
        None for an exact histogram (8- and 16-bit planes with one bin per
        value, integer planes with no more values than bins, constant planes)
        and for a plane without finite pixels.
    """
    return _measure(blocks, dtype, bin_count, settings, check=True)


def estimate_histogram_memory(
    pixel_count: int,
    dtype: np.dtype,
    bin_count: int | None = None,
    *,
    settings: HistogramSettings | None = None,
    range_check: bool = False,
) -> int:
    """Estimate the temporary memory :func:`compute_plane_histogram` needs, in bytes.

    Covers the arrays created while binning, not the plane itself. It is an
    upper estimate: integer planes allow for the widest per-value count table
    (used when a plane holds up to 2**24 distinct values), whatever the actual
    range of the plane turns out to be. The robust range and the range check
    (``range_check=True``, :func:`histogram_with_range_check`) add the arrays of
    the quantile search for types wider than 16 bits.

    Parameters
    ----------
    pixel_count
        Number of pixels in the plane.
    dtype
        Pixel type of the plane.
    bin_count
        Number of bins; defaults as in :func:`compute_plane_histogram`.
    settings, range_check
        The histogram range and whether the range is checked.
    """
    dt = check_supported(dtype)
    n = default_bin_count(dt) if bin_count is None else bin_count
    # Bin counts, limits and representatives: a few float64/int64 arrays per bin.
    per_bin = 4 * 8 * n
    if is_integer(dt):
        # Either up to four 8-byte arrays per pixel (ordered copy, offsets, bin index), or
        # two per pixel while the per-value count table and its lookups (four 8-byte
        # arrays per value) exist.
        table = min(2 ** (dt.itemsize * 8), _PER_VALUE_COUNT_LIMIT)
        estimate = max(4 * 8 * pixel_count, 2 * 8 * pixel_count + 4 * 8 * table) + per_bin
    else:
        # Finite mask, float64 copy, two scaled temporaries and the bin index.
        estimate = (1 + 4 * 8) * pixel_count + per_bin
    robust = settings is not None and settings.robust
    if (robust or range_check) and not (is_integer(dt) and dt.itemsize <= 2):
        # Quantile search: finite copy, keys, shifted keys, digits, a mask and the
        # selected digits per pixel; two digit tables per followed quantile.
        estimate += (5 * 8 + 1) * pixel_count + 2 * _MAX_TARGETS * 8 * _DIGIT_VALUES
        if robust:
            # Tail masks and the copy of the pixels inside the robust limits.
            estimate += (2 + 8) * pixel_count
    return estimate


class _Order(NamedTuple):
    """An order statistic: its value, and how many pixels are below and equal to it."""

    value: int | float
    below: int
    equal: int


def _measure(
    blocks: Callable[[], Iterable[np.ndarray]],
    dtype: np.dtype,
    bin_count: int | None,
    settings: HistogramSettings | None,
    check: bool,
) -> tuple[PlaneHistogram, RangeCheck | None]:
    dtype = check_supported(np.dtype(dtype))
    n = validate_bin_count(dtype, default_bin_count(dtype) if bin_count is None else bin_count)
    settings = HistogramSettings() if settings is None else settings
    if is_integer(dtype) and dtype.itemsize <= 2:
        return _small_integer_histogram(blocks(), dtype, n, settings, check)
    return _wide_histogram(blocks, dtype, n, settings, check)


def _integer_bin_starts(width: int, bin_count: int) -> list[int]:
    """Offsets at which bins ``0 .. bin_count`` start (the last entry is ``width``)."""
    if width <= bin_count:
        # One value per bin from the start; the remaining bins stay empty.
        return [min(i, width) for i in range(bin_count + 1)]
    return [-((-i * width) // bin_count) for i in range(bin_count + 1)]


def _as_ordered_uint64(values: np.ndarray) -> np.ndarray:
    """Map integers to uint64 while keeping their order, without overflow."""
    if values.dtype.kind == "u":
        return values.astype(np.uint64)
    return values.astype(np.int64).view(np.uint64) ^ np.uint64(1 << 63)


def _counts_from_per_value(per_value: np.ndarray, starts: list[int], n: int) -> np.ndarray:
    """Sum per-value counts (offsets ``0 .. width - 1``) into bins starting at ``starts``."""
    bin_of_value = np.searchsorted(np.array(starts[:-1]), np.arange(per_value.size), side="right")
    counts = np.bincount(bin_of_value - 1, weights=per_value, minlength=n)
    return np.rint(counts).astype(np.int64)


def _small_integer_histogram(
    blocks: Iterable[np.ndarray],
    dtype: np.dtype,
    n: int,
    settings: HistogramSettings,
    check: bool,
) -> tuple[PlaneHistogram, RangeCheck | None]:
    """One pass for 8- and 16-bit integers: count every possible value, then bin."""
    bits = dtype.itemsize * 8
    table = np.zeros(2**bits, dtype=np.int64)
    total = 0
    for block in blocks:
        values = np.asarray(block).ravel()
        if values.size == 0:
            continue
        total += int(values.size)
        unsigned = values.view(f"u{dtype.itemsize}")
        if dtype.kind == "i":
            # Flipping the sign bit keeps the order: the smallest value becomes 0.
            unsigned = unsigned ^ np.array(1 << (bits - 1), dtype=unsigned.dtype)
        table += np.bincount(unsigned, minlength=table.size)
    if total == 0:
        return PlaneHistogram(dtype, n, None, None, np.zeros(n, np.int64), 0, 0), None
    present = np.flatnonzero(table)
    first, last = int(present[0]), int(present[-1])
    smallest = int(np.iinfo(dtype).min)
    exact = last + 1 - first <= n
    if exact or not (settings.robust or check):
        starts = _integer_bin_starts(last + 1 - first, n)
        counts = _counts_from_per_value(table[first : last + 1], starts, n)
        histogram = PlaneHistogram(
            dtype, n, smallest + first, smallest + last, counts, total, total
        )
        return histogram, None
    # The per-value counts give every order statistic exactly, without another pass.
    cumulative = np.cumsum(table)

    def order(q: float) -> _Order:
        index = int(np.searchsorted(cumulative, _inverse_cdf_rank(q, total), side="left"))
        below = int(cumulative[index - 1]) if index else 0
        return _Order(smallest + index, below, int(table[index]))

    quantiles = [*settings.robust_quantiles, *(CENTRAL_PERCENTILES if check else ())]
    orders = [order(q) for q in quantiles]
    if settings.robust:
        low, high = orders[0].value - smallest, orders[1].value - smallest
        starts = _integer_bin_starts(high + 1 - low, n)
        counts = _counts_from_per_value(table[low : high + 1], starts, n)
        histogram = PlaneHistogram(
            dtype,
            n,
            smallest + low,
            smallest + high,
            counts,
            total,
            total,
            pixel_min=smallest + first,
            pixel_max=smallest + last,
            underflow=_table_tail(table[first:low], smallest + first),
            overflow=_table_tail(table[high + 1 : last + 1], smallest + high + 1),
        )
    else:
        starts = _integer_bin_starts(last + 1 - first, n)
        counts = _counts_from_per_value(table[first : last + 1], starts, n)
        histogram = PlaneHistogram(
            dtype, n, smallest + first, smallest + last, counts, total, total
        )
    return histogram, _range_check(histogram, orders, settings) if check else None


def _wide_histogram(
    blocks: Callable[[], Iterable[np.ndarray]],
    dtype: np.dtype,
    n: int,
    settings: HistogramSettings,
    check: bool,
) -> tuple[PlaneHistogram, RangeCheck | None]:
    """Floats and wider integers: exact extrema first, then the bin counts.

    Quantiles, when needed, are found by radix selection: the first pass also
    counts the top 16 bits of every finite pixel, and each further pass the
    next 16 bits of the pixels that can still hold a wanted order statistic.
    """
    integer = is_integer(dtype)
    number = int if integer else float
    selection = _RadixSelection(dtype.itemsize * 8) if settings.robust or check else None
    total = valid = 0
    lo = hi = None
    for block in blocks():
        values = np.asarray(block).ravel()
        total += int(values.size)
        finite = values if integer else values[np.isfinite(values)]
        if finite.size == 0:
            continue
        valid += int(finite.size)
        low, high = number(finite.min()), number(finite.max())
        lo = low if lo is None else min(lo, low)
        hi = high if hi is None else max(hi, high)
        if selection is not None:
            selection.feed(_ordered_keys(finite, dtype))
    if valid == 0:
        return PlaneHistogram(dtype, n, None, None, np.zeros(n, np.int64), total, 0), None
    exact = hi + 1 - lo <= n if integer else lo == hi
    if exact or selection is None:
        return _bin_pass(blocks, dtype, n, lo, hi, total, valid), None
    quantiles = [*settings.robust_quantiles, *(CENTRAL_PERCENTILES if check else ())]
    selection.start([_inverse_cdf_rank(q, valid) for q in quantiles])
    # With the min-max range the bins do not wait for the quantiles, so the
    # last round of the selection shares the binning pass.
    shared_round = 0 if settings.robust else 1
    while selection.rounds_left > shared_round:
        for block in blocks():
            values = np.asarray(block).ravel()
            finite = values if integer else values[np.isfinite(values)]
            if finite.size:
                selection.feed(_ordered_keys(finite, dtype))
        selection.end_round()
    if settings.robust:
        orders = selection.orders(dtype)
        histogram = _bin_pass(
            blocks, dtype, n, orders[0].value, orders[1].value, total, valid, extrema=(lo, hi)
        )
    else:
        histogram = _bin_pass(blocks, dtype, n, lo, hi, total, valid, selection=selection)
        selection.end_round()
        orders = selection.orders(dtype)
    return histogram, _range_check(histogram, orders, settings) if check else None


def _bin_pass(
    blocks: Callable[[], Iterable[np.ndarray]],
    dtype: np.dtype,
    n: int,
    lo: int | float,
    hi: int | float,
    total: int,
    valid: int,
    *,
    extrema: tuple[int | float, int | float] | None = None,
    selection: _RadixSelection | None = None,
) -> PlaneHistogram:
    """Count the pixels into bins over ``[lo, hi]``.

    ``extrema`` gives the plane's exact minimum and maximum when the bins span
    a robust range: the pixels outside ``[lo, hi]`` then go to the tails.
    ``selection`` is fed every block on the way.
    """
    under, over = _TailSums(), _TailSums()
    if is_integer(dtype):
        counts = _integer_bins(blocks, dtype, n, lo, hi, extrema, selection, under, over)
    else:
        counts = _float_bins(blocks, dtype, n, lo, hi, valid, extrema, selection, under, over)
    if extrema is None:
        return PlaneHistogram(dtype, n, lo, hi, counts, total, valid)
    return PlaneHistogram(
        dtype,
        n,
        lo,
        hi,
        counts,
        total,
        valid,
        pixel_min=extrema[0],
        pixel_max=extrema[1],
        underflow=under.tail(),
        overflow=over.tail(),
    )


def _integer_bins(
    blocks: Callable[[], Iterable[np.ndarray]],
    dtype: np.dtype,
    n: int,
    lo: int,
    hi: int,
    extrema: tuple[int, int] | None,
    selection: _RadixSelection | None,
    under: _TailSums,
    over: _TailSums,
) -> np.ndarray:
    width = hi + 1 - lo
    starts = _integer_bin_starts(width, n)
    origin = _as_ordered_uint64(np.array([lo], dtype=dtype))[0]
    per_value = np.zeros(width, dtype=np.int64) if width <= _PER_VALUE_COUNT_LIMIT else None
    bin_starts = np.array(starts[:-1], dtype=np.uint64)
    counts = np.zeros(n, dtype=np.int64)
    for block in blocks():
        values = np.asarray(block).ravel()
        if values.size == 0:
            continue
        if selection is not None:
            selection.feed(_ordered_keys(values, dtype))
        if extrema is not None:
            values = _split_tails(values, lo, hi, under, over)
            if values.size == 0:
                continue
        offsets = _as_ordered_uint64(values) - origin
        if per_value is not None:
            # Count over the block's own range only, so that many blocks stay cheap.
            first, last = int(offsets.min()), int(offsets.max())
            per_value[first : last + 1] += np.bincount((offsets - np.uint64(first)).astype(np.intp))
        else:
            index = np.searchsorted(bin_starts, offsets, side="right") - 1
            counts += np.bincount(index, minlength=n)
    if per_value is not None:
        counts = _counts_from_per_value(per_value, starts, n)
    return counts


def _float_bins(
    blocks: Callable[[], Iterable[np.ndarray]],
    dtype: np.dtype,
    n: int,
    lo: float,
    hi: float,
    valid: int,
    extrema: tuple[float, float] | None,
    selection: _RadixSelection | None,
    under: _TailSums,
    over: _TailSums,
) -> np.ndarray:
    counts = np.zeros(n, dtype=np.int64)
    if lo == hi and extrema is None:
        counts[0] = valid
        return counts
    # Halving keeps hi - lo finite even for ranges near the float64 limits.
    half_range = hi * 0.5 - lo * 0.5
    for block in blocks():
        values = np.asarray(block).ravel()
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        if selection is not None:
            selection.feed(_ordered_keys(finite, dtype))
        finite = finite.astype(np.float64, copy=False)
        if extrema is not None:
            finite = _split_tails(finite, lo, hi, under, over)
            if half_range == 0:
                # Robust limits that coincide: every pixel inside them has that value.
                counts[0] += finite.size
                continue
        index = np.floor((finite * 0.5 - lo * 0.5) * (n / half_range)).astype(np.int64)
        np.clip(index, 0, n - 1, out=index)
        counts += np.bincount(index, minlength=n)
    return counts


class _TailSums:
    """Running count, sum and sum of squares of the pixels of one tail."""

    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sum_of_squares = 0.0

    def add(self, values: np.ndarray) -> None:
        values = values.astype(np.float64)
        self.count += int(values.size)
        self.sum += float(np.sum(values))
        self.sum_of_squares += float(np.sum(values * values))

    def tail(self) -> HistogramTail:
        return (
            HistogramTail(self.count, self.sum, self.sum_of_squares) if self.count else EMPTY_TAIL
        )


def _split_tails(
    values: np.ndarray, lo: int | float, hi: int | float, under: _TailSums, over: _TailSums
) -> np.ndarray:
    """Add the values outside ``[lo, hi]`` to the tails; return the values inside."""
    below = values < lo
    above = values > hi
    outside = below | above
    if not outside.any():
        return values
    under.add(values[below])
    over.add(values[above])
    return values[~outside]


def _table_tail(per_value: np.ndarray, first_value: int) -> HistogramTail:
    """Sum the tail of per-value counts whose first entry counts ``first_value``."""
    count = int(per_value.sum())
    if count == 0:
        return EMPTY_TAIL
    values = first_value + np.arange(per_value.size, dtype=np.float64)
    weights = per_value.astype(np.float64)
    return HistogramTail(
        count, float(np.dot(values, weights)), float(np.dot(values * values, weights))
    )


def _inverse_cdf_rank(q: float, total: int) -> int:
    """One-based rank of the inverse-CDF ``q`` percentile among ``total`` values (§1.2.3)."""
    return max(1, int(np.ceil(q / 100 * total)))


def _ordered_keys(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Unsigned integers of the width of ``dtype`` that sort as the (finite) values do."""
    values = values.astype(dtype, copy=False)
    unsigned = np.dtype(f"u{values.dtype.itemsize}")
    raw = values.view(unsigned)
    top = unsigned.type(1 << (values.dtype.itemsize * 8 - 1))
    if values.dtype.kind == "u":
        return raw
    if values.dtype.kind == "i":
        return raw ^ top
    # Floats: setting the sign bit of positive numbers and flipping every bit
    # of negative ones gives keys in the order of the values.
    return np.where((raw & top) != 0, ~raw, raw | top)


def _key_value(key: int, dtype: np.dtype) -> int | float:
    """Return the pixel value of ``dtype`` whose ordered key is ``key``."""
    bits = dtype.itemsize * 8
    top = 1 << (bits - 1)
    if dtype.kind == "u":
        raw = key
    elif dtype.kind == "i":
        raw = key ^ top
    else:
        raw = key ^ top if key & top else ~key & ((1 << bits) - 1)
    return np.array([raw], dtype=f"u{dtype.itemsize}").view(dtype)[0].item()


class _RadixSelection:
    """Exact order statistics of keys seen in several passes, 16 bits per pass.

    Every pass feeds the same keys, block by block. The first pass counts the
    keys' top 16 bits; :meth:`start` then fixes the wanted ranks and the top
    digit of each wanted key. Each later pass counts the next 16 bits of the
    keys that share every digit found so far with a wanted key, until every
    bit is known. A key of ``b`` bits needs ``b / 16`` passes.
    """

    def __init__(self, key_bits: int) -> None:
        self._key_bits = key_bits
        self._resolved = 0
        self._tables = {0: np.zeros(_DIGIT_VALUES, dtype=np.int64)}
        # Per wanted key: [digits found, rank among the keys sharing them,
        # keys below, keys equal to it once complete].
        self._targets: list[list[int]] = []

    @property
    def rounds_left(self) -> int:
        """Passes still needed."""
        return (self._key_bits - self._resolved) // _DIGIT_BITS

    def feed(self, keys: np.ndarray) -> None:
        kind = keys.dtype.type
        shift = self._key_bits - self._resolved - _DIGIT_BITS
        digits = ((keys >> kind(shift)) & kind(_DIGIT_VALUES - 1)).astype(np.intp)
        if self._resolved == 0:
            self._tables[0] += np.bincount(digits, minlength=_DIGIT_VALUES)
            return
        high = keys >> kind(shift + _DIGIT_BITS)
        for prefix, table in self._tables.items():
            table += np.bincount(digits[high == prefix], minlength=_DIGIT_VALUES)

    def start(self, ranks: list[int]) -> None:
        """Fix the one-based ranks wanted, after the first pass, and resolve its digit."""
        self._targets = [[0, rank, 0, 0] for rank in ranks]
        self.end_round()

    def end_round(self) -> None:
        """Resolve the digit of every wanted key counted in this pass."""
        for target in self._targets:
            prefix, rank, below, _ = target
            table = self._tables[prefix]
            cumulative = np.cumsum(table)
            digit = int(np.searchsorted(cumulative, rank, side="left"))
            before = int(cumulative[digit - 1]) if digit else 0
            target[:] = [(prefix << _DIGIT_BITS) | digit, rank - before, below + before]
            target.append(int(table[digit]))
        self._resolved += _DIGIT_BITS
        self._tables = {
            target[0]: np.zeros(_DIGIT_VALUES, dtype=np.int64)
            for target in self._targets
            if self.rounds_left
        }

    def orders(self, dtype: np.dtype) -> list[_Order]:
        """Return the wanted order statistics as pixel values, in the order of the ranks."""
        return [
            _Order(_key_value(key, dtype), below, equal) for key, _, below, equal in self._targets
        ]


def _range_check(
    histogram: PlaneHistogram, orders: list[_Order], settings: HistogramSettings
) -> RangeCheck:
    """Check the range of a histogram that is not exact, from its four order statistics."""
    robust_low, robust_high, central_low, central_high = orders
    limits = (robust_low.value, robust_high.value)
    n, dtype = histogram.bin_count, histogram.dtype

    def bins(lo: int | float, hi: int | float) -> int:
        first = _bin_index(central_low.value, dtype, n, lo, hi)
        return _bin_index(central_high.value, dtype, n, lo, hi) - first + 1

    above = histogram.valid_pixel_count - robust_high.below - robust_high.equal
    return RangeCheck(
        histogram_range=settings.histogram_range,
        central_percentiles=CENTRAL_PERCENTILES,
        central_bins=bins(histogram.binning_min, histogram.binning_max),
        robust_quantiles=settings.robust_quantiles,
        robust_limits=limits,
        outside_robust_count=robust_low.below + above,
        robust_central_bins=bins(*limits),
        min_central_bins=settings.min_central_bins,
    )


def _bin_index(
    value: int | float, dtype: np.dtype, n: int, lo: int | float, hi: int | float
) -> int:
    """Return the bin of ``value`` with bins over ``[lo, hi]``, values outside clipped."""
    if is_integer(dtype):
        starts = _integer_bin_starts(hi + 1 - lo, n)
        index = bisect.bisect_right(starts, value - lo, hi=n) - 1
    elif lo == hi:
        index = 0
    else:
        # The arithmetic of _float_bins, in Python floats (IEEE float64 as NumPy's).
        half_range = hi * 0.5 - lo * 0.5
        index = math.floor((value * 0.5 - lo * 0.5) * (n / half_range))
    return min(max(index, 0), n - 1)
