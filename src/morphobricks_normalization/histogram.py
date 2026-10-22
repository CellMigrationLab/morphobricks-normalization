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

Every pixel lies within half a bin width of its bin's representative value,
which bounds the error of any percentile computed from these histograms (see
:attr:`PlaneHistogram.max_representative_error`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .dtypes import check_supported, default_bin_count, is_integer, validate_bin_count

# Above this many distinct values, integer planes are binned with a search over
# bin starts instead of a full per-value count.
_PER_VALUE_COUNT_LIMIT = 2**24


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
        Range the bins cover: the plane's finite minimum and maximum, or None
        when the plane has no finite pixels.
    counts
        Number of pixels in each bin (int64, length ``bin_count``).
    total_pixel_count
        All pixels in the plane.
    valid_pixel_count
        Finite pixels, i.e. the pixels counted in ``counts``.
    """

    dtype: np.dtype
    bin_count: int
    binning_min: int | float | None
    binning_max: int | float | None
    counts: np.ndarray
    total_pixel_count: int
    valid_pixel_count: int

    @property
    def pixel_min(self) -> int | float | None:
        """Exact finite minimum of the plane (None if it has no finite pixels)."""
        return self.binning_min

    @property
    def pixel_max(self) -> int | float | None:
        """Exact finite maximum of the plane (None if it has no finite pixels)."""
        return self.binning_max

    @property
    def is_empty(self) -> bool:
        """True when the plane has no finite pixels."""
        return self.valid_pixel_count == 0

    @property
    def non_finite_pixel_count(self) -> int:
        """Pixels ignored because they are NaN or infinite."""
        return self.total_pixel_count - self.valid_pixel_count

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
        """Largest distance between any pixel and its bin's representative value.

        Integer planes: (most values in one bin - 1) / 2, so 0 for exact histograms.
        Float planes: half the bin width.
        """
        if self.is_empty:
            return 0.0
        if is_integer(self.dtype):
            first, last = self._integer_bin_limits
            widest = max(b - a + 1 for a, b in zip(first, last, strict=True))
            return (widest - 1) / 2
        lo, hi = float(self.binning_min), float(self.binning_max)
        return (hi * 0.5 - lo * 0.5) / self.bin_count

    @property
    def is_exact(self) -> bool:
        """True when every bin holds a single value (integer planes only)."""
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


def compute_plane_histogram(plane: np.ndarray, bin_count: int | None = None) -> PlaneHistogram:
    """Compute the histogram of one plane.

    Parameters
    ----------
    plane
        Pixel values of a single 2D plane (any shape is accepted and flattened).
    bin_count
        Number of bins. Defaults to one bin per value for 8- and 16-bit
        integers, 65,536 for wider integers and 10,000 for floats.

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
    return compute_block_histogram(lambda: (values,), values.dtype, bin_count)


def compute_block_histogram(
    blocks: Callable[[], Iterable[np.ndarray]],
    dtype: np.dtype,
    bin_count: int | None = None,
) -> PlaneHistogram:
    """Compute the histogram of a plane given in blocks, without joining them.

    The result is identical to :func:`compute_plane_histogram` of the whole
    plane: a first pass over the blocks finds the exact extrema, a second pass
    counts every pixel into the bins with the same formulas. Integer planes of 8
    and 16 bits need only one pass: they are counted per value first, and the
    counts are put into bins once the extrema are known.

    Parameters
    ----------
    blocks
        Called once per pass; returns the blocks of the plane (arrays of any
        shape, together holding every pixel of the plane exactly once).
    dtype
        Pixel type of the plane.
    bin_count
        Number of bins; defaults as in :func:`compute_plane_histogram`.

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
    dtype = check_supported(np.dtype(dtype))
    n = validate_bin_count(dtype, default_bin_count(dtype) if bin_count is None else bin_count)
    if is_integer(dtype) and dtype.itemsize <= 2:
        return _small_integer_histogram(blocks(), dtype, n)
    if is_integer(dtype):
        return _integer_histogram(blocks, dtype, n)
    return _float_histogram(blocks, dtype, n)


def estimate_histogram_memory(
    pixel_count: int, dtype: np.dtype, bin_count: int | None = None
) -> int:
    """Estimate the temporary memory :func:`compute_plane_histogram` needs, in bytes.

    Covers the arrays created while binning, not the plane itself. It is an
    upper estimate: integer planes allow for the widest per-value count table
    (used when a plane holds up to 2**24 distinct values), whatever the actual
    range of the plane turns out to be.

    Parameters
    ----------
    pixel_count
        Number of pixels in the plane.
    dtype
        Pixel type of the plane.
    bin_count
        Number of bins; defaults as in :func:`compute_plane_histogram`.
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
        return max(4 * 8 * pixel_count, 2 * 8 * pixel_count + 4 * 8 * table) + per_bin
    # Finite mask, float64 copy, two scaled temporaries and the bin index.
    return (1 + 4 * 8) * pixel_count + per_bin


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
    blocks: Iterable[np.ndarray], dtype: np.dtype, n: int
) -> PlaneHistogram:
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
        return PlaneHistogram(dtype, n, None, None, np.zeros(n, np.int64), 0, 0)
    present = np.flatnonzero(table)
    first, last = int(present[0]), int(present[-1])
    smallest = int(np.iinfo(dtype).min)
    starts = _integer_bin_starts(last + 1 - first, n)
    counts = _counts_from_per_value(table[first : last + 1], starts, n)
    return PlaneHistogram(dtype, n, smallest + first, smallest + last, counts, total, total)


def _integer_histogram(
    blocks: Callable[[], Iterable[np.ndarray]], dtype: np.dtype, n: int
) -> PlaneHistogram:
    """Two passes for wider integers: exact extrema first, then the bin counts."""
    total, lo, hi = 0, None, None
    for block in blocks():
        values = np.asarray(block).ravel()
        if values.size == 0:
            continue
        total += int(values.size)
        low, high = int(values.min()), int(values.max())
        lo = low if lo is None else min(lo, low)
        hi = high if hi is None else max(hi, high)
    if total == 0:
        return PlaneHistogram(dtype, n, None, None, np.zeros(n, np.int64), 0, 0)
    width = hi + 1 - lo
    starts = _integer_bin_starts(width, n)
    origin = _as_ordered_uint64(np.array([lo], dtype=dtype))[0]
    if width <= _PER_VALUE_COUNT_LIMIT:
        per_value = np.zeros(width, dtype=np.int64)
        for block in blocks():
            values = np.asarray(block).ravel()
            if values.size == 0:
                continue
            offsets = _as_ordered_uint64(values) - origin
            # Count over the block's own range only, so that many blocks stay cheap.
            first, last = int(offsets.min()), int(offsets.max())
            per_value[first : last + 1] += np.bincount((offsets - np.uint64(first)).astype(np.intp))
        counts = _counts_from_per_value(per_value, starts, n)
    else:
        bin_starts = np.array(starts[:-1], dtype=np.uint64)
        counts = np.zeros(n, dtype=np.int64)
        for block in blocks():
            values = np.asarray(block).ravel()
            if values.size == 0:
                continue
            offsets = _as_ordered_uint64(values) - origin
            index = np.searchsorted(bin_starts, offsets, side="right") - 1
            counts += np.bincount(index, minlength=n)
    return PlaneHistogram(dtype, n, lo, hi, counts, total, total)


def _float_histogram(
    blocks: Callable[[], Iterable[np.ndarray]], dtype: np.dtype, n: int
) -> PlaneHistogram:
    """Two passes: exact finite extrema first, then the bin counts."""
    total = valid = 0
    lo = hi = None
    for block in blocks():
        values = np.asarray(block).ravel()
        total += int(values.size)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        valid += int(finite.size)
        low, high = float(finite.min()), float(finite.max())
        lo = low if lo is None else min(lo, low)
        hi = high if hi is None else max(hi, high)
    counts = np.zeros(n, dtype=np.int64)
    if valid == 0:
        return PlaneHistogram(dtype, n, None, None, counts, total, 0)
    if lo == hi:
        counts[0] = valid
        return PlaneHistogram(dtype, n, lo, hi, counts, total, valid)
    # Halving keeps hi - lo finite even for ranges near the float64 limits.
    half_range = hi * 0.5 - lo * 0.5
    for block in blocks():
        values = np.asarray(block).ravel()
        finite = values[np.isfinite(values)].astype(np.float64, copy=False)
        if finite.size == 0:
            continue
        index = np.floor((finite * 0.5 - lo * 0.5) * (n / half_range)).astype(np.int64)
        np.clip(index, 0, n - 1, out=index)
        counts += np.bincount(index, minlength=n)
    return PlaneHistogram(dtype, n, lo, hi, counts, total, valid)
