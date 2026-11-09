"""Which float bins a representable value can fall in, and whether centres map back to them.

When ``[lo, hi]`` holds fewer float64 values than there are bins (for example
``[0, 5e-324]`` with 4 bins), some bins can hold no value at all. Those bins may share a
representative centre; the invariant is for the others. This module states it as code:

- :func:`float_bin_indices` is the binning of the histograms, as a function;
- :func:`reachable_float_bins` marks the bins some float64 value in the range falls in;
- :func:`centres_map_back` marks the bins whose centre falls in that same bin.

Every reachable bin must be one whose centre maps back to it. Nothing here changes how
pixels are binned or where centres are placed.
"""

from __future__ import annotations

import numpy as np

from .bin_centres import float_bin_centres, float_bin_position

_SIGN_BIT = np.uint64(1) << np.uint64(63)
_MAGNITUDE = ~_SIGN_BIT


def float_bin_indices(values: np.ndarray | float, lo: float, hi: float, n: int) -> np.ndarray:
    """Bin of each value, ``n`` bins over ``[lo, hi]``, as the histograms bin pixels.

    Parameters
    ----------
    values
        Finite values; values outside the range are clipped to the first or last bin.
    lo, hi
        The binning range, ``lo <= hi``. A constant range puts every value in bin 0.
    n
        Number of bins.

    Returns
    -------
    numpy.ndarray
        ``int64`` bin index of every value.
    """
    values = np.asarray(values, dtype=np.float64)
    if lo == hi:
        return np.zeros(values.shape, dtype=np.int64)
    index = np.floor(float_bin_position(values, lo, hi, n)).astype(np.int64)
    return np.clip(index, 0, n - 1)


def reachable_float_bins(lo: float, hi: float, n: int) -> np.ndarray:
    """Mark the bins that some float64 value in ``[lo, hi]`` falls in.

    The bin position is non-decreasing in the value, so the first value that reaches bin
    ``b`` is found by bisection over the ordered float64 values, and bin ``b`` is reachable
    when that value falls in ``b`` itself and not beyond it. This is exact and takes about
    64 vectorised steps for any range.

    Parameters
    ----------
    lo, hi
        The binning range, ``lo <= hi``.
    n
        Number of bins.

    Returns
    -------
    numpy.ndarray
        Boolean array of length ``n``, True for a bin that can hold a value.
    """
    wanted = np.arange(n, dtype=np.int64)
    left = np.full(n, _ordered_key(lo), dtype=np.int64)
    right = np.full(n, _ordered_key(hi), dtype=np.int64)
    while True:
        active = left < right
        if not active.any():
            break
        middle = (left >> 1) + (right >> 1) + (left & right & 1)  # floor mean, no overflow
        reaches = float_bin_indices(_from_ordered_key(middle), lo, hi, n) >= wanted
        right = np.where(active & reaches, middle, right)
        left = np.where(active & ~reaches, middle + 1, left)
    return float_bin_indices(_from_ordered_key(left), lo, hi, n) == wanted


def centres_map_back(lo: float, hi: float, n: int) -> np.ndarray:
    """Mark the bins whose centre falls in that same bin.

    Parameters
    ----------
    lo, hi
        The binning range, ``lo <= hi``.
    n
        Number of bins.

    Returns
    -------
    numpy.ndarray
        Boolean array of length ``n``.
    """
    centres = float_bin_centres(lo, hi, n)
    return float_bin_indices(centres, lo, hi, n) == np.arange(n)


def _ordered_key(value: float) -> np.int64:
    """Integer that grows with the float (``-0.0`` and ``0.0`` share key 0)."""
    bits = np.array(value, dtype=np.float64).view(np.uint64)
    magnitude = np.int64(bits & _MAGNITUDE)
    return -magnitude if bits & _SIGN_BIT else magnitude


def _from_ordered_key(keys: np.ndarray) -> np.ndarray:
    """Float64 values of integer keys made by :func:`_ordered_key`."""
    negative = keys < 0
    bits = np.abs(keys).astype(np.uint64) | np.where(negative, _SIGN_BIT, np.uint64(0))
    return bits.view(np.float64)
