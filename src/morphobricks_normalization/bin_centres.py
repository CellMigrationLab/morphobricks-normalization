"""Positions and centres of equal-width float bins, safe at both ends of the float64 range."""

from __future__ import annotations

import math

import numpy as np

_SMALLEST_NORMAL = float(np.finfo(np.float64).tiny)
_SAFE_HALF_RANGE = 2.0**-1000  # n / half_range stays far below the float64 maximum


def float_bin_position(values: np.ndarray | float, lo: float, hi: float, n: int) -> np.ndarray:
    """Position of ``values`` in bin units, ``n`` bins over ``[lo, hi]`` (float64, not floored).

    Where the halved arithmetic ``(value / 2 - lo / 2) * (n / (hi / 2 - lo / 2))`` is exact
    and finite it is kept, so existing histograms do not change. It is not for a narrow
    range: halving rounds a subnormal endpoint (a half range of zero for distinct
    endpoints) and ``n / half_range`` overflows to infinity. There the position is a
    bounded fraction times ``n``, ``(value - lo) / (hi - lo) * n``, exact for subnormals;
    if ``hi - lo`` itself overflows the halves are used for the fraction. ``lo`` must be
    below ``hi``.
    """
    values = np.asarray(values, dtype=np.float64)
    half_range = hi * 0.5 - lo * 0.5  # halving keeps the range finite
    if _halving_is_exact(lo, hi, half_range, n):
        return (values * 0.5 - lo * 0.5) * (n / half_range)
    span = hi - lo
    if math.isfinite(span):
        return (values - lo) / span * n
    return (values * 0.5 - lo * 0.5) / half_range * n


def _halving_is_exact(lo: float, hi: float, half_range: float, n: int) -> bool:
    """Whether halving loses nothing and ``n / half_range`` is finite (far from both limits)."""
    exact = all(x == 0 or abs(x) >= 2 * _SMALLEST_NORMAL for x in (lo, hi))
    return exact and half_range >= n * _SAFE_HALF_RANGE


def float_bin_centres(lo: float, hi: float, bin_count: int) -> np.ndarray:
    """Centre of each of ``bin_count`` equal bins between ``lo`` and ``hi`` (float64).

    Valid outputs are the plain arithmetic ``lo + half_range * fraction``. That product
    reaches twice the half range (up to the full range) for the last bins, which
    overflows near the float64 limits although the centre is representable; only then are
    the affected bins stepped from the midpoint instead.

    A range that holds fewer representable floats than ``bin_count`` has bins no value can
    fall in (``[0, 5e-324]`` with 4 bins: ``0, 0, 5e-324, 5e-324``). Such unreachable bins may
    share a centre. For every bin that some float64 value falls in, the centre maps back to
    that bin, except that a bin holding exactly one float can have a centre that rounds to the
    neighbouring float, in the adjacent bin (its midpoint is not representable). The binning
    and the centres do not depend on this; ``bin_reachability`` states and tests it.
    """
    fraction = (2 * np.arange(bin_count) + 1) / bin_count
    span = hi - lo
    if math.isfinite(span):
        # Half the fraction keeps the product below the range; no halving of the range,
        # which would round to zero between distinct subnormal endpoints.
        return lo + span * (fraction * 0.5)
    half_range = hi * 0.5 - lo * 0.5  # halving keeps the range finite
    with np.errstate(over="ignore"):  # an overflowing step is redone just below
        centres = lo + half_range * fraction
    over = ~np.isfinite(centres)
    if over.any():
        midpoint = lo * 0.5 + hi * 0.5
        centres[over] = midpoint + half_range * (fraction[over] - 1)
    return centres
