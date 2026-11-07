"""Centres of equal-width float bins, without overflow near the float64 limits."""

from __future__ import annotations

import numpy as np


def float_bin_centres(lo: float, hi: float, bin_count: int) -> np.ndarray:
    """Centre of each of ``bin_count`` equal bins between ``lo`` and ``hi`` (float64).

    Valid outputs are the plain arithmetic ``lo + half_range * fraction``. That product
    reaches twice the half range (up to the full range) for the last bins, which
    overflows near the float64 limits although the centre is representable; only then are
    the affected bins stepped from the midpoint instead.
    """
    half_range = hi * 0.5 - lo * 0.5  # halving keeps the range finite
    fraction = (2 * np.arange(bin_count) + 1) / bin_count
    with np.errstate(over="ignore"):  # an overflowing step is redone just below
        centres = lo + half_range * fraction
    over = ~np.isfinite(centres)
    if over.any():
        midpoint = lo * 0.5 + hi * 0.5
        centres[over] = midpoint + half_range * (fraction[over] - 1)
    return centres
