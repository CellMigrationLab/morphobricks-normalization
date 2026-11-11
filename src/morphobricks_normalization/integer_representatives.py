"""Representative values of integer bins, and their distance from the bins' pixels (#28).

A bin of the integers ``first .. last`` stands for its pixels by its midpoint
``(first + last) / 2``, divided once as Python integers, so it is the float64 nearest the
midpoint. Below ``2**51`` every integer, every half-integer midpoint and every distance
between them is exact in float64, and the bound is ``(widest bin - 1) / 2``. From ``2**51``
on (64-bit pixel types) float64 cannot hold every value: ``2**53 + 1`` is stored as
``2**53``. There the bound is the exact largest distance from a bin's or tail's integers to
the float64 standing in for them, rounded up, so a histogram is exact only when every
representative equals the pixel values it stands for.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # the histogram module imports this one
    from .histogram import PlaneHistogram

FLOAT_EXACT_LIMIT = 2**51
"""Below this magnitude integer midpoints and their distances are exact float64 arithmetic."""


def integer_representatives(histogram: PlaneHistogram) -> np.ndarray:
    """Return the float64 nearest the midpoint of every bin of a non-empty integer plane."""
    return np.array(
        [(first + last) / 2 for first, last in histogram.integer_bin_limits()], dtype=np.float64
    )


def beyond_float_exact(histogram: PlaneHistogram) -> bool:
    """Whether a non-empty integer plane has a pixel of magnitude ``2**51`` or more."""
    return max(-int(histogram.pixel_min), int(histogram.pixel_max)) >= FLOAT_EXACT_LIMIT


def wide_integer_error(histogram: PlaneHistogram) -> float:
    """Return the exact bound of an integer plane beyond ``2**51``, rounded up to a float64.

    Every bin that holds an integer, and every non-empty tail (its mean against the plane's
    extremum and the binning limit), counts with the larger exact distance from the value
    standing in for it to its two ends.
    """
    ends = [
        (rep, first, last)
        for rep, (first, last) in zip(
            histogram.representatives.tolist(), histogram.integer_bin_limits(), strict=True
        )
        if first <= last  # a bin that holds no integer has no pixel to stand in for
    ]
    for tail, first, last in (
        (histogram.underflow, histogram.pixel_min, histogram.binning_min),
        (histogram.overflow, histogram.binning_max, histogram.pixel_max),
    ):
        if not tail.is_empty:
            ends.append((tail.mean, first, last))
    worst = max(
        max(abs(Fraction(rep) - int(first)), abs(Fraction(rep) - int(last)))
        for rep, first, last in ends
    )
    nearest = float(worst)
    return nearest if Fraction(nearest) >= worst else math.nextafter(nearest, math.inf)
