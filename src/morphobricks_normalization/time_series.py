"""Brightness of a time series: its trend, and two opt-in corrections (contract §3.7).

Methods that assume constant brightness across time (optical flow, for example) read a
global brightness change (illumination warm-up, focus settling, bleaching, expression
changes) partly as motion. The functions here measure that change and, only when the caller
asks, correct it against one **reference** time point. Time is the first axis of every array;
the other axes (``ZYX`` for a 3-D stack, ``YX`` for a plane) are pooled.

- :func:`frame_means`: the mean of every time point, in float64.
- :func:`brightness_trend`: the least-squares slope of those means against the time index,
  relative to the reference mean (``0.05`` is a brightening by 5 % of the reference
  brightness per time point). It is a check: it changes nothing.
- :func:`equalise_brightness`: every time point multiplied so that its mean equals the
  reference's mean (Fiji Bleach Correction's "Simple Ratio" without a background). It is
  linear, so relative intensities within a time point are kept; a constant camera offset is
  scaled too, so the correction is approximate when the offset is a large part of the mean.
- :func:`match_histograms`: every time point's intensity distribution mapped onto the
  reference's (Fiji Bleach Correction's "Histogram Matching"): ``p' = F_ref^-1(F_t(p))`` with
  linear interpolation between the reference's distinct values. It is monotonic but
  non-linear: it changes relative intensities, which the Fiji author recommends for
  segmentation rather than for intensity measurements.

The input is never modified; the corrections return new float64 arrays and the reference
time point is returned unchanged (as float64). Non-finite pixels are refused: a mean or a
distribution with them has no defined value. Moved from the OpticalFlow3D Profiler
(``brightness.py``), whose optical-flow input used the central frame of each time window as
the reference (plan N72).
"""

from __future__ import annotations

import numpy as np


class TimeSeriesError(ValueError):
    """A time series that cannot be measured or corrected (see the message)."""


def _checked(stack: np.ndarray, reference: int | None = None) -> np.ndarray:
    """Return ``stack`` as an array after the checks every function shares."""
    stack = np.asarray(stack)
    if stack.ndim < 2 or stack.shape[0] < 1 or stack[0].size == 0:
        raise TimeSeriesError(
            f"A time series needs time as the first axis and at least one pixel per time point "
            f"(got shape {stack.shape}); pass an array of shape (T, ...), for example TZYX or TYX."
        )
    if not (np.issubdtype(stack.dtype, np.integer) or np.issubdtype(stack.dtype, np.floating)):
        raise TimeSeriesError(
            f"Pixel type {stack.dtype} is not an integer or a floating-point type; convert the "
            "time series to one before measuring its brightness."
        )
    if np.issubdtype(stack.dtype, np.floating) and not np.isfinite(stack).all():
        raise TimeSeriesError(
            "The time series holds non-finite pixels (NaN or infinity); their mean and "
            "distribution are undefined. Remove them before measuring or correcting brightness."
        )
    if reference is not None and not (
        isinstance(reference, int | np.integer)
        and not isinstance(reference, bool)
        and 0 <= reference < stack.shape[0]
    ):
        raise TimeSeriesError(
            f"The reference time point {reference!r} is not an index of the {stack.shape[0]} "
            f"time points; choose a whole number from 0 to {stack.shape[0] - 1}."
        )
    return stack


def frame_means(stack: np.ndarray) -> np.ndarray:
    """Return the mean intensity of every time point of ``stack`` (time first), in float64.

    Raises
    ------
    TimeSeriesError
        For an array without a time axis or pixels, a non-numeric type or non-finite pixels.
    """
    stack = _checked(stack)
    return stack.reshape(stack.shape[0], -1).mean(axis=1, dtype=np.float64)


def brightness_trend(means: np.ndarray, reference: int) -> float:
    """Return the relative brightness change per time point of a series of means.

    Parameters
    ----------
    means
        :func:`frame_means` of the series, in time order.
    reference
        The time point whose mean the slope is expressed against.

    Returns
    -------
    float
        Least-squares slope of the means against the time index, divided by the reference
        mean. NaN when the reference mean is not above 0 or there is a single time point
        (no slope).

    Raises
    ------
    TimeSeriesError
        For means that are not a 1-D finite series or an invalid reference.
    """
    means = np.asarray(means, dtype=np.float64)
    if means.ndim != 1 or not np.isfinite(means).all():
        raise TimeSeriesError("The means must be one finite value per time point, in time order.")
    _checked(means[:, None], reference)
    if len(means) < 2 or not means[reference] > 0:
        return float("nan")
    slope = np.polyfit(np.arange(len(means), dtype=np.float64), means, 1)[0]
    return float(slope / means[reference])


def equalise_brightness(stack: np.ndarray, reference: int) -> tuple[np.ndarray, np.ndarray]:
    """Scale every time point so that its mean equals the reference time point's mean.

    Parameters
    ----------
    stack
        Time series, time first (not modified).
    reference
        The time point every other is matched to.

    Returns
    -------
    tuple of numpy.ndarray
        The scaled series (float64) and the factor of each time point (``mean_ref /
        mean_t``; 1 for the reference).

    Raises
    ------
    TimeSeriesError
        For an invalid series or reference, a time point whose mean is not above 0 (it
        cannot be scaled to a positive mean), a factor float64 cannot hold (an infinite
        factor, or a positive ratio rounded to 0, which would erase the time point), or a
        scaled value that is not finite (float64 overflow): never returned as infinite or
        erased pixels. Finiteness is checked one time point at a time, so no array larger
        than one time point is allocated besides the result.
    """
    means = frame_means(stack)
    stack = _checked(np.asarray(stack), reference)
    low = np.flatnonzero(~(means > 0))
    if low.size:
        raise TimeSeriesError(
            f"Time point(s) {', '.join(map(str, low))} have a mean intensity that is not above 0, "
            "so they cannot be scaled to the reference's mean; use histogram matching or no "
            "correction."
        )
    with np.errstate(over="ignore", under="ignore"):
        factors = means[reference] / means
    factors[reference] = 1.0
    bad = np.flatnonzero(~np.isfinite(factors) | ~(factors > 0))
    scaled = np.empty(stack.shape, dtype=np.float64)
    if not bad.size:
        with np.errstate(over="ignore"):
            for t, factor in enumerate(factors):
                np.multiply(stack[t], factor, out=scaled[t], dtype=np.float64)
                if not np.isfinite(scaled[t]).all():
                    bad = np.append(bad, t)
    if bad.size:
        raise TimeSeriesError(
            f"Scaling time point(s) {', '.join(map(str, bad))} to the reference's mean needs a "
            "factor or gives values float64 cannot hold (a mean many orders of magnitude from "
            "the reference's); use histogram matching or no correction."
        )
    return scaled, factors


def match_histograms(stack: np.ndarray, reference: int) -> np.ndarray:
    """Map every time point's intensity distribution onto the reference time point's.

    Parameters
    ----------
    stack
        Time series, time first (not modified).
    reference
        The time point whose distribution every other takes.

    Returns
    -------
    numpy.ndarray
        The matched series (float64); the reference time point is unchanged.

    Raises
    ------
    TimeSeriesError
        For an invalid series or reference.
    """
    stack = _checked(stack, reference)
    ref_values, ref_counts = np.unique(stack[reference], return_counts=True)
    ref_cdf = np.cumsum(ref_counts, dtype=np.float64) / stack[reference].size
    out = np.empty(stack.shape, dtype=np.float64)
    for t in range(stack.shape[0]):
        if t == reference:
            out[t] = stack[t]
            continue
        _, inverse, counts = np.unique(stack[t], return_inverse=True, return_counts=True)
        cdf = np.cumsum(counts, dtype=np.float64) / stack[t].size
        mapped = np.interp(cdf, ref_cdf, ref_values.astype(np.float64))
        out[t] = mapped[inverse].reshape(stack[t].shape)
    return out
