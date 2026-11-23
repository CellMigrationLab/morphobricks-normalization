"""Brightness of a time series (contract §3.7): each function against a direct implementation."""

import numpy as np
import pytest

from morphobricks_normalization import (
    TimeSeriesError,
    brightness_trend,
    equalise_brightness,
    frame_means,
    match_histograms,
)

rng = np.random.default_rng(7)
# A 3-D time series (TZYX) that brightens by 5 % per time point, and a 2-D one (TYX).
BASE = rng.integers(100, 400, (3, 12, 12)).astype(np.uint16)
STACK = np.stack([np.round(BASE * (1 + 0.05 * t)).astype(np.uint16) for t in range(7)])
PLANES = rng.normal(500, 40, (5, 16, 16))


def test_frame_means_pool_every_axis_but_time_in_float64():
    means = frame_means(STACK)
    assert means.dtype == np.float64 and means.shape == (7,)
    assert means == pytest.approx([STACK[t].astype(np.float64).mean() for t in range(7)])
    assert frame_means(PLANES) == pytest.approx(PLANES.mean(axis=(1, 2)))


def test_brightness_trend_is_the_slope_relative_to_the_reference():
    means = frame_means(STACK)
    slope = np.polyfit(np.arange(7.0), means, 1)[0]
    assert brightness_trend(means, 3) == slope / means[3]
    assert brightness_trend(means, 3) == pytest.approx(0.05 / 1.15, rel=0.02)
    assert brightness_trend(np.full(4, 10.0), 0) == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(brightness_trend(np.array([0.0, 1.0, 2.0]), 0))  # reference mean 0
    assert np.isnan(brightness_trend(np.array([5.0]), 0))  # no slope from one time point


def test_equalise_brightness_matches_every_mean_to_the_reference():
    before = STACK.copy()
    scaled, factors = equalise_brightness(STACK, 3)
    assert np.array_equal(STACK, before)  # never modified
    means = frame_means(STACK)
    assert np.array_equal(factors, np.where(np.arange(7) == 3, 1.0, means[3] / means))
    assert scaled.dtype == np.float64
    assert np.array_equal(scaled, STACK.astype(np.float64) * factors[:, None, None, None])
    assert frame_means(scaled) == pytest.approx(np.full(7, means[3]))
    assert np.array_equal(scaled[3], STACK[3].astype(np.float64))  # the reference unchanged
    with pytest.raises(TimeSeriesError, match="not above 0"):
        equalise_brightness(np.zeros((3, 4, 4)), 1)
    with pytest.raises(TimeSeriesError, match="float64 cannot hold"):
        equalise_brightness(np.array([[5e-324], [1.0]]), 1)  # factor overflows to infinity
    with pytest.raises(TimeSeriesError, match="float64 cannot hold"):
        # 1e-200 / 1e200 rounds to 0: the second time point would be erased (re-validation R5).
        equalise_brightness(np.array([[1e-200], [1e200]]), 0)
    floats = rng.uniform(1, 1e4, (4, 3, 5))  # float input: the same products as broadcasting
    scaled, factors = equalise_brightness(floats, 2)
    assert np.array_equal(scaled, floats * factors[:, None, None])


def test_match_histograms_gives_every_time_point_the_reference_distribution():
    before = STACK.copy()
    matched = match_histograms(STACK, 3)
    assert np.array_equal(STACK, before)
    assert matched.dtype == np.float64
    assert np.array_equal(matched[3], STACK[3].astype(np.float64))
    # Direct implementation: p' = F_ref^-1(F_t(p)), linear between the reference's values.
    ref = np.sort(STACK[3].ravel()).astype(np.float64)
    values, counts = np.unique(STACK[3], return_counts=True)
    ref_cdf = np.cumsum(counts) / STACK[3].size
    for t in (0, 6):
        flat = STACK[t].ravel()
        own_cdf = np.array([np.mean(flat <= v) for v in flat])
        expected = np.interp(own_cdf, ref_cdf, values.astype(np.float64))
        assert matched[t].ravel() == pytest.approx(expected)
        # Monotonic: a brighter pixel never becomes darker.
        order = np.argsort(flat, kind="stable")
        assert np.all(np.diff(matched[t].ravel()[order]) >= 0)
    assert frame_means(matched)[0] == pytest.approx(ref.mean(), rel=0.02)


def test_refusals_name_the_problem():
    with pytest.raises(TimeSeriesError, match="first axis"):
        frame_means(np.arange(5.0))
    with pytest.raises(TimeSeriesError, match="non-finite"):
        frame_means(np.array([[1.0, np.nan]]))
    with pytest.raises(TimeSeriesError, match="not an integer or a floating-point"):
        frame_means(np.ones((2, 3), dtype=bool))
    for bad in (7, -1, 1.0, True):
        with pytest.raises(TimeSeriesError, match="reference time point"):
            match_histograms(STACK, bad)
    with pytest.raises(TimeSeriesError, match="one finite value per time point"):
        brightness_trend(np.ones((2, 2)), 0)
    assert isinstance(TimeSeriesError("x"), ValueError)
