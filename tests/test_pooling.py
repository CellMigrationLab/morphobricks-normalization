import numpy as np
import pytest

from morphobricks_normalization import (
    NoValidPixelsError,
    PooledHistogram,
    compute_plane_histogram,
    group_records,
    pooled_statistics,
)
from morphobricks_normalization.pooling import DEFAULT_PERCENTILES, _linear_percentile
from morphobricks_normalization.ranges import HistogramSettings
from simulation import direct_stats, make_records, simulate_experiment

ROBUST = HistogramSettings(histogram_range="robust", robust_quantiles=(1.0, 99.0))

LEVELS = ("image", "technical_repeat", "biological_repeat", "condition")


def pixels_by_group(planes, records, level):
    lookup = {path: pixels for _, path, pixels in planes}
    return {
        key: np.concatenate([lookup[r.source_relative_path].ravel() for r in members])
        for key, members in group_records(records, level).items()
    }


@pytest.mark.parametrize("level", LEVELS)
def test_exact_integer_histograms_reproduce_pooled_pixels(level):
    planes = simulate_experiment(np.uint16, seed=1)
    records = make_records(planes)  # default: one bin per value
    groups = group_records(records, level)
    for key, pixels in pixels_by_group(planes, records, level).items():
        stats = pooled_statistics(groups[key])
        ref = direct_stats(pixels, DEFAULT_PERCENTILES)
        assert stats.mean == pytest.approx(ref["mean"], rel=1e-12)
        assert stats.std == pytest.approx(ref["std"], rel=1e-12)
        assert stats.median == ref["median"] and stats.mad == ref["mad"]
        assert stats.percentiles == ref["percentiles"]
        assert (stats.minimum, stats.maximum) == (ref["minimum"], ref["maximum"])


@pytest.mark.parametrize(
    "dtype, bin_count",
    [
        (np.uint16, 1024),
        (np.uint16, 100),
        (np.float64, 8192),
        (np.float64, 1024),
        (np.float32, 256),
    ],
)
@pytest.mark.parametrize("level", LEVELS)
def test_percentile_error_never_exceeds_half_the_widest_bin(dtype, bin_count, level):
    planes = simulate_experiment(dtype, seed=2)
    records = make_records(planes, bin_count)
    groups = group_records(records, level)
    for key, pixels in pixels_by_group(planes, records, level).items():
        pooled = PooledHistogram(groups[key])
        bound = pooled.max_representative_error
        ref = direct_stats(pixels, DEFAULT_PERCENTILES)
        for q, value in ref["percentiles"].items():
            assert abs(pooled.percentile(q) - value) <= bound * (1 + 1e-9) + 1e-12
        assert abs(pooled.mad - ref["mad"]) <= 2 * bound * (1 + 1e-9) + 1e-12
        assert (pooled.minimum, pooled.maximum) == (ref["minimum"], ref["maximum"])


def test_planes_of_different_sizes_are_weighted_by_pixel_count():
    small = np.full(10, 100, np.uint16)
    large = np.full(990, 200, np.uint16)
    stats = pooled_statistics([compute_plane_histogram(small), compute_plane_histogram(large)])
    assert stats.mean == pytest.approx(199.0)  # not the mean of means, 150


def test_planes_without_finite_pixels_are_excluded():
    good = compute_plane_histogram(np.array([1.0, 2.0, 3.0]), 10)
    empty = compute_plane_histogram(np.full(5, np.nan), 10)
    stats = pooled_statistics([good, empty])
    assert (stats.n_planes, stats.n_excluded_planes, stats.pixel_count) == (1, 1, 3)
    with pytest.raises(NoValidPixelsError):
        pooled_statistics([empty])


def test_percentile_limits():
    pooled = PooledHistogram([compute_plane_histogram(np.arange(10, dtype=np.uint8))])
    assert pooled.percentile(0) == 0 and pooled.percentile(100) == 9
    with pytest.raises(ValueError):
        pooled.percentile(101)


def test_opposing_infinite_tail_sums_are_a_named_refusal():
    # Float64 pixels at both ends overflow the tails' sums to +inf and -inf; the mean then
    # needs a refusal that says why, not a bare error from math.fsum.
    plane = np.full((80, 80), 1.0)
    plane.ravel()[:5] = -1.7e308
    plane.ravel()[5:10] = 1.7e308
    plane.ravel()[10:] = np.linspace(0, 10, 6390)
    histogram = compute_plane_histogram(plane, settings=ROBUST)
    assert histogram.underflow.sum == -np.inf and histogram.overflow.sum == np.inf
    with pytest.raises(ValueError, match="opposite sign") as caught:
        _ = PooledHistogram([histogram]).mean
    assert "tail 1 of 2" in str(caught.value) and "tail 2 of 2" in str(caught.value)


@pytest.mark.parametrize("q", [0, 25, 50, 75, 100])
@pytest.mark.parametrize("low, high", [(-1e308, 1e308), (-1.7e308, 1.7e308), (-1.7e308, 5e307)])
def test_linear_percentile_between_finite_extremes_does_not_overflow(q, low, high):
    # The gap between the endpoints exceeds float64 so above - below is infinite, yet
    # the interpolation lies between them and is representable: it must be finite and
    # equal the exact value (computed with fractions) to float64 precision.
    from fractions import Fraction

    values = np.array([low, high])
    cumulative = np.array([1, 2])
    got = _linear_percentile(values, cumulative, 2, q)
    exact = Fraction(low) + (Fraction(high) - Fraction(low)) * Fraction(q, 100)
    assert np.isfinite(got)
    assert abs(Fraction(got) - exact) <= abs(exact) * Fraction(1, 10**15) + Fraction(1, 10**300)
