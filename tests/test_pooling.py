import numpy as np
import pytest

from morphobricks_normalization import (
    NoValidPixelsError,
    PooledHistogram,
    compute_plane_histogram,
    group_records,
    pooled_statistics,
)
from morphobricks_normalization.pooling import DEFAULT_PERCENTILES
from simulation import direct_stats, make_records, simulate_experiment

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
