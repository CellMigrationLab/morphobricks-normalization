import numpy as np
import pytest

from morphobricks_normalization import (
    PooledHistogram,
    compute_plane_histogram,
    min_max,
    percentile_range,
    robust_z_score,
    z_score,
)
from morphobricks_normalization.normalization import NORMAL_CONSISTENCY

rng = np.random.default_rng(3)
PLANES = [rng.integers(50, 3000, (32, 32)).astype(np.uint16) for _ in range(3)]
POOLED = PooledHistogram([compute_plane_histogram(p) for p in PLANES])
ALL = np.concatenate([p.ravel() for p in PLANES]).astype(np.float64)


def test_min_max_uses_exact_extrema():
    params = min_max(POOLED)
    assert (params.offset, params.scale) == (ALL.min(), ALL.max() - ALL.min())
    result = params.apply(PLANES[0])
    assert result.dtype == np.float32
    assert result.min() >= 0 and result.max() <= 1


def test_percentile_range():
    params = percentile_range(POOLED, 1, 99)
    low = np.percentile(ALL, 1, method="inverted_cdf")
    high = np.percentile(ALL, 99, method="inverted_cdf")
    assert (params.offset, params.scale) == (low, high - low)
    clipped = params.apply(PLANES[0], clip=(0, 1))
    assert clipped.min() == 0 and clipped.max() == 1
    with pytest.raises(ValueError):
        percentile_range(POOLED, 99, 1)


def test_z_scores():
    params = z_score(POOLED)
    assert params.offset == pytest.approx(ALL.mean()) and params.scale == pytest.approx(ALL.std())
    median = np.percentile(ALL, 50, method="inverted_cdf")
    mad = np.percentile(np.abs(ALL - median), 50, method="inverted_cdf")
    robust = robust_z_score(POOLED)
    assert (robust.offset, robust.scale) == (median, pytest.approx(mad * NORMAL_CONSISTENCY))


def test_apply_does_not_modify_input():
    plane = PLANES[1].copy()
    z_score(POOLED).apply(plane, dtype=np.float64)
    np.testing.assert_array_equal(plane, PLANES[1])


def test_zero_spread_is_rejected_with_a_clear_message():
    constant = PooledHistogram([compute_plane_histogram(np.full((4, 4), 7, np.uint16))])
    for method in (min_max, z_score, robust_z_score):
        with pytest.raises(ValueError, match="spread is zero"):
            method(constant)
