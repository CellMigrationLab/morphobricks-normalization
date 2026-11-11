import numpy as np
import pytest

from morphobricks_normalization import (
    NormalizationParameters,
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


def test_non_finite_scale_is_refused_as_overflow_not_zero_spread():
    plane = np.array([[-1.7e308, 1.7e308], [-1.7e308, 1.7e308]])
    pooled = PooledHistogram([compute_plane_histogram(plane)])
    with pytest.raises(ValueError, match="not a finite number") as caught:
        z_score(pooled)
    assert "spread is zero" not in str(caught.value)


def test_non_finite_scale_message_does_not_blame_sums_of_squares_for_min_max():
    plane = np.array([[-1.7e308, 1.7e308], [-1.7e308, 1.7e308]])
    pooled = PooledHistogram([compute_plane_histogram(plane)])
    with pytest.raises(ValueError, match="not a finite number") as caught:
        min_max(pooled)
    assert "sums of squares" not in str(caught.value)
    assert "spread is zero" not in str(caught.value)


ZSCORE = NormalizationParameters("z-score", 0.0, 1.0)


@pytest.mark.parametrize(
    "clip, message",
    [
        ((1.0, 0.0), r"clip must be \(low, high\) with low < high; got low=1.0, high=0.0"),
        ((0.5, 0.5), r"clip must be \(low, high\) with low < high; got low=0.5, high=0.5"),
        ((0.0, np.nan), r"clip limits must be finite numbers; got \(0.0, nan\)"),
        ((-np.inf, 1.0), r"clip limits must be finite numbers; got \(-inf, 1.0\)"),
        ((0.0,), r"clip must be two numbers, \(low, high\); got \(0.0,\)"),
        ((0.0, 1.0, 2.0), r"clip must be two numbers, \(low, high\); got \(0.0, 1.0, 2.0\)"),
        ("01", r"clip must be two numbers, \(low, high\); got '01'"),
        ((0.0, "x"), r"clip must be two numbers, \(low, high\); got \(0.0, 'x'\)"),
    ],
)
def test_apply_refuses_invalid_clip_limits_by_name(clip, message):
    # Issue #39: a reversed clip collapsed every pixel to one value without a word.
    pixels = np.array([-1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match=message):
        ZSCORE.apply(pixels, clip=clip)
    np.testing.assert_array_equal(pixels, [-1.0, 0.0, 1.0])


@pytest.mark.parametrize(
    "dtype", [np.uint8, np.int16, np.int64, bool, np.complex64, object, np.float16, "U4"]
)
def test_apply_refuses_result_types_other_than_float32_and_float64(dtype):
    # Issue #39: an integer result type truncated and wrapped the normalized values.
    name = np.dtype(dtype).name
    with pytest.raises(ValueError, match=f"dtype must be float32 or float64; got {name!r}"):
        ZSCORE.apply(np.array([-1.0, 0.0, 1.0]), dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, "float32", np.dtype("float64")])
def test_apply_accepts_float32_and_float64_and_keeps_non_finite_pixels(dtype):
    pixels = np.array([-1.0, 0.5, np.nan, np.inf])
    result = ZSCORE.apply(pixels, clip=(-0.5, 0.75), dtype=dtype)
    assert result.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(result, np.array([-0.5, 0.5, np.nan, 0.75], dtype=dtype))
