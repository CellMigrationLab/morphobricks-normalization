from fractions import Fraction

import numpy as np
import pytest

from morphobricks_normalization import (
    NormalizationSettings,
    NoValidPixelsError,
    PlaneRecord,
    PooledHistogram,
    compute_plane_histogram,
    fit,
    group_records,
    pooled_statistics,
    read_plane_histograms,
    write_plane_histograms,
    z_score,
)
from morphobricks_normalization.pooling import (
    DEFAULT_PERCENTILES,
    WeightedHistogram,
    _linear_percentile,
)
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


def test_tails_at_both_float64_limits_have_a_finite_exact_mean():
    # Float64 pixels at both ends overflowed the draft format's raw tail sums to +inf and
    # -inf (a named refusal then). Their tails' means are finite, and the pooled mean is
    # found by exact power-of-two scaling.
    plane = np.full((80, 80), 1.0)
    plane.ravel()[:5] = -1.7e308
    plane.ravel()[5:10] = 1.7e308
    plane.ravel()[10:] = np.linspace(0, 10, 6390)
    histogram = compute_plane_histogram(plane, settings=ROBUST)
    assert histogram.underflow.mean < -1e307 and histogram.overflow.mean > 1e307
    pooled = PooledHistogram([histogram])
    stored = [
        (c, Fraction(v)) for c, v in zip(pooled.bin_counts, pooled.bin_values.tolist(), strict=True)
    ]
    stored += [(c, Fraction(m)) for c, m, _ in pooled.tails]
    oracle = sum(c * v for c, v in stored) / pooled.pixel_count  # exact, from what is stored
    # Each count * value product is rounded once (as for the bins): within that bound.
    bound = sum(abs(c * v) for c, v in stored) / pooled.pixel_count * Fraction(2) ** -52
    assert abs(Fraction(pooled.mean) - oracle) <= bound
    # From the pixels: as close as float64 tail means allow (each within half a step of
    # 1.7e308, and the two tails cancel), never inf or nan.
    exact = sum(Fraction(v) for v in plane.ravel().tolist()) / plane.size
    assert abs(pooled.mean - float(exact)) <= 1e-15 * 1.7e308
    assert np.isfinite(pooled.std)


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


@pytest.mark.parametrize("q", [0, 25, 50, 75, 100])
@pytest.mark.parametrize("limit", [np.finfo(np.float64).max, 1e308])
def test_percentiles_from_pixels_near_float64_limits_are_finite_and_exact(q, limit):
    # End to end (#75): pixels at -limit and +limit -> histogram -> pooling -> percentile.
    # Bin centres reach twice the half range, which overflowed before the guard.
    from fractions import Fraction

    from morphobricks_normalization.histogram import compute_plane_histogram

    pixels = np.array([[-limit, limit]], dtype=np.float64)
    n = 4
    histogram = compute_plane_histogram(pixels, bin_count=n)
    low, high = Fraction(-limit), Fraction(limit)
    centres = [low + (high - low) * Fraction(2 * i + 1, 2 * n) for i in range(n)]
    assert np.isfinite(histogram.representatives).all()
    for got, want in zip(histogram.representatives, centres, strict=True):
        assert abs(Fraction(float(got)) - want) <= abs(want) * Fraction(1, 10**14)
    pooled = PooledHistogram([histogram])
    gamma = Fraction(q, 100)
    expected = centres[0] + (centres[-1] - centres[0]) * gamma
    result = pooled.linear_percentile(q)
    assert np.isfinite(result)
    assert abs(Fraction(result) - expected) <= abs(expected) * Fraction(1, 10**14)
    assert np.isfinite(pooled.percentile(q))


# Standard deviations of large finite values (#33) -------------------------------------


def exact_std(values, weights, tails=()):
    """Population standard deviation of weighted values and (count, mean, std) tails.

    Every float is taken at its exact value (Fraction) and the square root in 40 digits, so
    nothing here can overflow or round like the package's float64 arithmetic.
    """
    from decimal import Decimal, localcontext
    from fractions import Fraction

    values = [Fraction(v) for v in np.asarray(values, dtype=float).tolist()]
    weights = [Fraction(w) for w in np.asarray(weights, dtype=float).tolist()]
    tails = [tuple(Fraction(x) for x in tail) for tail in tails]
    total = sum(weights) + sum(c for c, _, _ in tails)
    mean = (
        sum(w * v for w, v in zip(weights, values, strict=True)) + sum(c * m for c, m, _ in tails)
    ) / total
    squares = sum(w * (v - mean) ** 2 for w, v in zip(weights, values, strict=True))
    squares += sum(c * s * s + c * (m - mean) ** 2 for c, m, s in tails)
    variance = squares / total
    with localcontext() as context:
        context.prec = 40
        return float((Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt())


LARGE_PLANES = {
    "opposite 1e200": [-1e200, 1e200],
    "1e200 +- 1e185": [1e200 - 1e185, 1e200, 1e200 + 1e185],
    "opposite 1e308": [-1e308, 0.0, 1e308],
    "constant 1e200": [1e200] * 5,
}


@pytest.mark.parametrize("name", LARGE_PLANES)
def test_std_of_large_finite_values_is_finite_and_exact(name):
    hist = compute_plane_histogram(np.array(LARGE_PLANES[name]), 64)
    pooled = PooledHistogram([hist])
    expected = exact_std(pooled.values, pooled.counts)
    assert np.isfinite(expected)
    assert pooled.std == pytest.approx(expected, rel=1e-14, abs=0)
    if name != "constant 1e200":
        parameters = z_score(pooled)
        assert parameters.scale == pooled.std and np.isfinite(parameters.offset)


def test_weighted_std_of_large_finite_values_is_finite_and_exact():
    parts = [
        PooledHistogram([compute_plane_histogram(np.array(LARGE_PLANES[name]), 64)])
        for name in ("opposite 1e200", "1e200 +- 1e185")
    ]
    weighted = WeightedHistogram(parts, [0.5, 2.0])
    expected = exact_std(weighted.values, weighted.weights)
    assert weighted.std == pytest.approx(expected, rel=1e-14, abs=0)


def test_std_with_large_tails_is_found_by_scaling():
    plane = np.random.default_rng(7).normal(0, 1, 10000) * 1e153
    settings = HistogramSettings(histogram_range="robust", robust_quantiles=(0.05, 99.95))
    hist = compute_plane_histogram(plane, 256, settings=settings)
    assert hist.has_tails and np.isfinite(hist.overflow.std)
    pooled = PooledHistogram([hist])
    expected = exact_std(pooled.bin_values, pooled.bin_counts, pooled.tails)
    assert pooled.std == pytest.approx(expected, rel=1e-12, abs=0)
    assert pooled.std == pytest.approx(np.std(plane / 1e153) * 1e153, rel=1e-4)


def test_tails_whose_squares_overflow_float64_give_the_exact_std(tmp_path):
    # The draft format's raw sums of squares overflowed here and the z-score was refused;
    # centred moments keep the spread, through the statistics file.
    plane = np.random.default_rng(1).normal(0, 1, 10000) * 1e150
    plane[:5] = (1e200, 2e200, -1e200, 3e200, -2e200)
    hist = compute_plane_histogram(plane, 256, settings=ROBUST)
    record = PlaneRecord("a.tif", "c", "r", "t", [hist], position_id="a", channels=["DNA"])
    (back,) = read_plane_histograms(
        write_plane_histograms([record], tmp_path / "p.csv", channel_ids=["DNA"])
    )
    pooled = PooledHistogram([back.histograms[0]])
    expected = exact_std(pooled.bin_values, pooled.bin_counts, pooled.tails)
    assert pooled.std == pytest.approx(expected, rel=1e-12, abs=0)
    assert pooled.std == pytest.approx(np.std(plane / 1e150) * 1e150, rel=1e-6)
    parameters = fit([back], NormalizationSettings(method="zscore", scope="image"), group="a")
    assert parameters.scale[0] == pooled.std
