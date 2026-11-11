"""Centred tail moments (schema version 5) against an exact Fraction oracle (#91 owner question).

A tail stores its pixels' count, mean and population standard deviation, each the float64
nearest its exact value. The oracle here takes every pixel at its exact rational value, so
nothing in it can cancel, overflow or round; the square root is taken in 60 digits and
rounded once. The raw sums of squares of the draft format are shown to cancel on the same
pixels.
"""

from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

from morphobricks_normalization import (
    HistogramSettings,
    PooledHistogram,
    compute_block_histogram,
    compute_plane_histogram,
)
from morphobricks_normalization.tail_moments import (
    TailMoments,
    centred_moments,
    counted_sums,
    exact_sums,
    sqrt_nearest,
)

ROBUST = HistogramSettings(histogram_range="robust", robust_quantiles=(2.0, 98.0))


def oracle(values):
    """``(count, mean, std)`` of ``values``, exact and rounded once to float64."""
    exact = [Fraction(v) for v in np.asarray(values).ravel().tolist()]
    n = len(exact)
    mean = sum(exact) / n
    variance = sum((v - mean) ** 2 for v in exact) / n
    return n, float(mean), _sqrt(variance)


def _sqrt(value):
    with localcontext() as context:
        context.prec = 60
        return float((Decimal(value.numerator) / Decimal(value.denominator)).sqrt())


def moments(values):
    tail = TailMoments()
    tail.add(np.asarray(values))
    return tail.moments()


@pytest.mark.parametrize(
    "values",
    [
        1e12 + np.random.default_rng(1).uniform(0, 1e-3, 500),  # large values, tiny spread
        -3e15 + np.arange(400, dtype=np.float64) * 0.5,
        np.random.default_rng(2).normal(0, 1, 300) * 1e200,  # squares beyond float64
        np.array([5e-324, 1e-320, 2.5e-310]),  # subnormals
        np.array([-1.7e308, 1.7e308, 1.7e308]),
        np.array([7.25] * 9),  # a constant tail: std exactly 0
        np.random.default_rng(3).normal(100, 10, 1000).astype(np.float32),
    ],
)
def test_float_tails_are_the_exact_moments_rounded_once(values):
    assert moments(values) == oracle(values)


@pytest.mark.parametrize(
    "values",
    [
        np.array([2**62 + 1, 2**62 + 3, 2**62 + 4, 2**62 + 4], np.int64),  # beyond float64
        np.array([2**64 - 1, 2**64 - 2, 2**63], np.uint64),
        np.array([-(2**63), -(2**63) + 7], np.int64),
        np.random.default_rng(4).integers(0, 65535, 2000).astype(np.uint16),
    ],
)
def test_integer_tails_are_exact_even_where_float64_cannot_hold_the_values(values):
    exact = [int(v) for v in values.tolist()]
    n = len(exact)
    mean = Fraction(sum(exact), n)
    variance = sum((v - mean) ** 2 for v in exact) / n
    assert moments(values) == (n, float(mean), _sqrt(variance))


def test_large_values_with_a_small_spread_do_not_cancel():
    # Raw float64 sums lose the spread: the sum of squares is about 5e20, whose float64 step
    # (65536) is far above the 500 * 8.3 the centred squares add up to.
    values = 1e9 + np.random.default_rng(5).uniform(0, 10, 500)
    n, mean, std = moments(values)
    assert std == oracle(values)[2] and 2 < std < 4
    s, s2 = float(np.sum(values)), float(np.sum(values * values))
    raw = np.sqrt(max(s2 - s * (s / n), 0.0) / n)  # the draft's formula
    assert abs(raw - std) > std  # wrong by more than the spread itself


def test_blocks_in_any_order_give_the_same_bits():
    values = np.random.default_rng(6).normal(5e9, 3.0, 3000)
    whole = moments(values)
    for pieces in (np.array_split(values, 7), np.array_split(values[::-1], 3)):
        tail = TailMoments()
        for piece in pieces:
            tail.add(piece)
        assert tail.moments() == whole


def test_exact_sums_and_counted_sums_agree_with_fractions():
    values = np.array([0.1, -2.5, 1e300, 3e-320])
    total, squares = exact_sums(values)
    assert total == sum(Fraction(v) for v in values.tolist())
    assert squares == sum(Fraction(v) ** 2 for v in values.tolist())
    counts = np.array([0, 3, 0, 2, 5], np.int64)
    total, squares = counted_sums(counts, 1000)
    assert (total, squares) == (
        3 * 1001 + 2 * 1003 + 5 * 1004,
        3 * 1001**2 + 2 * 1003**2 + 5 * 1004**2,
    )


@pytest.mark.parametrize(
    "value",
    [Fraction(2), Fraction(1, 3), Fraction(10**400 + 1, 7), Fraction(1, 10**600), Fraction(0)],
)
def test_square_roots_are_rounded_once(value):
    assert sqrt_nearest(value) == (_sqrt(value) if value else 0.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        sqrt_nearest(Fraction(-1))


def test_no_values_give_zeros():
    assert centred_moments(0, Fraction(0), Fraction(0)) == (0, 0.0, 0.0)


def test_histogram_tails_whole_and_in_blocks_are_the_oracle_of_their_pixels():
    plane = 1e9 + np.random.default_rng(7).uniform(0, 10, (60, 70))
    whole = compute_plane_histogram(plane, 256, settings=ROBUST)
    blocks = compute_block_histogram(
        lambda: iter(np.array_split(plane, 9)), plane.dtype, 256, settings=ROBUST
    )
    for name, pixels in (
        ("underflow", plane[plane < whole.binning_min]),
        ("overflow", plane[plane > whole.binning_max]),
    ):
        tail = getattr(whole, name)
        assert (tail.count, tail.mean, tail.std) == oracle(pixels)
        assert getattr(blocks, name) == tail  # bit for bit


def test_pooled_tails_are_exact_given_the_stored_moments_and_close_to_the_pixels():
    rng = np.random.default_rng(8)
    planes = [1e9 + rng.uniform(0, 10, (50, 50)) + k * 2.0 for k in range(3)]
    histograms = [compute_plane_histogram(p, 64, settings=ROBUST) for p in planes]
    pooled = PooledHistogram(histograms)
    # Exact pooling of what is stored: bins at their representatives, tails by moments.
    stored = list(
        zip(pooled.bin_counts.tolist(), map(Fraction, pooled.bin_values.tolist()), strict=True)
    )
    n = pooled.pixel_count
    mean = (sum(c * v for c, v in stored) + sum(c * Fraction(m) for c, m, _ in pooled.tails)) / n
    squares = sum(c * (v - mean) ** 2 for c, v in stored)
    squares += sum(c * Fraction(s) ** 2 + c * (Fraction(m) - mean) ** 2 for c, m, s in pooled.tails)
    assert pooled.std == pytest.approx(_sqrt(squares / n), rel=1e-9)
    # Against the pixels themselves: the bins' half-width is the only approximation.
    assert pooled.std == pytest.approx(oracle(np.concatenate(planes))[2], rel=1e-2)
