"""Ultra-narrow float ranges (#86): bins no float64 value can fall in, and the centres' invariant.

The invariant, checked against every float64 value of narrow ranges (enumerated with
``nextafter``) and against the histograms of those values:

- a bin that holds no representable value ("unreachable") may share its centre with another;
- a bin that holds two or more representable values has a centre that maps back to it;
- a bin that holds exactly one may have a centre that rounds to the neighbouring float, which
  is in the adjacent bin (the midpoint of a one-float bin is not representable).
"""

import numpy as np
import pytest

from morphobricks_normalization import compute_plane_histogram
from morphobricks_normalization.bin_centres import float_bin_centres, float_bin_position
from morphobricks_normalization.bin_reachability import (
    centres_map_back,
    float_bin_indices,
    reachable_float_bins,
)

SMALLEST_SUBNORMAL = 5e-324
SMALLEST_NORMAL = float(np.finfo(np.float64).tiny)
MAX = float(np.finfo(np.float64).max)
BIN_COUNTS = (1, 2, 3, 4, 5, 7, 16, 17, 64, 256)
STARTS = (
    0.0,
    -0.0,
    SMALLEST_SUBNORMAL,
    -SMALLEST_SUBNORMAL,
    1e-320,
    SMALLEST_NORMAL,
    1e-300,
    0.1,
    1.0,
    -2.0,
    3.7,
    1e10,
    1e300,
    -1e-310,
    -1e300,
    1e308,
    -MAX,
    float(np.nextafter(MAX, 0.0)),
)
STEPS = (1, 2, 3, 4, 5, 6, 7, 8, 11, 20, 33, 100)


def representable(lo: float, steps: int) -> np.ndarray:
    """``steps + 1`` consecutive float64 values from ``lo`` (every value of the range)."""
    values = [lo]
    for _ in range(steps):
        with np.errstate(over="ignore"):  # stepping past the maximum gives infinity
            values.append(float(np.nextafter(values[-1], np.inf)))
    return np.array(values)


def narrow_ranges() -> list[tuple[float, int]]:
    cases = []
    for start in STARTS:
        for steps in STEPS:
            if np.isfinite(representable(start, steps)).all():
                cases.append((start, steps))
    return cases


def test_issue_example_smallest_subnormal_with_four_bins():
    lo, hi = 0.0, SMALLEST_SUBNORMAL
    assert reachable_float_bins(lo, hi, 4).tolist() == [True, False, False, True]
    centres = float_bin_centres(lo, hi, 4)
    assert centres.tolist() == [0.0, 0.0, SMALLEST_SUBNORMAL, SMALLEST_SUBNORMAL]
    # The unreachable inner bins share a centre with a reachable one; the reachable map back.
    assert centres_map_back(lo, hi, 4).tolist() == [True, False, False, True]
    assert float_bin_indices(np.array([lo, hi]), lo, hi, 4).tolist() == [0, 3]
    plane = np.array([0.0, SMALLEST_SUBNORMAL, SMALLEST_SUBNORMAL])
    hist = compute_plane_histogram(plane, 4)
    assert hist.counts.tolist() == [1, 0, 0, 2]
    assert hist.representatives.tolist() == centres.tolist()


@pytest.mark.parametrize(("lo", "steps"), narrow_ranges())
def test_reachable_bins_are_exactly_those_holding_a_representable_value(lo, steps):
    values = representable(lo, steps)
    hi = float(values[-1])
    for n in BIN_COUNTS:
        counts = np.bincount(float_bin_indices(values, lo, hi, n), minlength=n)
        np.testing.assert_array_equal(reachable_float_bins(lo, hi, n), counts > 0)


@pytest.mark.parametrize(("lo", "steps"), narrow_ranges())
def test_centre_of_every_bin_with_two_values_maps_back(lo, steps):
    values = representable(lo, steps)
    hi = float(values[-1])
    for n in BIN_COUNTS:
        counts = np.bincount(float_bin_indices(values, lo, hi, n), minlength=n)
        back = centres_map_back(lo, hi, n)
        assert back[counts >= 2].all(), (lo, hi, n)
        # Every value lands in a bin whose centre maps back, unless it is that bin's only value.
        landed = float_bin_indices(values, lo, hi, n)
        assert (back[landed] | (counts[landed] == 1)).all()


@pytest.mark.parametrize(("lo", "steps"), narrow_ranges())
def test_centre_of_a_one_value_bin_falls_in_an_adjacent_bin(lo, steps):
    values = representable(lo, steps)
    hi = float(values[-1])
    for n in BIN_COUNTS:
        counts = np.bincount(float_bin_indices(values, lo, hi, n), minlength=n)
        where = float_bin_indices(float_bin_centres(lo, hi, n), lo, hi, n)
        single = counts == 1
        assert (abs(where[single] - np.arange(n)[single]) <= 1).all(), (lo, hi, n)


def test_a_bin_with_one_value_can_have_a_centre_in_the_next_bin():
    # [0, 3 * smallest subnormal], 3 bins: bin 1 holds only 1 * 5e-324, its midpoint 1.5 *
    # 5e-324 rounds to 2 * 5e-324, which is in bin 2. Documented, not changed (#86).
    hi = 3 * SMALLEST_SUBNORMAL
    assert reachable_float_bins(0.0, hi, 3).all()
    assert centres_map_back(0.0, hi, 3).tolist() == [True, False, True]
    assert float_bin_centres(0.0, hi, 3).tolist() == [0.0, 2 * SMALLEST_SUBNORMAL, hi]


@pytest.mark.parametrize(("lo", "steps"), narrow_ranges())
def test_histograms_of_every_value_leave_exactly_the_unreachable_bins_empty(lo, steps):
    values = representable(lo, steps)
    hi = float(values[-1])
    for n in (2, 4, 7, 64):
        hist = compute_plane_histogram(values, n)
        assert (hist.binning_min, hist.binning_max) == (lo + 0.0, hi)
        reachable = reachable_float_bins(hist.binning_min, hist.binning_max, n)
        np.testing.assert_array_equal(hist.counts > 0, reachable)
        assert hist.counts.sum() == values.size
        centres_back = centres_map_back(hist.binning_min, hist.binning_max, n)
        np.testing.assert_array_equal(hist.representatives, float_bin_centres(lo, hi, n))
        assert centres_back[hist.counts >= 2].all()


def test_wide_ranges_map_every_centre_back():
    for lo, hi in ((-MAX, MAX), (0.0, MAX), (-1e308, 1e308), (-1.0, 1.0), (0.0, 1e-306)):
        for n in (2, 4, 16, 256, 1024):
            assert reachable_float_bins(lo, hi, n).all()
            assert centres_map_back(lo, hi, n).all()


def test_random_ranges_of_every_magnitude_map_every_reachable_centre_back():
    rng = np.random.default_rng(86)
    checked = 0
    while checked < 3000:
        exponents = rng.integers(-1074, 1024, 2).astype(float)
        signs = rng.choice([-1.0, 1.0], 2)
        with np.errstate(all="ignore"):
            lo, hi = sorted(signs * rng.random(2) * np.exp2(exponents))
        if not (np.isfinite([lo, hi]).all() and lo < hi):
            continue
        n = int(rng.choice([2, 3, 7, 16, 256, 1024]))
        reachable = reachable_float_bins(lo, hi, n)
        assert centres_map_back(lo, hi, n)[reachable].all(), (lo, hi, n)
        checked += 1


def test_bin_position_never_decreases_with_the_value():
    rng = np.random.default_rng(87)
    for lo, hi, n in ((0.0, SMALLEST_SUBNORMAL, 4), (-MAX, MAX, 256), (1.0, 1.0 + 1e-15, 64)):
        fractions = np.sort(rng.random(2000))
        values = np.clip(lo * (1 - fractions) + hi * fractions, lo, hi)
        assert (np.diff(float_bin_position(values, lo, hi, n)) >= 0).all()


def test_constant_range_has_one_reachable_bin():
    assert reachable_float_bins(2.5, 2.5, 4).tolist() == [True, False, False, False]
    assert float_bin_indices(np.array([2.5]), 2.5, 2.5, 4).tolist() == [0]
