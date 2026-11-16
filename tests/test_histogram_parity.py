"""The per-bin work of integer planes in NumPy int64: bitwise the outputs of v0.7.0.

``tests/data/histogram_v070_parity.json`` holds what v0.7.0 (``c6a9f9d``) gave for the
planes of ``tests/histogram_parity_cases.py`` (the preprocessing study's differential check,
made smaller). Every plane must give the same bits now, through the int64 paths of
``integer_bins`` and, with those paths switched off, through the Python integers kept for
64-bit ranges. The conditions choosing between the two, and each int64 formula against the
Python one, are tested directly below.
"""

import bisect
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from histogram_parity_cases import BASE_COMMIT, outputs, planes, variants
from morphobricks_normalization import compute_plane_histogram, integer_bins
from morphobricks_normalization import histogram as histogram_module
from morphobricks_normalization.histogram import PlaneHistogram
from morphobricks_normalization.integer_bins import (
    EXACT_FLOAT_TOTAL,
    INT64_STARTS_LIMIT,
    counts_from_per_value,
    integer_bin_index,
    integer_bin_offsets,
    integer_bin_starts,
    integer_bin_starts_int64,
    search_starts,
)

V070 = json.loads((Path(__file__).parent / "data" / "histogram_v070_parity.json").read_text())
PLANES = list(planes())


def _digest(plane, bin_counts):
    texts = [outputs(plane, *variant) for variant in variants(plane, bin_counts)]
    return [len(texts), hashlib.sha256("\n".join(texts).encode()).hexdigest()]


def test_the_table_is_v070s_and_covers_every_plane():
    assert V070["base_commit"] == BASE_COMMIT == "c6a9f9d"
    assert set(V070["planes"]) == {label for label, _, _ in PLANES}
    # The ten pixel types, and both sides of the int64 limits of integer_bins.
    assert {plane.dtype.name for _, plane, _ in PLANES} == {
        "uint8",
        "int8",
        "uint16",
        "int16",
        "uint32",
        "int32",
        "uint64",
        "int64",
        "float32",
        "float64",
    }
    assert sum(count for count, _ in V070["planes"].values()) > 4000


@pytest.mark.parametrize(
    ("label", "plane", "bin_counts"), PLANES, ids=[label for label, _, _ in PLANES]
)
def test_every_output_is_bitwise_v070s(label, plane, bin_counts):
    assert _digest(plane, bin_counts) == V070["planes"][label]


# Integer planes, with the bin counts for which the Python path is quick.
INTEGER_PLANES = [case for case in PLANES if case[1].dtype.kind in "iu" and case[1].size <= 35]
INTEGER_PLANES += [case for case in PLANES if "_range_" in case[0]]


@pytest.mark.parametrize(
    ("label", "plane", "bin_counts"), INTEGER_PLANES, ids=[label for label, _, _ in INTEGER_PLANES]
)
def test_the_python_path_kept_for_wide_ranges_gives_the_same_bits(
    monkeypatch, label, plane, bin_counts
):
    quick = [bins for bins in bin_counts if bins is not None and bins <= 4096]
    fast = [outputs(plane, *variant) for variant in variants(plane, quick)]
    # No int64 starts at all: every limit, sum and representative takes the Python integers.
    monkeypatch.setattr(integer_bins, "INT64_STARTS_LIMIT", 0)
    assert integer_bin_starts_int64(1, 1) is None
    assert [outputs(plane, *variant) for variant in variants(plane, quick)] == fast
    assert fast


@pytest.mark.parametrize("perturbation", ["bin_index", "representative"])
def test_the_table_catches_a_one_step_change(monkeypatch, perturbation):
    """Negative control: one bin or one float64 step off is a different digest."""
    label, plane, bin_counts = next(p for p in PLANES if p[0] == "uint16_120x97_k4")
    if perturbation == "bin_index":
        original = histogram_module.integer_bin_index
        monkeypatch.setattr(
            histogram_module, "integer_bin_index", lambda o, w, n: min(original(o, w, n) + 1, n)
        )
    else:
        original = histogram_module.integer_representatives
        monkeypatch.setattr(
            histogram_module,
            "integer_representatives",
            lambda h: np.nextafter(original(h), np.inf),
        )
    assert _digest(plane, bin_counts) != V070["planes"][label]


# --- each int64 formula against the Python integers, and the conditions between them ---


def test_int64_bin_starts_are_the_python_integers():
    widths = (*range(1, 70), 255, 256, 257, 4900, 65_535, 65_536, 65_537, 2**24 + 1, 2**32)
    pairs = [(w, n) for w in widths for n in (1, 2, 3, 7, 64, 65, 256, 500, 10_000, 65_536)]
    pairs += [(2**40 + 1, 500), (2**46 - 1, 65_536), (2**51, 2_047), (2**50 - 1, 4096)]
    for width, n in pairs:
        starts = integer_bin_starts_int64(width, n)
        assert starts.dtype == np.int64
        assert starts.tolist() == integer_bin_starts(width, n), (width, n)
        assert search_starts(width, n).tolist() == integer_bin_starts(width, n)[:-1]


def test_bin_starts_take_python_integers_from_2_62():
    # Below the limit: int64; at it: None, so that i * width can never overflow int64.
    assert integer_bin_starts_int64(2**50 - 1, 4096) is not None
    assert integer_bin_starts_int64(2**50, 4096) is None
    assert 4096 * 2**50 == INT64_STARTS_LIMIT
    assert integer_bin_starts_int64(2**64, 3) is None
    starts = search_starts(2**64, 3)  # a full uint64 plane
    assert starts.dtype == np.uint64
    assert starts.tolist() == integer_bin_starts(2**64, 3)[:-1]


def test_bin_offsets_keep_python_integers_for_numpy_limits():
    # Histograms built by a caller with NumPy integer limits keep v0.7.0's NumPy results.
    first, last = integer_bin_offsets(np.int64(10), np.int64(20), 4)
    assert isinstance(first, list) and isinstance(last, list)
    first, last = integer_bin_offsets(10, 20, 4)
    assert first.tolist() == [0, 3, 6, 9] and last.tolist() == [2, 5, 8, 10]
    counts = np.array([3, 3, 3, 2], np.int64)
    numpy_limits = PlaneHistogram(
        np.dtype("uint16"), 4, np.int64(10), np.int64(20), counts, 11, 11, 10, 20
    )
    python_limits = PlaneHistogram(np.dtype("uint16"), 4, 10, 20, counts, 11, 11, 10, 20)
    assert numpy_limits.integer_bin_limits() == python_limits.integer_bin_limits()
    assert type(numpy_limits.integer_bin_limits()[0][0]) is np.int64
    assert type(python_limits.integer_bin_limits()[0][0]) is int
    assert type(numpy_limits.max_representative_error) is np.float64
    assert type(python_limits.max_representative_error) is float
    assert numpy_limits.max_representative_error == python_limits.max_representative_error == 1.0
    np.testing.assert_array_equal(numpy_limits.representatives, python_limits.representatives)


@pytest.mark.parametrize("width", [1, 2, 3, 7, 40])
@pytest.mark.parametrize("n", [1, 2, 3, 5, 40, 41])
def test_bin_index_is_the_search_of_the_starts(width, n):
    starts = integer_bin_starts(width, n)
    for offset in range(-3, width + 4):
        assert integer_bin_index(offset, width, n) == bisect.bisect_right(starts, offset, hi=n) - 1


def test_bin_index_of_wide_ranges_is_exact():
    rng = np.random.default_rng(7)
    for width, n in [(2**64, 65_536), (2**64, 3), (2**51 + 3, 500), (10**6 + 1, 65_536)]:
        starts = integer_bin_starts(width, n)
        offsets = [0, width - 1, width, *starts[1:4], *(s - 1 for s in starts[1:4])]
        offsets += [int(x) for x in rng.integers(0, width - 1, 50, dtype=np.uint64, endpoint=True)]
        for offset in offsets:
            expected = bisect.bisect_right(starts, offset, hi=n) - 1
            assert integer_bin_index(offset, width, n) == expected


def _v070_counts(per_value, n):
    """The v0.7.0 sums: float64 weights per bin, rounded back to integers."""
    starts = integer_bin_starts(per_value.size, n)
    bin_of_value = np.searchsorted(np.array(starts[:-1]), np.arange(per_value.size), side="right")
    return np.rint(np.bincount(bin_of_value - 1, weights=per_value, minlength=n)).astype(np.int64)


@pytest.mark.parametrize(("width", "n"), [(1, 1), (5, 3), (3, 5), (1000, 7), (70_001, 65_536)])
def test_int64_sums_of_per_value_counts_are_v070s(width, n):
    per_value = np.random.default_rng(width).integers(0, 10**6, width).astype(np.int64)
    per_value[::3] = 0
    counts = counts_from_per_value(per_value, n)
    assert counts.dtype == np.int64
    np.testing.assert_array_equal(counts, _v070_counts(per_value, n))
    assert int(counts.sum()) == int(per_value.sum())


def test_sums_from_2_52_pixels_stay_v070s_float_sums():
    # Never for one plane in memory; there v0.7.0's float64 sums are kept, so nothing changes.
    below = np.array([2**51, 2**51 - 2, 1], np.int64)
    assert float(below.sum()) < EXACT_FLOAT_TOTAL
    np.testing.assert_array_equal(counts_from_per_value(below, 1), [2**52 - 1])
    beyond = np.array([2**53, 1], np.int64)  # float64 cannot hold 2**53 + 1
    np.testing.assert_array_equal(counts_from_per_value(beyond, 1), _v070_counts(beyond, 1))
    assert counts_from_per_value(beyond, 1).tolist() == [2**53]


def test_representatives_below_2_51_are_python_midpoints():
    plane = np.array([-(2**51) + 1, 2**51 - 1, 0, 12345], np.int64)
    for n in (1, 3, 500, 65_536):
        hist = compute_plane_histogram(plane, n)
        first, _ = hist._integer_bin_limits
        assert isinstance(first, np.ndarray) == (n < 65_536)  # 65,536 * 2**52 >= 2**62
        limits = hist.integer_bin_limits()
        midpoints = [(a + b) / 2 for a, b in limits]
        assert hist.representatives.tobytes() == np.array(midpoints).tobytes()
        widest = max(b - a + 1 for a, b in limits)
        assert hist.max_representative_error == (widest - 1) / 2


def test_bin_offsets_keep_python_integers_for_a_numpy_bin_count():
    # A caller may build a histogram with a NumPy bin count. v0.7.0 then gave NumPy limits,
    # or refused with an OverflowError where i * width does not fit int64; int64 starts would
    # wrap there silently (negative limits for a uint64 plane), so the Python path is kept.
    counts = np.array([3, 3, 3, 2], np.int64)
    numpy_count = PlaneHistogram(np.dtype("uint16"), np.int64(4), 10, 20, counts, 11, 11, 10, 20)
    python_count = PlaneHistogram(np.dtype("uint16"), 4, 10, 20, counts, 11, 11, 10, 20)
    assert numpy_count.integer_bin_limits() == python_count.integer_bin_limits()
    assert type(numpy_count.integer_bin_limits()[0][1]) is np.int64
    assert type(numpy_count.max_representative_error) is np.float64
    assert numpy_count.representatives.tobytes() == python_count.representatives.tobytes()
    counts = np.zeros(65_536, np.int64)
    counts[[0, -1]] = 1
    wide = PlaneHistogram(np.dtype("uint64"), np.int64(65_536), 0, 2**50, counts, 2, 2, 0, 2**50)
    try:
        limits = wide.integer_bin_limits()
    except OverflowError:  # v0.7.0's refusal: Python int too large to convert to C long
        return
    assert limits[-1] == (2**50 - 2**34 + 1, 2**50)
