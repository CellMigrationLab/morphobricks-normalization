import bisect

import numpy as np
import pytest

from morphobricks_normalization import (
    UnsupportedDtypeError,
    compute_block_histogram,
    compute_plane_histogram,
)
from morphobricks_normalization import histogram as histogram_module

rng = np.random.default_rng(42)


def test_uint16_default_is_one_bin_per_value():
    plane = rng.integers(100, 5000, (128, 128)).astype(np.uint16)
    hist = compute_plane_histogram(plane)
    assert hist.bin_count == 65_536
    assert (hist.binning_min, hist.binning_max) == (int(plane.min()), int(plane.max()))
    assert hist.is_exact
    width = hist.binning_max - hist.binning_min + 1
    expected = np.bincount(plane.ravel() - hist.binning_min, minlength=width)
    np.testing.assert_array_equal(hist.counts[:width], expected)
    assert hist.counts[width:].sum() == 0
    np.testing.assert_array_equal(hist.representatives[:width], np.arange(width) + hist.binning_min)


@pytest.mark.parametrize("bin_count", [1, 7, 1000, 4900, 8192])
def test_integer_bins_have_whole_number_contiguous_limits(bin_count):
    plane = np.arange(100, 5000, dtype=np.uint16)  # 4,900 distinct values
    hist = compute_plane_histogram(plane, bin_count)
    limits = [(a, b) for a, b in hist.integer_bin_limits() if a <= b]
    assert limits[0][0] == 100 and limits[-1][1] == 4999
    for (_, last), (first, _) in zip(limits, limits[1:], strict=False):
        assert first == last + 1  # no gaps, no overlaps
    widths = {b - a + 1 for a, b in limits}
    assert max(widths) - min(widths) <= 1
    # each value is counted in the bin whose limits contain it
    for (a, b), count in zip(hist.integer_bin_limits(), hist.counts, strict=True):
        assert count == max(0, b - a + 1)


def test_range_narrower_than_bin_count_is_exact():
    plane = rng.integers(0, 300, (64, 64)).astype(np.uint16)
    hist = compute_plane_histogram(plane, 1024)
    assert hist.is_exact and hist.max_representative_error == 0


def test_signed_integers_keep_negative_values():
    plane = rng.integers(-500, 500, (64, 64)).astype(np.int16)
    hist = compute_plane_histogram(plane)
    assert hist.binning_min == int(plane.min()) < 0
    assert hist.is_exact
    width = hist.binning_max - hist.binning_min + 1
    np.testing.assert_array_equal(
        hist.counts[:width], np.bincount(plane.ravel().astype(int) - hist.binning_min)
    )


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.int32, np.uint32])
def test_wide_integers_over_their_full_range(dtype):
    info = np.iinfo(dtype)
    plane = np.array([info.min, info.max, 0, 1, info.max - 1], dtype=dtype)
    hist = compute_plane_histogram(plane, 65_536)
    assert hist.counts.sum() == plane.size
    assert hist.counts[0] >= 1 and hist.counts[-1] >= 2  # minimum first, the two largest last
    limits = hist.integer_bin_limits()
    assert limits[0][0] == info.min and limits[-1][1] == info.max
    assert np.all(np.isfinite(hist.representatives))


@pytest.mark.parametrize("dtype", [np.int32, np.uint64])
def test_per_value_and_search_paths_agree(monkeypatch, dtype):
    # 8- and 16-bit planes are always counted per value; wider ones may use either path.
    plane = rng.integers(0, 40_000, (64, 64)).astype(dtype)
    fast = compute_plane_histogram(plane, 1000)
    monkeypatch.setattr(histogram_module, "_PER_VALUE_COUNT_LIMIT", 0)
    slow = compute_plane_histogram(plane, 1000)
    np.testing.assert_array_equal(fast.counts, slow.counts)


def test_bin_count_limits():
    with pytest.raises(ValueError, match="between 1 and 256"):
        compute_plane_histogram(np.zeros(4, np.uint8), 257)
    assert compute_plane_histogram(np.zeros(4, np.uint8)).bin_count == 256
    with pytest.raises(ValueError, match="between 2 and 65,536"):
        compute_plane_histogram(np.zeros(4), 1)
    with pytest.raises(ValueError, match="whole number"):
        compute_plane_histogram(np.zeros(4), 10.5)


@pytest.mark.parametrize("dtype", [bool, np.complex64, np.float16])
def test_unsupported_dtypes_are_rejected(dtype):
    with pytest.raises(UnsupportedDtypeError):
        compute_plane_histogram(np.zeros(4, dtype=dtype))


def test_float_minimum_in_first_bin_maximum_in_last():
    plane = rng.normal(0, 10, (64, 64))
    hist = compute_plane_histogram(plane, 100)
    assert hist.counts.sum() == plane.size == hist.valid_pixel_count
    assert hist.counts[0] >= 1 and hist.counts[-1] >= 1
    assert (hist.pixel_min, hist.pixel_max) == (plane.min(), plane.max())
    assert hist.max_representative_error == pytest.approx((plane.max() - plane.min()) / 200)


def test_float_default_bin_count_is_10000():
    assert compute_plane_histogram(rng.normal(size=16)).bin_count == 10_000


def test_float_non_finite_pixels_are_ignored_and_counted():
    plane = rng.normal(0, 1, 100)
    plane[:3] = [np.nan, np.inf, -np.inf]
    hist = compute_plane_histogram(plane, 50)
    assert (hist.total_pixel_count, hist.valid_pixel_count) == (100, 97)
    assert hist.non_finite_pixel_count == 3 and hist.counts.sum() == 97


def test_float_plane_without_finite_pixels():
    hist = compute_plane_histogram(np.full((4, 4), np.nan), 10)
    assert hist.is_empty and hist.binning_min is None and hist.binning_max is None
    assert hist.counts.sum() == 0


def test_float_constant_plane():
    hist = compute_plane_histogram(np.full((8, 8), 3.25), 10)
    assert (hist.binning_min, hist.binning_max) == (3.25, 3.25)
    assert hist.counts[0] == 64 and hist.counts[1:].sum() == 0
    assert hist.representatives[0] == 3.25 and hist.max_representative_error == 0


def test_float_extreme_range_does_not_overflow():
    plane = np.array([-1e308, 0.0, 1e308])
    hist = compute_plane_histogram(plane, 4)
    np.testing.assert_array_equal(hist.counts, [1, 0, 1, 1])
    assert np.all(np.isfinite(hist.representatives))


def test_float32_input():
    plane = rng.normal(0, 1, (32, 32)).astype(np.float32)
    hist = compute_plane_histogram(plane, 64)
    assert hist.dtype == np.float32 and hist.counts.sum() == plane.size
    assert hist.pixel_min == float(plane.min())


# Planes in blocks ----------------------------------------------------------------------

DTYPES = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
DTYPES += ["float32", "float64"]


def reference_histogram(plane, n):
    """The whole-plane histogram as computed before planes could be read in blocks.

    Kept here, independent of the package, so that block histograms are compared
    with the original formulas and not only with themselves.
    """
    values = np.asarray(plane).ravel()
    if values.dtype.kind == "f":
        finite = values[np.isfinite(values)].astype(np.float64)
        counts = np.zeros(n, dtype=np.int64)
        if finite.size == 0:
            return None, None, counts
        lo, hi = float(finite.min()), float(finite.max())
        if lo == hi:
            counts[0] = finite.size
        else:
            half_range = hi * 0.5 - lo * 0.5
            index = np.floor((finite * 0.5 - lo * 0.5) * (n / half_range)).astype(np.int64)
            np.clip(index, 0, n - 1, out=index)
            counts += np.bincount(index, minlength=n)
        return lo, hi, counts
    if values.size == 0:
        return None, None, np.zeros(n, np.int64)
    lo, hi = int(values.min()), int(values.max())
    width = hi + 1 - lo
    if width <= n:
        starts = [min(i, width) for i in range(n + 1)]
    else:
        starts = [-((-i * width) // n) for i in range(n + 1)]
    # Exact Python integers: no overflow for any 64-bit range.
    offsets = [int(v) - lo for v in values]
    index = [bisect.bisect_right(starts, o, hi=n) - 1 for o in offsets]
    return lo, hi, np.bincount(np.array(index, dtype=np.int64), minlength=n)


def random_plane(dtype, shape=(37, 53), seed=0):
    generator = np.random.default_rng(seed)
    dt = np.dtype(dtype)
    if dt.kind == "f":
        plane = generator.normal(100, 30, shape).astype(dt)
        plane.flat[[3, 10, plane.size - 2]] = [np.nan, np.inf, -np.inf]
        return plane
    info = np.iinfo(dt)
    plane = generator.integers(info.min, info.max, shape, dtype=dt, endpoint=True)
    plane.flat[0], plane.flat[-1] = info.min, info.max
    return plane


def split(plane, rows, cols):
    """Cut a plane into blocks at the given row and column positions."""
    row_edges = [0, *rows, plane.shape[0]]
    col_edges = [0, *cols, plane.shape[1]]
    return [
        plane[a:b, c:d]
        for a, b in zip(row_edges, row_edges[1:], strict=False)
        for c, d in zip(col_edges, col_edges[1:], strict=False)
    ]


def assert_same(block_hist, whole_hist):
    assert block_hist.dtype == whole_hist.dtype
    assert type(block_hist.binning_min) is type(whole_hist.binning_min)
    assert (block_hist.binning_min, block_hist.binning_max) == (
        whole_hist.binning_min,
        whole_hist.binning_max,
    )
    assert block_hist.total_pixel_count == whole_hist.total_pixel_count
    assert block_hist.valid_pixel_count == whole_hist.valid_pixel_count
    np.testing.assert_array_equal(block_hist.counts, whole_hist.counts)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bins", [None, 7, 200])
def test_block_histograms_equal_whole_plane_histograms(dtype, bins):
    plane = random_plane(dtype)
    blocks = split(plane, [1, 2, 20], [5, 52])  # uneven blocks, some a single row
    whole = compute_plane_histogram(plane, bins)
    in_blocks = compute_block_histogram(lambda: iter(blocks), plane.dtype, bins)
    assert_same(in_blocks, whole)
    lo, hi, counts = reference_histogram(plane, whole.bin_count)
    assert (whole.binning_min, whole.binning_max) == (lo, hi)
    np.testing.assert_array_equal(whole.counts, counts)


@pytest.mark.parametrize("dtype", DTYPES)
def test_block_histograms_of_narrow_ranges(dtype):
    # Narrow ranges away from zero: exact histograms with many empty bins.
    base = 100 if np.dtype(dtype).kind in "uf" else -50
    plane = (base + random_plane("uint8").astype(np.int64) % 5).astype(dtype)
    blocks = split(plane, [10], [17, 30])
    assert_same(
        compute_block_histogram(lambda: iter(blocks), plane.dtype, 64),
        compute_plane_histogram(plane, 64),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_constant_plane_in_blocks(dtype):
    plane = np.full((9, 12), 3, dtype=dtype)
    blocks = split(plane, [4], [6])
    hist = compute_block_histogram(lambda: iter(blocks), plane.dtype, 16)
    assert_same(hist, compute_plane_histogram(plane, 16))
    assert hist.counts[0] == plane.size


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_non_finite_and_empty_float_planes_in_blocks(dtype):
    plane = np.full((6, 8), np.nan, dtype=dtype)
    plane[2:4, 3] = [np.inf, -np.inf]
    blocks = split(plane, [3], [4])
    hist = compute_block_histogram(lambda: iter(blocks), plane.dtype, 10)
    assert_same(hist, compute_plane_histogram(plane, 10))
    assert hist.is_empty and hist.non_finite_pixel_count == plane.size
    # Finite values in one block only, the others all NaN.
    plane[5, 7] = 2.5
    blocks = split(plane, [3], [4])
    hist = compute_block_histogram(lambda: iter(blocks), plane.dtype, 10)
    assert_same(hist, compute_plane_histogram(plane, 10))
    assert (hist.binning_min, hist.valid_pixel_count) == (2.5, 1)


@pytest.mark.parametrize("dtype", DTYPES)
def test_empty_blocks_and_planes(dtype):
    plane = random_plane(dtype, (4, 5))
    empty = plane[:0]
    blocks = [empty, plane[:2], empty, plane[2:]]
    assert_same(
        compute_block_histogram(lambda: iter(blocks), plane.dtype, 8),
        compute_plane_histogram(plane, 8),
    )
    hist = compute_block_histogram(lambda: iter([empty]), plane.dtype, 8)
    assert hist.is_empty and hist.total_pixel_count == 0 and hist.counts.sum() == 0


@pytest.mark.parametrize("dtype", [np.int32, np.uint64])
def test_block_search_path_agrees_with_per_value_path(monkeypatch, dtype):
    plane = rng.integers(0, 40_000, (40, 30)).astype(dtype)
    blocks = split(plane, [13], [11])
    fast = compute_block_histogram(lambda: iter(blocks), plane.dtype, 1000)
    monkeypatch.setattr(histogram_module, "_PER_VALUE_COUNT_LIMIT", 0)
    slow = compute_block_histogram(lambda: iter(blocks), plane.dtype, 1000)
    assert_same(fast, slow)
    assert_same(fast, compute_plane_histogram(plane, 1000))


@pytest.mark.parametrize(("dtype", "passes"), [("uint8", 1), ("int16", 1), ("uint32", 2)])
def test_blocks_are_read_once_for_small_integers_and_twice_otherwise(dtype, passes):
    plane = random_plane(dtype, (6, 6))
    calls = []

    def blocks():
        calls.append(1)
        return iter(split(plane, [3], []))

    compute_block_histogram(blocks, plane.dtype, 4)
    assert len(calls) == passes
    calls.clear()
    floats = plane.astype(np.float32)
    compute_block_histogram(lambda: (calls.append(1), iter([floats]))[1], floats.dtype, 4)
    assert len(calls) == 2  # floats are always read twice
