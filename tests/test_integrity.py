"""Regression tests for the integrity of the statistics (issues #25, #26, #31, #43-#48)."""

import csv

import numpy as np
import pytest

from morphobricks_normalization import (
    HistogramSettings,
    HistogramTail,
    NormalizationSettings,
    PlaneHistogram,
    PlaneRecord,
    PooledHistogram,
    compute_block_histogram,
    compute_plane_histogram,
    fit,
    histogram_with_range_check,
    image_summaries,
    read_plane_histograms,
    scope_groups,
    write_plane_histograms,
)
from morphobricks_normalization.csv_io import CsvFormatError
from morphobricks_normalization.integrity import LIMIT_NAMES

ROBUST = HistogramSettings(histogram_range="robust")
rng = np.random.default_rng(44)


def assert_same(a, b):
    assert a.dtype == b.dtype and a.bin_count == b.bin_count
    assert (a.pixel_min, a.binning_min, a.binning_max, a.pixel_max) == (
        b.pixel_min,
        b.binning_min,
        b.binning_max,
        b.pixel_max,
    )
    assert (a.total_pixel_count, a.valid_pixel_count) == (b.total_pixel_count, b.valid_pixel_count)
    np.testing.assert_array_equal(a.counts, b.counts)
    assert (a.underflow, a.overflow) == (b.underflow, b.overflow)


def plane_of(dtype):
    if np.dtype(dtype).kind == "f":
        plane = rng.normal(-3, 50, (40, 30)).astype(dtype)
        plane.ravel()[:3] = [np.nan, -1e6, 1e6]
        return plane
    info = np.iinfo(dtype)
    plane = rng.integers(max(info.min, -3000), min(info.max, 3000), (40, 30)).astype(dtype)
    plane.ravel()[:2] = [info.min, info.max]
    return plane


def blocks_of(plane):
    return lambda: (plane[:7], plane[7:20, ::2], plane[7:20, 1::2], plane[20:])


# --- Pixel blocks (#25, #26) -------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["u1", "i1", "u2", "i2", "u4", "i4", "u8", "i8", "f4", "f8"])
@pytest.mark.parametrize("settings", [None, ROBUST])
def test_other_byte_order_gives_the_histogram_of_the_values(dtype, settings):
    native = plane_of(np.dtype(dtype))
    for order in "<>":
        swapped = native.astype(np.dtype(dtype).newbyteorder(order))
        np.testing.assert_array_equal(swapped, native)  # same values, other bytes
        expected = compute_plane_histogram(native, 50, settings=settings)
        assert_same(compute_plane_histogram(swapped, 50, settings=settings), expected)
        blockwise = compute_block_histogram(
            blocks_of(swapped), swapped.dtype, 50, settings=settings
        )
        assert_same(blockwise, expected)
        checked, check = histogram_with_range_check(blocks_of(swapped), swapped.dtype, 50)
        assert_same(checked, compute_plane_histogram(native, 50))
        assert check == histogram_with_range_check(blocks_of(native), native.dtype, 50)[1]


def test_big_endian_uint16_values_are_not_reinterpreted():
    histogram = compute_plane_histogram(np.array([1, 2, 3], dtype=">u2"))
    assert (histogram.pixel_min, histogram.pixel_max) == (1, 3)
    assert histogram.counts[:3].tolist() == [1, 1, 1]


@pytest.mark.parametrize(
    "blocks, declared, observed",
    [
        ((np.array([1, 2], np.uint8),), np.uint16, "uint8"),
        ((np.array([1.5], np.float64),), np.float32, "float64"),
        ((np.array([-1, 2], np.int16),), np.uint16, "int16"),
        ((np.array([1], np.uint32),), np.int32, "uint32"),
        ((np.array([1, 2], np.uint16), np.array([3], np.int64)), np.uint16, "int64"),
    ],
)
def test_blocks_of_another_pixel_type_are_refused(blocks, declared, observed):
    index = len(blocks) - 1
    message = f"Block {index} of the plane holds {observed} pixels, but the plane's pixel type"
    with pytest.raises(ValueError, match=message):
        compute_block_histogram(lambda: blocks, declared)
    with pytest.raises(ValueError, match=message):
        histogram_with_range_check(lambda: blocks, declared, settings=ROBUST)


def test_empty_blocks_of_any_type_are_skipped():
    blocks = (np.array([], np.float64), np.array([1, 2], np.uint16))
    histogram = compute_block_histogram(lambda: blocks, np.uint16)
    assert histogram.valid_pixel_count == int(histogram.counts.sum()) == 2


# --- Pooled counts (#31) -----------------------------------------------------------------------


def singleton(count, value=0):
    counts = np.array([count], dtype=np.int64)
    return PlaneHistogram(
        np.dtype(np.uint8), 1, value, value, counts, count, count, pixel_min=value, pixel_max=value
    )


def test_pooled_counts_stay_exact_above_2_to_the_53():
    big = 2**53 + 1
    pooled = PooledHistogram([singleton(big)])
    assert pooled.pixel_count == big and pooled.counts.tolist() == [big]
    overlapping = PooledHistogram([singleton(big), singleton(big), singleton(3, 7)])
    assert overlapping.counts.tolist() == [2 * big, 3]
    assert overlapping.pixel_count == 2 * big + 3
    assert overlapping.percentile(100) == 7.0 and overlapping.percentile(0) == 0.0


def test_pooled_counts_beyond_int64_are_refused():
    with pytest.raises(ValueError, match="more than 2\\*\\*63 - 1 finite pixels"):
        PooledHistogram([singleton(2**62), singleton(2**62)])


# --- Histogram invariants (#45, #47) -----------------------------------------------------------


def histogram(**changes):
    fields = dict(
        dtype=np.dtype(np.uint8),
        bin_count=2,
        binning_min=0,
        binning_max=1,
        counts=np.array([1, 1], np.int64),
        total_pixel_count=2,
        valid_pixel_count=2,
        pixel_min=0,
        pixel_max=1,
    )
    return PlaneHistogram(**{**fields, **changes})


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"counts": np.array([3, -1], np.int64)}, "cannot be negative"),
        ({"counts": np.array([1.0, 1.0])}, "int64 array"),
        ({"counts": np.array([[1, 1]], np.int64)}, "int64 array"),
        ({"counts": np.array([2], np.int64)}, "1 bin counts for a bin count of 2"),
        ({"valid_pixel_count": 3}, "3 valid pixels but only 2"),
        ({"valid_pixel_count": -1}, "between 0 and 2"),
        ({"total_pixel_count": 2.0}, "whole number"),
        ({"valid_pixel_count": True}, "whole number"),
        ({"total_pixel_count": 3, "valid_pixel_count": 3}, "hold 2 pixel"),
        ({"binning_min": None, "binning_max": None}, "four finite intensity limits"),
        ({"binning_max": None, "pixel_max": None}, "four finite intensity limits"),
        ({"binning_min": 1, "binning_max": 0}, "must be ordered"),
        ({"binning_max": float("nan")}, "four finite intensity limits"),
        ({"pixel_min": -1}, "differ exactly when"),
    ],
)
def test_impossible_histograms_are_refused(changes, message):
    with pytest.raises(ValueError, match=message):
        histogram(**changes)


@pytest.mark.parametrize("field", LIMIT_NAMES)
def test_a_single_missing_limit_is_refused_not_filled_in(field):
    # Before #47 was fixed, a missing pixel_min or pixel_max was taken from the binning limit.
    with pytest.raises(ValueError, match=f"four finite intensity limits .*; {field} is missing"):
        histogram(**{field: None})


@pytest.mark.parametrize("field", ["pixel_min", "pixel_max"])
def test_a_missing_extremum_of_a_robust_histogram_is_refused(field):
    good = robust_plane_histogram()
    fields = {name: getattr(good, name) for name in good.__dataclass_fields__}
    with pytest.raises(ValueError, match=f"{field} is missing"):
        PlaneHistogram(**{**fields, field: None})


def test_computed_min_max_histograms_give_their_extrema():
    for plane in (np.array([3, 9, 4], np.uint8), np.array([1.5, -2.0, np.nan], np.float32)):
        hist = compute_plane_histogram(plane, 4)
        assert (hist.pixel_min, hist.pixel_max) == (hist.binning_min, hist.binning_max)
        assert (hist.pixel_min, hist.pixel_max) == (np.nanmin(plane), np.nanmax(plane))


def test_wrapping_int64_sum_is_refused():
    # The int64 sum of these counts wraps to 3 (issue #45).
    counts = np.array([2**62] * 4 + [3], np.int64)
    assert int(counts.sum()) == 3
    with pytest.raises(ValueError, match=f"bins hold {2**64 + 3} pixel"):
        histogram(
            bin_count=5, binning_max=4, counts=counts, total_pixel_count=3, valid_pixel_count=3
        )


def test_a_plane_without_valid_pixels_has_no_limits():
    empty = dict(counts=np.zeros(2, np.int64), valid_pixel_count=0)
    no_limits = dict.fromkeys(LIMIT_NAMES)
    assert histogram(**empty, **no_limits).is_empty
    with pytest.raises(ValueError, match="without valid pixels can have no intensity range"):
        histogram(**empty)


# --- Tails (#46) -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tail, message",
    [
        ((2, float("nan"), 1.0), "the sum must be a number"),
        ((2, 1.0, -1.0), "sum of squares 0 or more"),
        ((2, 1.0, float("nan")), "sum of squares 0 or more"),
        ((2, float("inf"), 4.0), "infinite sum"),
        ((2, 10.0, 1.0), "variance would be negative"),
        ((-1, 0.0, 0.0), "between 0 and 2"),
        ((0, 1.0, 0.0), "empty underflow or overflow must have zero sums"),
    ],
)
def test_impossible_tail_sums_are_refused(tail, message):
    with pytest.raises(ValueError, match=message):
        HistogramTail(*tail)


def test_signed_and_constant_tails_are_accepted():
    assert HistogramTail(2, -6.0, 18.0).mean == -3.0
    third = 0.1 + 0.2  # rounding of the sums must not refuse a constant tail
    assert HistogramTail(3, 3 * third, 3 * third * third).count == 3


def robust_plane_histogram(dtype=np.float64, outliers=(-1e6, 1e6)):
    # Ten distinct outliers at each end, so the robust limits leave some below and above.
    plane = rng.normal(0, 1, 40_000).astype(dtype)
    plane[:10] = np.linspace(outliers[0], outliers[0] / 10, 10)
    plane[10:20] = np.linspace(outliers[1] / 10, outliers[1], 10)
    return compute_plane_histogram(plane, 100, settings=ROBUST)


def test_tails_must_fit_their_range():
    good = robust_plane_histogram()
    assert not good.underflow.is_empty and not good.overflow.is_empty
    lo, above = good.underflow, good.binning_min + 1.0
    for bad in (
        HistogramTail(lo.count, lo.count * above, lo.count * above * above),  # above the limit
        HistogramTail(lo.count, lo.sum, lo.sum_of_squares * 10),  # values beyond the minimum
        HistogramTail(lo.count, lo.sum, float("inf")),  # cannot overflow in this range
    ):
        with pytest.raises(ValueError, match="which values in that range cannot have"):
            PlaneHistogram(
                good.dtype,
                good.bin_count,
                good.binning_min,
                good.binning_max,
                good.counts,
                good.total_pixel_count,
                good.valid_pixel_count,
                good.pixel_min,
                good.pixel_max,
                bad,
                good.overflow,
            )


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")  # issue #33
def test_a_sum_of_squares_that_overflows_float64_is_accepted_where_possible():
    # Values near 1e200 overflow their squares (issue #33): legitimate, not corruption.
    result = robust_plane_histogram(outliers=(-1e200, 1e200))
    assert result.underflow.sum_of_squares == np.inf and result.overflow.count >= 1


# --- The statistics file (#43, #44, #45, #46, #47, #48) ----------------------------------------


def record(path="c/r/a.tif", histograms=None, **coords):
    if histograms is None:
        histograms = [compute_plane_histogram(np.array([[0, 1], [1, 255]], np.uint8), 2)]
    return PlaneRecord(path, "c", "r", "t", histograms, **coords)


def edited(tmp_path, records, changes, row=0):
    path = write_plane_histograms(records, tmp_path / "plane_histograms.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        header, *rows = list(csv.reader(fh))
    for column, value in changes.items():
        rows[row][header.index(column)] = value
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([header, *rows])
    return path


LIMITS = (
    "Intensity_MinIntensity_C0",
    "Intensity_MaxIntensity_C0",
    "Histogram_BinningMin_C0",
    "Histogram_BinningMax_C0",
)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"Histogram_Count_C0_00000": "4", "Histogram_Count_C0_00001": "-1"}, "negative"),
        ({"Histogram_Count_C0_00000": str(2**63)}, "larger than 2\\*\\*63 - 1"),
        ({"Histogram_Count_C0_00000": "1.5"}, "invalid literal"),
        ({"Histogram_ValidPixelCount_C0": "-3", "total_pixel_count": "-3"}, "between 0"),
        ({"Histogram_ValidPixelCount_C0": "5"}, "5 valid pixels but only 4"),
        (dict.fromkeys(LIMITS, ""), "four finite intensity limits"),
        ({"Intensity_MaxIntensity_C0": "", "Histogram_BinningMax_C0": ""}, "four finite"),
    ],
)
def test_impossible_counts_and_limits_in_the_file_are_refused(tmp_path, changes, message):
    path = edited(tmp_path, [record()], changes)
    with pytest.raises(CsvFormatError, match=f"line 2: the bin counts of channel 0 .*{message}"):
        read_plane_histograms(path)


@pytest.mark.parametrize("column", LIMITS)
def test_a_single_blank_limit_in_the_file_is_refused(tmp_path, column):
    # The default record's limits are min-max, so a blank extremum equals its binning limit
    # and was silently filled in from it before #47 was fixed.
    path = edited(tmp_path, [record()], {column: ""})
    message = f"line 2: the bin counts of channel 0 .*{column} is blank, but the channel has 4"
    with pytest.raises(CsvFormatError, match=message):
        read_plane_histograms(path)


def test_blank_limits_of_a_channel_without_valid_pixels_are_read(tmp_path):
    empty = compute_plane_histogram(np.full((2, 2), np.nan, np.float32), 2)
    path = write_plane_histograms([record(histograms=[empty])], tmp_path / "plane_histograms.csv")
    (back,) = read_plane_histograms(path)
    assert back.histograms[0].is_empty
    assert [getattr(back.histograms[0], name) for name in LIMIT_NAMES] == [None] * 4


def test_wrapping_counts_in_the_file_are_refused(tmp_path):
    histograms = [compute_plane_histogram(np.arange(5, dtype=np.uint8), 5)]
    changes = {f"Histogram_Count_C0_{i:05d}": str(2**62) for i in range(4)}
    changes |= {"Histogram_Count_C0_00004": "3", "Histogram_ValidPixelCount_C0": "3"}
    path = edited(tmp_path, [record(histograms=histograms)], changes | {"total_pixel_count": "3"})
    with pytest.raises(CsvFormatError, match=f"bins hold {2**64 + 3} pixel"):
        read_plane_histograms(path)


@pytest.mark.parametrize(
    "column, value",
    [
        ("Histogram_UnderflowSum_C0", "nan"),
        ("Histogram_UnderflowSumOfSquares_C0", "-1"),
        ("Histogram_UnderflowSumOfSquares_C0", "inf"),
        ("Histogram_OverflowSum_C0", "inf"),
        ("Histogram_OverflowSum_C0", "0.0"),
    ],
)
def test_impossible_sums_of_a_populated_tail_are_refused(tmp_path, column, value):
    path = edited(tmp_path, [record(histograms=[robust_plane_histogram()])], {column: value})
    with pytest.raises(CsvFormatError, match="line 2: the bin counts of channel 0"):
        read_plane_histograms(path)


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")  # issue #33
def test_populated_tails_and_overflowed_squares_round_trip(tmp_path):
    records = [
        record("c/r/a.tif", [robust_plane_histogram()]),
        record("c/r/b.tif", [robust_plane_histogram(outliers=(-1e200, 1e200))]),
    ]
    back = read_plane_histograms(write_plane_histograms(records, tmp_path / "t.csv"))
    for a, b in zip(records, back, strict=True):
        assert_same(a.histograms[0], b.histograms[0])


def test_a_table_without_rows_is_refused(tmp_path):
    path = write_plane_histograms([record()], tmp_path / "plane_histograms.csv")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text(lines[0], encoding="utf-8")
    with pytest.raises(CsvFormatError, match="has the plane-histogram columns but no rows"):
        read_plane_histograms(path)


@pytest.mark.parametrize("conflicting", [False, True])
def test_a_repeated_plane_row_is_refused(tmp_path, conflicting):
    a, b = record("c/r/a.tif"), record("c/r/b.tif")
    path = write_plane_histograms([a, b], tmp_path / "plane_histograms.csv")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    repeated = lines[2]
    if conflicting:  # same plane, other counts: still one plane twice
        other = [compute_plane_histogram(np.array([[9, 9], [9, 200]], np.uint8), 2)]
        table = write_plane_histograms([record("c/r/b.tif", other)], tmp_path / "other.csv")
        repeated = table.read_text(encoding="utf-8").splitlines(keepends=True)[1]
        assert repeated != lines[2]
    path.write_text("".join([*lines, repeated]), encoding="utf-8")
    with pytest.raises(
        CsvFormatError, match="line 4: plane c/r/b.tif::z=0,t=0 is already in line 3"
    ):
        read_plane_histograms(path)


def test_planes_that_differ_in_z_t_or_folder_are_not_repeats(tmp_path):
    records = [
        record("c/r/a.tif"),
        record("c/r/a.tif", t=1),
        record("c/r/a.tif", z=2, t=1),
        record("c/s/a.tif"),
    ]
    back = read_plane_histograms(write_plane_histograms(records, tmp_path / "t.csv"))
    assert [r.plane_id for r in back] == [r.plane_id for r in records]


def test_repeated_planes_are_refused_by_every_entry_point(tmp_path):
    a, b = record("c/r/a.tif"), record("c/r/b.tif")
    message = "more than once: c/r/b.tif::z=0,t=0 \\(2 times\\)"
    with pytest.raises(ValueError, match=message):
        write_plane_histograms([a, b, b], tmp_path / "t.csv")
    assert not (tmp_path / "t.csv").exists()
    with pytest.raises(ValueError, match=message):
        scope_groups([a, b, b], NormalizationSettings())
    with pytest.raises(ValueError, match=message):
        fit([a, b, b], NormalizationSettings(method="minmax"))
    with pytest.raises(ValueError, match=message):
        image_summaries([a, b, b])


@pytest.mark.parametrize("value", [-1, True, 1.5, 1.0, "1", np.float64(2.0), np.bool_(True)])
@pytest.mark.parametrize("name", ["z", "t"])
def test_plane_coordinates_must_be_whole_numbers_from_zero(name, value):
    with pytest.raises(
        ValueError, match=f"its {name} coordinate must be a whole number 0 or above"
    ):
        record(**{name: value})


def test_valid_plane_coordinates():
    plane = record(z=np.int64(3), t=None)
    assert (plane.z, plane.t) == (3, 0) and type(plane.z) is int
    assert record(z=2**40, t=7).plane_id == f"c/r/a.tif::z={2**40},t=7"


@pytest.mark.parametrize("column", ["z", "t"])
def test_negative_coordinates_in_the_file_are_refused(tmp_path, column):
    path = edited(tmp_path, [record()], {column: "-1", "plane_id": "c/r/a.tif::z=0,t=0"})
    with pytest.raises(CsvFormatError, match=f"line 2: .*its {column} coordinate must be"):
        read_plane_histograms(path)
