"""Robust histogram range, underflow and overflow, and the range check (Milestone 5, issue #12).

Contract §1.4, decisions N54–N60. Quantiles are compared with
``np.quantile(..., method="inverted_cdf")``, the library's percentile rule.
Means and standard deviations come from sums, so they are compared within a
relative tolerance, never bit for bit.
"""

import csv

import numpy as np
import pytest

from morphobricks_normalization import (
    HistogramSettings,
    PlaneRecord,
    PooledHistogram,
    RangeCheck,
    WeightedHistogram,
    compute_block_histogram,
    compute_plane_histogram,
    fit,
    histogram_with_range_check,
    image_summaries,
    percentile_range,
    range_check_issues,
    read_plane_histograms,
    write_plane_histograms,
)
from morphobricks_normalization.csv_io import CsvFormatError
from morphobricks_normalization.histogram import estimate_histogram_memory
from morphobricks_normalization.settings import NormalizationSettings

ROBUST = HistogramSettings(histogram_range="robust")
ZERO_SPREAD = "intensity spread is zero"


def whole(plane):
    return lambda: (plane,)


def in_blocks(plane, rows=37, columns=53):
    """Rectangles of the plane, as a reader of tiles would give them."""

    def blocks():
        for top in range(0, plane.shape[0], rows):
            for left in range(0, plane.shape[1], columns):
                yield plane[top : top + rows, left : left + columns]

    return blocks


def counting(blocks):
    """Wrap ``blocks`` and count how often the plane is read."""
    calls = []

    def wrapped():
        calls.append(1)
        return blocks()

    return wrapped, calls


def quantile(pixels, q):
    finite = pixels[np.isfinite(pixels)] if pixels.dtype.kind == "f" else pixels.ravel()
    return np.quantile(finite, q / 100, method="inverted_cdf").item()


def issue_planes(seed=0):
    """The reproduction of issue #12: a normal float32 plane, and the same with one hot pixel."""
    rng = np.random.default_rng(seed)
    clean = rng.normal(100, 10, (512, 512)).astype(np.float32)
    hot = clean.copy()
    hot[0, 0] = 1e6
    return clean, hot


def assert_same_histogram(a, b, sums_exact=True):
    assert a.dtype == b.dtype and a.bin_count == b.bin_count
    assert (a.binning_min, a.binning_max) == (b.binning_min, b.binning_max)
    assert (a.pixel_min, a.pixel_max) == (b.pixel_min, b.pixel_max)
    assert (a.total_pixel_count, a.valid_pixel_count) == (b.total_pixel_count, b.valid_pixel_count)
    np.testing.assert_array_equal(a.counts, b.counts)
    for ta, tb in ((a.underflow, b.underflow), (a.overflow, b.overflow)):
        assert ta.count == tb.count
        if sums_exact:
            assert (ta.sum, ta.sum_of_squares) == (tb.sum, tb.sum_of_squares)
        else:
            np.testing.assert_allclose(
                [ta.sum, ta.sum_of_squares], [tb.sum, tb.sum_of_squares], rtol=1e-12
            )


# --- The reproduction of issue #12 -------------------------------------------------------------


def test_minmax_keeps_todays_result_and_the_check_flags_it():
    _, hot = issue_planes()
    histogram, check = histogram_with_range_check(whole(hot), hot.dtype)
    # The default range is unchanged: the same histogram as without settings...
    assert_same_histogram(histogram, compute_plane_histogram(hot))
    assert (histogram.binning_min, histogram.binning_max) == (float(hot.min()), 1e6)
    assert not histogram.has_tails
    # ...and the same zero-spread error (fit gives no new warning, N57).
    with pytest.raises(ValueError, match=ZERO_SPREAD):
        percentile_range(PooledHistogram([histogram]), 1, 99.8)
    record = PlaneRecord("c/img.tif", "c", "r", "t", histogram)
    with pytest.raises(ValueError, match=ZERO_SPREAD):
        fit([record], NormalizationSettings())
    # The check sees the collapsed central range and what the robust range would give.
    assert check.histogram_range == "minmax" and check.narrow
    assert check.central_bins == 1
    assert check.robust_limits == (quantile(hot, 0.01), quantile(hot, 99.99))
    assert check.outside_robust_count == 52  # 26 pixels below and 26 above
    assert check.robust_central_bins > 5000
    (issue,) = range_check_issues([(record.plane_id, 0, check)])
    assert issue.code == "narrow_histogram_range" and issue.channel == 0
    assert issue.items == ("c/img.tif::z=0,t=0",)
    assert "fewer than 100 of the histogram's bins (as few as 1)" in issue.message
    assert "robust histogram range" in issue.action


def test_robust_range_gives_percentiles_close_to_the_clean_planes():
    clean, hot = issue_planes()
    expected = percentile_range(PooledHistogram([compute_plane_histogram(clean)]), 1, 99.8)
    histogram = compute_plane_histogram(hot, settings=ROBUST)
    assert (histogram.binning_min, histogram.binning_max) == (
        quantile(hot, 0.01),
        quantile(hot, 99.99),
    )
    # The exact extrema stay recorded; the outliers are in the tails.
    assert (histogram.pixel_min, histogram.pixel_max) == (float(hot.min()), 1e6)
    assert (histogram.underflow.count, histogram.overflow.count) == (26, 26)
    assert histogram.counts.sum() + 52 == histogram.valid_pixel_count == hot.size
    pooled = PooledHistogram([histogram])
    parameters = percentile_range(pooled, 1, 99.8)
    # Bins of (137.5 - 62.6) / 10,000: the clean plane's error bound is about 0.005.
    assert parameters.offset == pytest.approx(expected.offset, abs=0.01)
    assert parameters.scale == pytest.approx(expected.scale, abs=0.02)
    exact = np.quantile(hot, [0.01, 0.998], method="inverted_cdf").astype(np.float64)
    assert parameters.offset == pytest.approx(exact[0], abs=0.01)
    # Pixels in the tails are represented by their mean, so the bound covers them.
    assert pooled.max_representative_error > 1e5
    record = PlaneRecord("c/img.tif", "c", "r", "t", histogram)
    parameters = fit([record], NormalizationSettings())
    assert not parameters.exact
    assert [i.code for i in parameters.issues] == ["approximate_parameters"]
    # Min-max parameters keep using the exact extrema.
    minmax = fit([record], NormalizationSettings(method="minmax"))
    assert minmax.offset == (float(hot.min()),) and minmax.scale == (1e6 - float(hot.min()),)


# --- Exact histograms never change -------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int8, np.int16])
def test_exact_histograms_are_identical_in_both_modes(dtype):
    rng = np.random.default_rng(3)
    info = np.iinfo(dtype)
    plane = rng.integers(info.min, info.max, (64, 64), endpoint=True).astype(dtype)
    plane[0, 0] = info.max  # a "hot" pixel at the type's maximum
    default = compute_plane_histogram(plane)
    robust, check = histogram_with_range_check(whole(plane), dtype, settings=ROBUST)
    assert default.is_exact and robust.is_exact and check is None
    assert_same_histogram(default, robust)
    assert not robust.has_tails


def test_integer_planes_with_fewer_values_than_bins_are_exact_in_both_modes():
    plane = np.arange(-500, 500, dtype=np.int64).reshape(10, 100)
    default = compute_plane_histogram(plane, 1000)
    robust, check = histogram_with_range_check(whole(plane), plane.dtype, 1000, settings=ROBUST)
    assert robust.is_exact and check is None
    assert_same_histogram(default, robust)


def test_constant_float_planes_are_not_checked():
    plane = np.full((8, 8), 3.5, np.float32)
    histogram, check = histogram_with_range_check(whole(plane), plane.dtype, settings=ROBUST)
    assert check is None and histogram.counts[0] == 64 and not histogram.has_tails


# --- Quantiles and blocks ---------------------------------------------------------------------


def mixed_plane(dtype, rng):
    """Values of both signs with a few extreme pixels at both ends."""
    if np.dtype(dtype).kind == "f":
        plane = rng.normal(-3, 50, (160, 101)).astype(dtype)
        plane.ravel()[:5] = [-1e30, 1e30, np.nan, np.inf, -np.inf]
        return plane
    info = np.iinfo(dtype)
    centre = 0 if info.min < 0 else 2**20
    plane = (centre + rng.normal(0, 3000, (160, 101))).astype(dtype)
    plane.ravel()[:4] = [info.min, info.max, info.min + 1, info.max - 1]
    return plane


DTYPES = [np.float32, np.float64, np.int32, np.uint32, np.int64, np.uint64]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quantiles", [(0.01, 99.99), (2.5, 97.5), (0.0, 100.0), (50.0, 50.5)])
def test_robust_limits_are_exact_inverse_cdf_quantiles(dtype, quantiles):
    plane = mixed_plane(dtype, np.random.default_rng(11))
    settings = HistogramSettings("robust", quantiles)
    histogram, check = histogram_with_range_check(whole(plane), dtype, settings=settings)
    limits = tuple(quantile(plane, q) for q in quantiles)
    assert (histogram.binning_min, histogram.binning_max) == limits == check.robust_limits
    finite = plane[np.isfinite(plane)] if plane.dtype.kind == "f" else plane.ravel()
    below, above = finite[finite < limits[0]], finite[finite > limits[1]]
    assert (histogram.underflow.count, histogram.overflow.count) == (below.size, above.size)
    assert check.outside_robust_count == below.size + above.size
    if below.size:
        assert histogram.underflow.mean == pytest.approx(below.astype(np.float64).mean())
    assert histogram.counts.sum() + below.size + above.size == finite.size
    assert histogram.valid_pixel_count == finite.size
    assert (histogram.pixel_min, histogram.pixel_max) == (finite.min().item(), finite.max().item())


@pytest.mark.parametrize("dtype", DTYPES)
def test_blockwise_equals_whole_in_minmax_mode(dtype):
    plane = mixed_plane(dtype, np.random.default_rng(5))
    expected, check = histogram_with_range_check(whole(plane), dtype)
    for blocks in (in_blocks(plane), in_blocks(plane, 1, plane.shape[1]), in_blocks(plane, 160, 7)):
        histogram, block_check = histogram_with_range_check(blocks, dtype)
        assert_same_histogram(histogram, expected)
        assert block_check == check
        assert_same_histogram(compute_block_histogram(blocks, dtype), expected)


@pytest.mark.parametrize("dtype", DTYPES)
def test_blockwise_gives_the_same_robust_limits_and_counts(dtype):
    # Only the tails' float sums may differ in the last bits (summation order).
    plane = mixed_plane(dtype, np.random.default_rng(6))
    expected, check = histogram_with_range_check(whole(plane), dtype, settings=ROBUST)
    histogram, block_check = histogram_with_range_check(in_blocks(plane), dtype, settings=ROBUST)
    assert_same_histogram(histogram, expected, sums_exact=False)
    assert block_check == check


@pytest.mark.parametrize(
    "dtype, settings, check, reads",
    [
        (np.float32, None, False, 2),
        (np.float32, None, True, 2),  # the check shares the binning pass
        (np.float32, ROBUST, False, 3),
        (np.int32, None, True, 2),
        (np.int32, ROBUST, True, 3),
        (np.float64, None, True, 4),
        (np.float64, ROBUST, True, 5),
        (np.uint64, ROBUST, False, 5),
        (np.uint16, ROBUST, True, 1),
    ],
)
def test_passes_over_the_blocks(dtype, settings, check, reads):
    if dtype == np.uint16:
        plane = np.random.default_rng(1).integers(0, 60000, (32, 32)).astype(np.uint16)
    else:
        plane = mixed_plane(dtype, np.random.default_rng(1))
    blocks, calls = counting(in_blocks(plane))
    if check:
        histogram_with_range_check(
            blocks, dtype, 1000 if dtype == np.uint16 else None, settings=settings
        )
    else:
        compute_block_histogram(blocks, dtype, settings=settings)
    assert len(calls) == reads


def test_small_integers_with_fewer_bins_than_values_use_the_robust_range():
    rng = np.random.default_rng(8)
    plane = rng.integers(1000, 3000, (256, 256)).astype(np.uint16)
    plane[:2, :3] = 65535
    histogram, check = histogram_with_range_check(whole(plane), plane.dtype, 256, settings=ROBUST)
    assert (histogram.binning_min, histogram.binning_max) == (
        quantile(plane, 0.01),
        quantile(plane, 99.99),
    )
    assert histogram.overflow.count >= 6 and histogram.pixel_max == 65535
    assert histogram.overflow.sum == float(plane[plane > histogram.binning_max].sum())
    minmax, minmax_check = histogram_with_range_check(in_blocks(plane), plane.dtype, 256)
    assert_same_histogram(minmax, compute_plane_histogram(plane, 256))
    assert minmax_check.robust_limits == check.robust_limits
    assert minmax_check.outside_robust_count == check.outside_robust_count


def test_central_bins_match_the_histograms_percentile_bins():
    rng = np.random.default_rng(9)
    plane = rng.gamma(2.0, 50.0, (200, 300)).astype(np.float32)
    plane[5, 5] = 1e5
    histogram, check = histogram_with_range_check(whole(plane), plane.dtype)
    cumulative = np.cumsum(histogram.counts)
    n = histogram.valid_pixel_count
    first, last = (
        int(np.searchsorted(cumulative, max(1, int(np.ceil(q / 100 * n))))) for q in (1, 99.8)
    )
    assert check.central_bins == last - first + 1
    robust = compute_plane_histogram(plane, settings=ROBUST)
    cumulative = np.cumsum(robust.counts)
    below = robust.underflow.count
    first, last = (
        int(np.searchsorted(cumulative, max(1, int(np.ceil(q / 100 * n))) - below))
        for q in (1, 99.8)
    )
    assert check.robust_central_bins == last - first + 1


def test_nan_and_infinite_pixels_are_left_out():
    plane = np.random.default_rng(2).normal(0, 1, (50, 50))
    plane[0, :10] = np.nan
    plane[1, :3] = np.inf
    histogram = compute_plane_histogram(plane, settings=ROBUST)
    assert (histogram.valid_pixel_count, histogram.non_finite_pixel_count) == (2487, 13)
    assert histogram.binning_min == quantile(plane, 0.01)
    assert np.isfinite([histogram.overflow.sum, histogram.underflow.sum]).all()


def test_coinciding_robust_limits():
    # Nearly every pixel is 0: both robust limits are 0, the rest are tails.
    plane = np.zeros((200, 200), np.float32)
    plane[0, 0], plane[0, 1] = -5.0, 7.0
    histogram = compute_plane_histogram(plane, settings=ROBUST)
    assert (histogram.binning_min, histogram.binning_max) == (0.0, 0.0)
    assert histogram.counts[0] == 39998 and histogram.counts[1:].sum() == 0
    assert (histogram.underflow.sum, histogram.overflow.sum) == (-5.0, 7.0)
    pooled = PooledHistogram([histogram])
    assert pooled.percentile(0.001) == -5.0 and pooled.percentile(100) == 7.0
    assert pooled.median == 0.0


# --- Mean and standard deviation from the sums -------------------------------------------------


def wide_integer_planes(rng, n_planes=3):
    """int32 planes of 0..999 with far outliers; inside the robust limits the bins are exact."""
    planes = []
    for _ in range(n_planes):
        plane = rng.integers(0, 1000, (200, 200)).astype(np.int32)
        plane.ravel()[rng.choice(plane.size, 4, replace=False)] = [
            -(2**31),
            -(10**9),
            2**31 - 1,
            10**9,
        ]
        planes.append(plane)
    return planes


def test_the_sums_make_mean_and_std_exact():
    planes = wide_integer_planes(np.random.default_rng(4))
    histograms = [compute_plane_histogram(p, 1000, settings=ROBUST) for p in planes]
    assert all(h.has_tails for h in histograms)
    assert all(h.binning_max - h.binning_min + 1 <= 1000 for h in histograms)  # exact bins
    pooled = PooledHistogram(histograms)
    pixels = np.concatenate([p.ravel() for p in planes]).astype(np.float64)
    assert pooled.mean == pytest.approx(pixels.mean(), rel=1e-12)
    assert pooled.std == pytest.approx(pixels.std(), rel=1e-12)
    # Representing the tails by their means would be far off for the standard deviation.
    values = np.repeat(pooled.values, pooled.counts)
    assert abs(values.std() - pixels.std()) > 1e-3 * pixels.std()
    # Minimum and maximum stay exact; percentiles are exact inside the robust limits.
    assert (pooled.minimum, pooled.maximum) == (pixels.min(), pixels.max())
    assert pooled.percentile(50) == np.quantile(pixels, 0.5, method="inverted_cdf")
    assert not pooled.max_representative_error == 0


def test_weighted_pooling_uses_the_sums_too():
    rng = np.random.default_rng(12)
    planes = wide_integer_planes(rng, 2)
    parts = [PooledHistogram([compute_plane_histogram(p, 1000, settings=ROBUST)]) for p in planes]
    weights = [1.0, 2.5]
    weighted = WeightedHistogram(parts, weights)
    pixels = np.concatenate([p.ravel() for p in planes]).astype(np.float64)
    w = np.repeat(weights, [p.size for p in planes])
    mean = np.average(pixels, weights=w)
    assert weighted.mean == pytest.approx(mean, rel=1e-12)
    assert weighted.std == pytest.approx(
        np.sqrt(np.average((pixels - mean) ** 2, weights=w)), rel=1e-12
    )
    # With weight 1 everywhere it is the pooled histogram's mean and std.
    plain = WeightedHistogram(parts, [1.0, 1.0])
    pooled = PooledHistogram([compute_plane_histogram(p, 1000, settings=ROBUST) for p in planes])
    assert plain.mean == pytest.approx(pooled.mean, rel=1e-14)
    assert plain.std == pytest.approx(pooled.std, rel=1e-14)


def test_zscore_from_a_robust_table_uses_the_exact_sums():
    planes = wide_integer_planes(np.random.default_rng(13), 2)
    records = [
        PlaneRecord(f"c/{i}.tif", "c", "r", "t", compute_plane_histogram(p, 1000, settings=ROBUST))
        for i, p in enumerate(planes)
    ]
    parameters = fit(records, NormalizationSettings(method="zscore", scope="biological_repeat"))
    pixels = np.concatenate([p.ravel() for p in planes]).astype(np.float64)
    assert parameters.offset[0] == pytest.approx(pixels.mean(), rel=1e-12)
    assert parameters.scale[0] == pytest.approx(pixels.std(), rel=1e-12)
    assert not parameters.exact


# --- Statistics file, schema version 4 ---------------------------------------------------------


def robust_records():
    rng = np.random.default_rng(21)
    float_planes = [mixed_plane(np.float32, rng) for _ in range(2)]
    int_planes = wide_integer_planes(rng, 2)
    return [
        PlaneRecord(
            f"c/f{i}.tif",
            "c",
            "r",
            "t",
            [compute_plane_histogram(p, 500, settings=ROBUST), compute_plane_histogram(p, 500)],
            t=i,
        )
        for i, p in enumerate(float_planes)
    ], [
        PlaneRecord(f"c/i{i}.tif", "c", "r", "t", compute_plane_histogram(p, settings=ROBUST))
        for i, p in enumerate(int_planes)
    ]


def test_schema_4_round_trip(tmp_path):
    for k, records in enumerate(robust_records()):
        path = write_plane_histograms(records, tmp_path / f"t{k}.csv")
        back = read_plane_histograms(path)
        assert len(back) == len(records)
        for a, b in zip(records, back, strict=True):
            assert a.plane_id == b.plane_id
            for ha, hb in zip(a.histograms, b.histograms, strict=True):
                assert_same_histogram(ha, hb)  # floats written in full, read back unchanged
                assert ha.max_representative_error == hb.max_representative_error
        with open(path, newline="", encoding="utf-8") as fh:
            row = next(csv.DictReader(fh))
        assert row["schema_version"] == "4"
        assert int(row["Histogram_UnderflowCount_C0"]) == records[0].histograms[0].underflow.count
        assert float(row["Histogram_OverflowSum_C0"]) == records[0].histograms[0].overflow.sum


def as_version_3(path):
    """Rewrite a version-4 table as version 3: no tail columns, schema_version 3."""
    with open(path, newline="", encoding="utf-8") as fh:
        header, *rows = list(csv.reader(fh))
    keep = [i for i, name in enumerate(header) if "flow" not in name]
    version = header.index("schema_version")
    for row in rows:
        row[version] = "3"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([[r[i] for i in keep] for r in [header, *rows]])
    return path


@pytest.mark.parametrize("with_rows", [True, False])
def test_version_3_is_refused(tmp_path, with_rows):
    records = [PlaneRecord("c/a.tif", "c", "r", "t", compute_plane_histogram(np.arange(6)))]
    path = as_version_3(write_plane_histograms(records, tmp_path / "plane_histograms.csv"))
    if not with_rows:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text(lines[0], encoding="utf-8")
    with pytest.raises(CsvFormatError) as error:
        read_plane_histograms(path)
    assert str(error.value) == (
        f"{path}: This plane_histograms.csv was written with schema version 3, without the "
        "underflow and overflow columns. Run the preprocessing brick again on this experiment: "
        "it rewrites the table with these columns (schema version 4)."
    )


def test_a_row_of_version_3_in_a_version_4_table_is_refused(tmp_path):
    records = [PlaneRecord("c/a.tif", "c", "r", "t", compute_plane_histogram(np.arange(6)))]
    path = write_plane_histograms(records, tmp_path / "t.csv")
    text = path.read_text(encoding="utf-8").replace(",4,", ",3,", 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(CsvFormatError, match="schema version 3, without the underflow"):
        read_plane_histograms(path)


@pytest.mark.parametrize(
    "column, value",
    [
        ("Histogram_UnderflowCount_C0", "0"),  # counts no longer add up
        ("Histogram_BinningMin_C0", None),  # limit at the extremum, yet a tail below it
        ("Histogram_OverflowSum_C1", "3.0"),  # sums of an empty tail
    ],
)
def test_inconsistent_tails_are_rejected(tmp_path, column, value):
    float_records, _ = robust_records()
    path = write_plane_histograms(float_records, tmp_path / "t.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        header, *rows = list(csv.reader(fh))
    if value is None:
        value = rows[0][header.index("Intensity_MinIntensity_C0")]
    rows[0][header.index(column)] = value
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([header, *rows])
    with pytest.raises(CsvFormatError, match="line 2"):
        read_plane_histograms(path)


# --- Settings, warnings, summaries -------------------------------------------------------------


def test_settings_round_trip_and_validation():
    settings = HistogramSettings("robust", [0.1, 99.9], 50)
    data = settings.to_dict()
    assert data["histogram_range"] == "robust" and data["robust_quantiles"] == [0.1, 99.9]
    assert HistogramSettings.from_dict(data) == settings
    assert HistogramSettings.from_dict({}) == HistogramSettings()
    assert HistogramSettings().robust_quantiles == (0.01, 99.99)
    with pytest.raises(ValueError, match="Choose 'minmax'"):
        HistogramSettings("quantile")
    with pytest.raises(ValueError, match="0 <= low < high <= 100"):
        HistogramSettings("robust", (99.0, 1.0))
    with pytest.raises(ValueError, match="two numbers"):
        HistogramSettings("robust", "1,99")
    with pytest.raises(ValueError, match="at least 1"):
        HistogramSettings(min_central_bins=0)
    with pytest.raises(ValueError, match="Unknown histogram setting"):
        HistogramSettings.from_dict({"range": "robust"})


def test_range_check_issues_one_per_channel():
    def check(bins, outside=3):
        return RangeCheck("minmax", (1.0, 99.8), bins, (0.01, 99.99), (0, 1), outside, 900, 100)

    issues = range_check_issues(
        [
            ("a::z=0,t=0", 1, check(5)),
            ("a::z=0,t=0", 0, check(500)),  # wide enough
            ("b::z=0,t=0", 1, check(20, outside=4)),
            ("b::z=0,t=0", 0, None),  # exact histogram
            ("c::z=0,t=0", 0, check(99)),
        ]
    )
    assert [(i.channel, i.items) for i in issues] == [
        (0, ("c::z=0,t=0",)),
        (1, ("a::z=0,t=0", "b::z=0,t=0")),
    ]
    assert "In 2 plane(s)" in issues[1].message and "as few as 5" in issues[1].message
    assert "7 pixel(s)" in issues[1].message and "at least 900 bins" in issues[1].message
    assert range_check_issues([("a", 0, check(100))]) == []


def test_saturated_fraction_is_blank_when_the_maximum_is_in_the_overflow():
    rng = np.random.default_rng(14)
    plane = rng.integers(0, 10**6, (256, 256)).astype(np.uint32)
    plane[0, :3] = np.iinfo(np.uint32).max
    histogram = compute_plane_histogram(plane, 1000, settings=ROBUST)
    assert histogram.overflow.count >= 3
    (summary,) = image_summaries([PlaneRecord("c/a.tif", "c", "r", "t", histogram)])
    assert summary.saturated_fraction == (None,)
    assert summary.exact == (False,)


def test_memory_estimate_covers_the_quantile_search():
    base = estimate_histogram_memory(10**6, np.float32)
    assert estimate_histogram_memory(10**6, np.float32, settings=HistogramSettings()) == base
    assert estimate_histogram_memory(10**6, np.float32, range_check=True) > base
    assert estimate_histogram_memory(10**6, np.float32, settings=ROBUST) > base
    small = estimate_histogram_memory(10**6, np.uint16)
    assert estimate_histogram_memory(10**6, np.uint16, settings=ROBUST, range_check=True) == small
