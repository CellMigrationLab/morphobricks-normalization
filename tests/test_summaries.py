"""Per-image intensity summaries from plane_histograms.csv rows (contract §7.1, N50)."""

import gc
import re
import weakref

import numpy as np
import pytest

from morphobricks_normalization import (
    SINGLE,
    IncompatibleSummariesError,
    PlaneRecord,
    compute_plane_histogram,
    image_summaries,
    iter_image_summaries,
    read_plane_histograms,
    summary_columns,
    write_plane_histograms,
)
from simulation import cached_design, simulate_plane

CHANNEL_COLUMN = re.compile(r"^[A-Z][A-Za-z]*_[A-Z][A-Za-z]*_C(0|[1-9][0-9]*)$")


def record(path, planes, z=0, t=0, bin_count=None, labels=("control", "rep1", SINGLE)):
    """One XYC plane of ``path``; ``planes`` holds one array per channel."""
    histograms = [compute_plane_histogram(p, bin_count) for p in planes]
    return PlaneRecord(path, *labels, histograms=histograms, z=z, t=t, position_id="p")


def direct(pixels, low=1.0, high=99.8):
    """Contract §7.1 on the pixels: inverse-CDF percentiles, fraction at the type's maximum."""
    x = np.asarray(pixels)
    finite = x[np.isfinite(x)]
    saturated = None
    if x.dtype.kind in "iu":
        saturated = np.count_nonzero(finite == np.iinfo(x.dtype).max) / finite.size
    return (
        np.percentile(finite, 50, method="inverted_cdf"),
        np.percentile(finite, low, method="inverted_cdf"),
        np.percentile(finite, high, method="inverted_cdf"),
        saturated,
    )


def test_summaries_match_numpy_on_the_pixels_for_exact_histograms():
    planes, records = cached_design(seed=10, n_times=2)
    summaries = image_summaries(records)
    pixels = {}
    for labels, path, c, _, plane in planes:
        pixels.setdefault((labels, path), {}).setdefault(c, []).append(plane)
    assert len(summaries) == len(pixels) == 16
    for summary in summaries:
        labels = (summary.condition, summary.biological_repeat, summary.technical_repeat)
        channels = pixels[labels, summary.source_relative_path]
        assert summary.channel_count == len(channels) == 2
        for c, stack in channels.items():
            median, low, high, saturated = direct(np.stack(stack))
            assert summary.median[c] == median
            assert summary.p_low[c] == low
            assert summary.p_high[c] == high
            assert summary.saturated_fraction[c] == saturated == 0.0
            assert summary.exact[c] is True


def test_one_row_per_image_sorted_by_labels_with_z_and_t_pooled():
    rng = np.random.default_rng(1)
    planes = {}
    records = []
    for path, labels in (
        ("treated/rep1/b.tif", ("treated", "rep1", SINGLE)),
        ("control/rep2/a.tif", ("control", "rep2", SINGLE)),
        ("control/rep1/c.tif", ("control", "rep1", SINGLE)),
    ):
        for z in range(3):
            for t in range(2):
                plane = simulate_plane(rng, np.uint16, (12, 12), brightness=1 + z + t)
                planes.setdefault(path, []).append(plane)
                records.append(record(path, [plane], z=z, t=t, labels=labels))
    summaries = image_summaries(records[::-1])
    assert [s.source_relative_path for s in summaries] == [
        "control/rep1/c.tif",
        "control/rep2/a.tif",
        "treated/rep1/b.tif",
    ]
    for summary in summaries:
        median, low, high, _ = direct(np.stack(planes[summary.source_relative_path]))
        assert (summary.median, summary.p_low, summary.p_high) == ((median,), (low,), (high,))
        assert summary.image_file == summary.source_relative_path.rsplit("/", 1)[-1]


def test_other_percentiles():
    rng = np.random.default_rng(2)
    plane = simulate_plane(rng, np.uint16, (20, 20))
    (summary,) = image_summaries([record("a.tif", [plane])], percentile=(5, 95))
    _, low, high, _ = direct(plane, 5, 95)
    assert summary.percentile == (5.0, 95.0)
    assert (summary.p_low, summary.p_high) == ((low,), (high,))


@pytest.mark.parametrize("percentile", [(99.8, 1.0), (-1, 50), (1, 101), (50, 50)])
def test_invalid_percentiles_are_refused(percentile):
    with pytest.raises(ValueError, match="0 <= low < high <= 100"):
        image_summaries([], percentile=percentile)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int8, np.int16])
def test_saturated_fraction_counts_the_type_maximum(dtype):
    rng = np.random.default_rng(3)
    top = np.iinfo(dtype).max
    planes = []
    for k in range(3):
        plane = rng.integers(np.iinfo(dtype).min, top, size=(16, 16), endpoint=False).astype(dtype)
        plane.flat[: 7 * k] = top  # 0, 7 and 14 saturated pixels
        planes.append(plane)
    records = [record("a.tif", [p], z=z) for z, p in enumerate(planes)]
    (summary,) = image_summaries(records)
    assert summary.saturated_fraction == (21 / (3 * 256),)
    assert summary.saturated_fraction[0] == direct(np.stack(planes))[3]


def test_saturated_fraction_with_coarse_bins():
    rng = np.random.default_rng(4)
    # 64 bins over the values 65436 to 65535 (100 values): bins of one or two
    # values, the last one holding only 65535.
    alone = rng.integers(65436, 65535, size=(16, 16), endpoint=True).astype(np.uint16)
    alone.flat[:2] = (65436, 65535)
    histogram = compute_plane_histogram(alone, 64)
    assert histogram.integer_bin_limits()[-1] == (65535, 65535)
    assert not histogram.is_exact
    # 64 bins over 0 to 65535: the last bin holds 1024 values.
    shared = rng.integers(0, 60000, size=(16, 16)).astype(np.uint16)
    shared[0, :3] = 65535
    below = rng.integers(0, 60000, size=(16, 16)).astype(np.uint16)

    (summary,) = image_summaries([record("a.tif", [alone], bin_count=64)])
    assert summary.saturated_fraction == (np.count_nonzero(alone == 65535) / 256,)
    assert summary.exact == (False,)
    # The maximum shares its bin with other values: unknown, so blank.
    (summary,) = image_summaries([record("a.tif", [shared], bin_count=64)])
    assert summary.saturated_fraction == (None,)
    # Bins of many values, but no pixel at the type's maximum: none saturated.
    (summary,) = image_summaries([record("a.tif", [below], bin_count=64)])
    assert summary.saturated_fraction == (0.0,)
    # One plane with an unknown count makes the image's fraction unknown.
    records = [record("a.tif", [alone], bin_count=64), record("a.tif", [shared], z=1, bin_count=64)]
    (summary,) = image_summaries(records)
    assert summary.saturated_fraction == (None,)


def test_float_images_have_no_saturated_fraction():
    rng = np.random.default_rng(5)
    plane = simulate_plane(rng, np.float32, (20, 20))
    (summary,) = image_summaries([record("a.tif", [plane], bin_count=4096)])
    median, low, high, saturated = direct(plane)
    assert saturated is None and summary.saturated_fraction == (None,)
    assert summary.exact == (False,)
    bound = compute_plane_histogram(plane, 4096).max_representative_error
    for value, expected in ((summary.median, median), (summary.p_low, low), (summary.p_high, high)):
        # In float64: NumPy 2 would subtract a Python float from a float32 in float32.
        assert abs(value[0] - float(expected)) <= bound * (1 + 1e-9)


def test_a_channel_without_finite_pixels_is_blank():
    rng = np.random.default_rng(6)
    good = simulate_plane(rng, np.float32, (8, 8))
    empty = np.full((8, 8), np.nan, np.float32)
    (summary,) = image_summaries([record("a.tif", [good, empty], bin_count=100)])
    assert summary.median[0] is not None
    assert (summary.median[1], summary.p_low[1], summary.p_high[1]) == (None, None, None)
    assert summary.saturated_fraction == (None, None)
    row = summary.to_row()
    assert row["Intensity_MedianIntensity_C1"] is None


def test_different_channel_counts_are_refused():
    rng = np.random.default_rng(7)
    plane = simulate_plane(rng, np.uint16, (8, 8))
    records = [record("a.tif", [plane]), record("b.tif", [plane, plane])]
    with pytest.raises(ValueError, match="different numbers of channels.*a.tif.*b.tif"):
        image_summaries(records)


def test_rows_have_the_contract_columns():
    _, records = cached_design(seed=10, n_times=2)
    summaries = image_summaries(records)
    columns = summary_columns(2)
    assert columns == [
        "source_relative_path",
        "condition",
        "biological_repeat",
        "technical_repeat",
        "image_file",
        "Intensity_MedianIntensity_C0",
        "Intensity_LowPercentile_C0",
        "Intensity_HighPercentile_C0",
        "Intensity_SaturatedFraction_C0",
        "Histogram_Exact_C0",
        "Intensity_MedianIntensity_C1",
        "Intensity_LowPercentile_C1",
        "Intensity_HighPercentile_C1",
        "Intensity_SaturatedFraction_C1",
        "Histogram_Exact_C1",
    ]
    assert all(CHANNEL_COLUMN.match(name) for name in columns[5:])
    summary = summaries[0]
    row = summary.to_row()
    assert list(row) == columns
    assert row["image_file"] == summary.image_file
    assert row["Intensity_MedianIntensity_C1"] == summary.median[1]
    assert row["Intensity_HighPercentile_C0"] == summary.p_high[0]
    assert row["Histogram_Exact_C1"] is True


def test_summaries_from_the_written_table_equal_those_from_the_records(tmp_path):
    _, records = cached_design(seed=10, n_times=2)
    path = write_plane_histograms(records, tmp_path / "plane_histograms.csv")
    assert image_summaries(read_plane_histograms(path)) == image_summaries(records)


def test_no_records_no_summaries():
    assert image_summaries([]) == []


@pytest.mark.parametrize("first_is_small", [True, False])
@pytest.mark.parametrize(
    ("small", "large"), [(np.uint8, np.uint16), (np.uint16, np.int16), (np.uint8, np.int8)]
)
def test_planes_of_one_image_with_different_pixel_types_are_refused(small, large, first_is_small):
    """Issue #54: no pixel is at either type's maximum, yet the first dtype used to decide."""
    a = record("a.tif", [np.full((8, 8), 100, dtype=small)], z=0)
    b = record("a.tif", [np.full((8, 8), 100, dtype=large)], z=1)
    with pytest.raises(IncompatibleSummariesError, match=r"a\.tif.*different pixel types") as info:
        image_summaries([a, b] if first_is_small else [b, a])
    assert str(np.dtype(small)) in str(info.value) and str(np.dtype(large)) in str(info.value)


def test_same_pixel_type_with_varying_planes_is_accepted():
    records = [record("a.tif", [np.full((8, 8), 10 * (z + 1), np.uint16)], z=z) for z in range(3)]
    (summary,) = image_summaries(records)
    assert summary.saturated_fraction == (0.0,)


def test_different_images_may_differ_in_pixel_type():
    records = [
        record("a.tif", [np.full((8, 8), 5, np.uint8)]),
        record("b.tif", [np.full((8, 8), 5, np.uint16)]),
    ]
    assert [s.dtype for s in image_summaries(records)] == [np.uint8, np.uint16]


# --- streaming API (issue #58) -------------------------------------------------------------


def grouped_records(n_images=4, planes_per_image=3, channels=1, seed=20):
    """Rows grouped by image and sorted as image_summaries sorts its output."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_images):
        labels = ("control", "rep1", SINGLE)
        for z in range(planes_per_image):
            planes = [simulate_plane(rng, np.uint16, (10, 10)) for _ in range(channels)]
            rows.append(record(f"img{i}.tif", planes, z=z, labels=labels))
    return rows


def test_streaming_summaries_equal_the_eager_ones():
    rows = grouped_records(channels=2)
    assert list(iter_image_summaries(rows)) == image_summaries(rows)
    assert list(iter_image_summaries(rows, (5, 95))) == image_summaries(rows, (5, 95))


def test_streaming_emits_one_image_at_a_time_and_drops_its_histograms():
    rows = grouped_records(n_images=3)
    pulled = []
    first_hist = weakref.ref(rows[0].histograms[0])

    def source():
        for row in rows:
            pulled.append(row.plane_id)
            yield row

    stream = iter_image_summaries(source())
    next(stream)
    # The first summary is out once the second image has started, not after all rows.
    assert len(pulled) == 4 < len(rows)
    del rows[:3]  # the caller's own list drops image 0; the stream must not keep it
    gc.collect()
    next(stream)
    gc.collect()
    assert first_hist() is None


def test_streaming_refuses_rows_that_are_not_together():
    rows = grouped_records(n_images=2)
    mixed = [rows[0], rows[3], rows[1], rows[4], rows[2], rows[5]]
    with pytest.raises(ValueError, match="not together"):
        list(iter_image_summaries(mixed))
    # The eager API still accepts any order.
    assert len(image_summaries(mixed)) == 2


def test_streaming_refuses_a_file_under_two_label_sets():
    plane = simulate_plane(np.random.default_rng(1), np.uint16, (8, 8))
    rows = [
        record("a.tif", [plane], z=0, labels=("control", "rep1", SINGLE)),
        record("a.tif", [plane], z=1, labels=("treated", "rep1", SINGLE)),
    ]
    with pytest.raises(ValueError, match="different condition or repeat labels"):
        list(iter_image_summaries(rows))


def test_streaming_refuses_repeated_planes_and_different_channel_counts():
    rows = grouped_records(n_images=1)
    with pytest.raises(ValueError, match="more than once"):
        list(iter_image_summaries([*rows, rows[0]]))
    one = grouped_records(n_images=1, channels=1)
    two = [
        r for r in grouped_records(n_images=2, channels=2) if r.source_relative_path == "img1.tif"
    ]
    with pytest.raises(ValueError, match="different numbers of channels"):
        list(iter_image_summaries([*one, *two]))


def test_streaming_validates_percentiles_at_the_call():
    with pytest.raises(ValueError, match="0 <= low < high <= 100"):
        iter_image_summaries([], percentile=(99.8, 1.0))


def test_streaming_no_records_no_summaries():
    assert list(iter_image_summaries([])) == []
