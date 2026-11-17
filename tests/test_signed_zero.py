"""Issue #105: every float the package writes as text writes ``-0.0`` as ``0.0``.

A float plane holding both zeros gets a ``pixel_min`` or ``pixel_max`` of ``-0.0`` or
``+0.0`` depending on NumPy's ``min``/``max``, which depends on the element order and the
platform. Owner decision (17 November 2026): the written files canonicalise signed zero to
``0.0``, so the same pixels write the same bytes on every platform. Nothing else changes: the
in-memory values keep their sign, and only zeros are touched.
"""

import csv
import io
import itertools
import math

import numpy as np
import pytest

from morphobricks_normalization import (
    NormalizationSettings,
    PlaneRecord,
    compute_plane_histogram,
    fit,
    read_plane_histograms,
    write_plane_histograms,
)
from morphobricks_normalization.float_text import canonical_zero, float_text, positive_zero
from morphobricks_normalization.histogram import HistogramTail, PlaneHistogram
from morphobricks_normalization.normalization import percentile_name
from morphobricks_normalization.summaries import ImageSummary

FLOAT_COLUMNS = (
    "Intensity_MinIntensity_DNA",
    "Intensity_MaxIntensity_DNA",
    "Histogram_BinningMin_DNA",
    "Histogram_BinningMax_DNA",
)


def _record(histogram):
    return PlaneRecord(
        "c/b/t/i.tif", "c", "b", "t", histograms=[histogram], channels=("DNA",), position_id="p"
    )


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_float_text_canonicalises_only_the_zero_sign():
    assert float_text(-0.0) == "0.0"
    assert float_text(0.0) == "0.0"
    assert math.copysign(1.0, canonical_zero(-0.0)) == 1.0
    for value in (1.5, -1.5, 5e-324, -5e-324, 1e308, -1e308, math.inf, -math.inf, 0.1 + 0.2):
        assert float_text(value) == repr(value)
        assert canonical_zero(value) == value
    assert float_text(math.nan) == "nan"
    assert float_text(np.float64(-0.0)) == "0.0"
    assert float_text(np.float32(-0.0)) == "0.0"


def test_a_computed_negative_zero_plane_writes_0_0(tmp_path):
    histogram = compute_plane_histogram(np.full((4, 4), -0.0, dtype=np.float32))
    # The computed values keep their sign: no numeric algorithm changes.
    assert math.copysign(1.0, histogram.pixel_min) == -1.0
    assert math.copysign(1.0, histogram.binning_max) == -1.0
    path = write_plane_histograms([_record(histogram)], tmp_path / "t.csv", channel_ids=["DNA"])
    text = path.read_text(encoding="utf-8")
    assert "-0.0" not in text
    (row,) = _rows(path)
    assert [row[column] for column in FLOAT_COLUMNS] == ["0.0"] * 4
    (back,) = read_plane_histograms(path)
    assert back.histograms[0].pixel_min == 0.0


@pytest.mark.parametrize(
    "pixels",
    [list(order) for order in itertools.permutations([-0.0, 0.0, -0.0, 0.0])]
    + [[-0.0, 0.0, 1.0, -1.0], [1.0, 0.0, -0.0, -1.0], [-1.0, -0.0, 0.0]],
)
def test_planes_mixing_signed_zeros_write_the_same_bytes_in_every_order(tmp_path, pixels):
    expected = sorted(pixels, key=lambda value: (value, math.copysign(1.0, value)))
    files = []
    for k, order in enumerate((pixels, expected, expected[::-1])):
        plane = np.asarray(order, dtype=np.float32).reshape(1, -1)
        record = _record(compute_plane_histogram(plane, 4))
        files.append(write_plane_histograms([record], tmp_path / f"{k}.csv", channel_ids=["DNA"]))
    data = [path.read_bytes() for path in files]
    assert data[0] == data[1] == data[2]
    assert b"-0.0," not in data[0]


def test_tail_mean_and_std_write_0_0(tmp_path):
    base = compute_plane_histogram(np.array([[1.0, 2.0]], dtype=np.float32), 2)
    tail = HistogramTail(1, -0.0, -0.0)
    histogram = PlaneHistogram(
        dtype=base.dtype,
        bin_count=base.bin_count,
        binning_min=base.binning_min,
        binning_max=base.binning_max,
        counts=base.counts,
        total_pixel_count=base.total_pixel_count + 1,
        valid_pixel_count=base.valid_pixel_count + 1,
        pixel_min=-0.0,
        pixel_max=base.pixel_max,
        underflow=tail,
        overflow=base.overflow,
    )
    path = write_plane_histograms([_record(histogram)], tmp_path / "t.csv", channel_ids=["DNA"])
    (row,) = _rows(path)
    assert row["Intensity_MinIntensity_DNA"] == "0.0"
    assert row["Histogram_UnderflowMean_DNA"] == "0.0"
    assert row["Histogram_UnderflowStd_DNA"] == "0.0"
    assert "-0.0" not in path.read_text(encoding="utf-8")


def test_parameter_columns_write_0_0():
    pixels = np.array([[-0.0, 1.0], [-0.0, 2.0]], dtype=np.float32)
    settings = NormalizationSettings(method="minmax", scope="image")
    parameters = fit(pixels, settings, channels=["DNA"])
    assert math.copysign(1.0, parameters.offset[0]) == -1.0  # fitted value unchanged
    columns = parameters.to_columns()
    assert columns["Normalization_Offset_DNA"] == "0.0"
    assert columns["Normalization_Scale_DNA"] == "2.0"
    assert type(parameters).from_columns(columns, settings) == parameters


SUMMARY_VALUE_COLUMNS = (
    "Intensity_MedianIntensity_DNA",
    "Intensity_LowPercentile_DNA",
    "Intensity_HighPercentile_DNA",
    "Intensity_SaturatedFraction_DNA",
)


def _summary(median, p_low, p_high, saturated):
    return ImageSummary(
        source_relative_path="c/b/t/i.tif",
        condition="c",
        biological_repeat="b",
        technical_repeat="t",
        dtype=np.dtype(np.float32),
        percentile=(1.0, 99.8),
        channels=("DNA",),
        median=(median,),
        p_low=(p_low,),
        p_high=(p_high,),
        saturated_fraction=(saturated,),
        exact=(True,),
    )


def _written(row):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(row))
    writer.writerow(row)
    return out.getvalue()


@pytest.mark.parametrize("zero", [-0.0, np.float64(-0.0), np.float32(-0.0), 0.0, np.float32(0.0)])
def test_summary_rows_hold_0_0(zero):
    row = _summary(zero, zero, zero, None).to_row()
    for column in SUMMARY_VALUE_COLUMNS[:3]:
        assert type(row[column]) is float and math.copysign(1.0, row[column]) == 1.0
        assert str(row[column]) == "0.0"
    assert row["Intensity_SaturatedFraction_DNA"] is None
    assert row["Histogram_Exact_DNA"] is True


def test_summary_rows_keep_every_other_value_as_given():
    # Review of #110: only zeros change; other values keep their type and written text, so a
    # row of caller-built values writes the bytes it wrote on v0.7.0 (values passed through).
    values = (np.float32(0.1), 5, np.float64(0.25), 0.5)
    row = _summary(*values).to_row()
    for column, value in zip(SUMMARY_VALUE_COLUMNS, values, strict=True):
        assert row[column] is value
    old = dict(row, **dict(zip(SUMMARY_VALUE_COLUMNS, values, strict=True)))
    assert _written(row) == _written(old)
    assert "0.1,5,0.25,0.5" in _written(row)


def test_positive_zero_changes_only_float_zeros():
    for value in (0, 5, True, False, None, "0", np.int64(0), np.float32(0.1), math.nan, -math.inf):
        assert positive_zero(value) is value
    for zero in (-0.0, 0.0, np.float32(-0.0), np.float64(-0.0), np.float16(-0.0)):
        result = positive_zero(zero)
        assert type(result) is float and math.copysign(1.0, result) == 1.0


def test_percentile_name_keeps_writing_zero_as_0():
    assert percentile_name(-0.0, 99.8) == percentile_name(0.0, 99.8) == "percentile 0-99.8"
