import csv
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from morphobricks_normalization import (
    PlaneRecord,
    compute_plane_histogram,
    csv_io,
    make_plane_id,
    read_plane_histograms,
    write_plane_histograms,
)
from morphobricks_normalization.csv_io import METADATA_COLUMNS, CsvFormatError, bin_column

rng = np.random.default_rng(7)
DATA = Path(__file__).parent / "data"
# Contract §6.1 (N42): <Category>_<Feature>_C<index>[_<parameter>], CamelCase words.
PER_CHANNEL_NAME = re.compile(r"[A-Z][A-Za-z]*_[A-Z][A-Za-z]*_C(0|[1-9][0-9]*)(_[0-9]{5})?")


def record(
    path="control/bio_rep1/tech_rep1/img.tif",
    labels=("control", "bio_rep1", "tech_rep1"),
    channels=None,
    bin_count=None,
    **coords,
):
    if channels is None:
        channels = [rng.integers(0, 4000, (16, 16)).astype(np.uint16) for _ in range(2)]
    histograms = [compute_plane_histogram(pixels, bin_count) for pixels in channels]
    return PlaneRecord(path, *labels, histograms=histograms, **coords)


def assert_same(a, b):
    for field in (
        "source_relative_path",
        "condition",
        "biological_repeat",
        "technical_repeat",
        "z",
        "t",
        "position_id",
        "plane_id",
        "image_file",
        "channel_count",
    ):
        assert getattr(a, field) == getattr(b, field)
    for ha, hb in zip(a.histograms, b.histograms, strict=True):
        assert ha.dtype == hb.dtype and ha.bin_count == hb.bin_count
        assert (ha.binning_min, ha.binning_max) == (hb.binning_min, hb.binning_max)
        assert (ha.total_pixel_count, ha.valid_pixel_count) == (
            hb.total_pixel_count,
            hb.valid_pixel_count,
        )
        np.testing.assert_array_equal(ha.counts, hb.counts)


def test_integer_round_trip(tmp_path):
    three = [rng.integers(0, 4000, (16, 16)).astype(np.uint16) for _ in range(3)]
    records = [
        record(channels=three),
        record("single/single/single/b.tif", ("single",) * 3, channels=three[::-1], z=0, t=5),
    ]
    path = write_plane_histograms(records, tmp_path / "plane_histograms.csv")
    for a, b in zip(records, read_plane_histograms(path), strict=True):
        assert_same(a, b)


def test_float_round_trip_keeps_full_precision(tmp_path):
    pixels = rng.normal(0, 1, (16, 16)) * np.pi
    pixels[0, 0] = np.nan
    original = [
        record(channels=[pixels, -pixels], bin_count=128),
        # A channel without finite pixels next to one with them.
        record(
            "x/a.tif",
            ("x", "single", "single"),
            channels=[np.full(4, np.nan), np.arange(4.0)],
            bin_count=128,
        ),
    ]
    path = write_plane_histograms(original, tmp_path / "t.csv")
    for a, b in zip(original, read_plane_histograms(path), strict=True):
        assert_same(a, b)


@pytest.mark.parametrize("label", ["NA", "nan", "None", "001", "TRUE", "1e3", "a,b", 'say "hi"'])
def test_labels_are_kept_as_text(tmp_path, label):
    original = record(f"{label}/img.tif", (label, "single", "single"))
    (back,) = read_plane_histograms(write_plane_histograms([original], tmp_path / "t.csv"))
    assert back.condition == label and back.source_relative_path == f"{label}/img.tif"


def test_header_follows_the_contract(tmp_path):
    path = write_plane_histograms([record(bin_count=3)], tmp_path / "t.csv")
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert tuple(header[: len(METADATA_COLUMNS)]) == METADATA_COLUMNS
    assert header[:9] == [
        "position_id",
        "image_file",
        "z",
        "t",
        "condition",
        "biological_repeat",
        "technical_repeat",
        "source_relative_path",
        "plane_id",
    ]
    assert "c" not in header
    per_channel = header[len(METADATA_COLUMNS) :]
    expected = []
    for c in (0, 1):  # grouped by channel: features, then bins
        expected += [
            f"Intensity_MinIntensity_C{c}",
            f"Intensity_MaxIntensity_C{c}",
            f"Histogram_ValidPixelCount_C{c}",
            f"Histogram_BinningMin_C{c}",
            f"Histogram_BinningMax_C{c}",
            f"Histogram_UnderflowCount_C{c}",
            f"Histogram_UnderflowSum_C{c}",
            f"Histogram_UnderflowSumOfSquares_C{c}",
            f"Histogram_OverflowCount_C{c}",
            f"Histogram_OverflowSum_C{c}",
            f"Histogram_OverflowSumOfSquares_C{c}",
        ]
        expected += [f"Histogram_Count_C{c}_{i:05d}" for i in range(3)]
    assert per_channel == expected
    assert all(PER_CHANNEL_NAME.fullmatch(name) for name in per_channel)


def test_rows_hold_every_channel_side_by_side(tmp_path):
    channels = [np.array([[3, 5], [5, 9]], np.uint16), np.array([[0, 0], [1, 2]], np.uint16)]
    path = write_plane_histograms([record(channels=channels, bin_count=8, z=2)], tmp_path / "t.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        (row,) = csv.DictReader(fh)
    assert row["plane_id"] == "control/bio_rep1/tech_rep1/img.tif::z=2,t=0"
    assert (row["dtype"], row["bit_depth"], row["channel_count"]) == ("uint16", "16", "2")
    assert (row["total_pixel_count"], row["bin_count"], row["schema_version"]) == ("4", "8", "4")
    assert (row["Intensity_MinIntensity_C0"], row["Intensity_MaxIntensity_C0"]) == ("3", "9")
    assert (row["Histogram_BinningMin_C1"], row["Histogram_BinningMax_C1"]) == ("0", "2")
    assert row["Histogram_ValidPixelCount_C1"] == "4"
    # The min-max range leaves no pixel outside the bins (contract §1.4).
    assert [
        row[f"Histogram_{tail}{what}_C0"]
        for tail in ("Underflow", "Overflow")
        for what in ("Count", "Sum", "SumOfSquares")
    ] == ["0", "0.0", "0.0"] * 2
    assert [row[bin_column(0, i)] for i in range(8)] == ["1", "0", "2", "0", "0", "0", "1", "0"]
    assert [row[bin_column(1, i)] for i in range(3)] == ["2", "1", "1"]


def test_plane_id_has_both_coordinates_and_no_channel(tmp_path):
    assert record().plane_id == "control/bio_rep1/tech_rep1/img.tif::z=0,t=0"
    assert record(z=3, t=1).plane_id == "control/bio_rep1/tech_rep1/img.tif::z=3,t=1"
    # An axis the image does not have is coordinate 0, also when given as None.
    assert (record(z=None).z, record(z=None).plane_id) == (0, record().plane_id)
    assert make_plane_id("img.tif", z=None, t=2) == "img.tif::z=0,t=2"


def test_mixed_bin_counts_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="same bin count"):
        write_plane_histograms([record(bin_count=10), record(bin_count=20)], tmp_path / "t.csv")


def test_images_with_different_channel_counts_are_not_written(tmp_path):
    one = [rng.integers(0, 10, (4, 4)).astype(np.uint16)]
    planes = [record("a/1.tif"), record("a/2.tif", channels=one), record("a/2.tif", channels=one)]
    with pytest.raises(ValueError) as error:
        write_plane_histograms(planes, tmp_path / "t.csv")
    message = str(error.value)
    assert "These images have different numbers of channels" in message
    assert "(1 channel(s): a/2.tif; 2 channel(s): a/1.tif)" in message
    assert "Every image of an experiment must have the same channels" in message
    assert not (tmp_path / "t.csv").exists()


def test_the_channels_of_a_plane_share_type_bins_and_size():
    a = compute_plane_histogram(np.arange(16, dtype=np.uint16).reshape(4, 4))
    with pytest.raises(ValueError, match="channel 1 has pixel type uint8, but channel 0"):
        PlaneRecord("p.tif", "c", "r", "t", (a, compute_plane_histogram(np.zeros(16, np.uint8))))
    with pytest.raises(ValueError, match="channel 1 has bin count 4"):
        PlaneRecord("p.tif", "c", "r", "t", (a, compute_plane_histogram(np.zeros(16, a.dtype), 4)))
    with pytest.raises(ValueError, match="channel 1 has number of pixels 9"):
        PlaneRecord("p.tif", "c", "r", "t", (a, compute_plane_histogram(np.zeros(9, a.dtype))))
    with pytest.raises(ValueError, match="has no channels"):
        PlaneRecord("p.tif", "c", "r", "t", ())
    single = PlaneRecord("p.tif", "c", "r", "t", a)
    assert single.histograms == (a,) and single.channel_count == 1
    with pytest.raises(ValueError, match="has 1 channel"):
        single.channel_histogram(1)


def test_failed_write_leaves_previous_file_untouched(tmp_path, monkeypatch):
    path = write_plane_histograms([record()], tmp_path / "t.csv")
    before = path.read_bytes()
    calls = iter([False, True])

    def failing_row(rec):
        if next(calls):
            raise RuntimeError("simulated failure")
        return original_row(rec)

    original_row = csv_io._row
    monkeypatch.setattr(csv_io, "_row", failing_row)
    with pytest.raises(RuntimeError):
        write_plane_histograms([record(), record()], path)
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["t.csv"]  # no temporary files left


def edit_table(path, changes, row_index=0):
    """Set ``changes`` (column -> value) in one data row of a table."""
    # plane_id holds commas, so rows are edited with the csv module.
    with open(path, newline="", encoding="utf-8") as fh:
        header, *rows = list(csv.reader(fh))
    for column, value in changes.items():
        rows[row_index][header.index(column)] = value
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([header, *rows])
    return path


def table_with(tmp_path, column, value):
    """Write a one-row table, then set ``column`` of its row to ``value``."""
    path = write_plane_histograms([record(bin_count=2)], tmp_path / "t.csv")
    return edit_table(path, {column: value})


def test_blank_coordinates_are_rejected(tmp_path):
    with pytest.raises(CsvFormatError, match="line 2: invalid literal"):
        read_plane_histograms(table_with(tmp_path, "z", ""))


def test_unknown_schema_version_is_rejected(tmp_path):
    with pytest.raises(CsvFormatError, match="schema version '99' is not supported"):
        read_plane_histograms(table_with(tmp_path, "schema_version", "99"))


@pytest.mark.parametrize(
    "column, value",
    [
        ("Histogram_ValidPixelCount_C1", "1"),
        ("Histogram_Count_C0_00001", "999"),
        ("Intensity_MaxIntensity_C1", "1"),
    ],
)
def test_inconsistent_channel_columns_are_rejected(tmp_path, column, value):
    channel = re.search(r"_C([0-9]+)", column)[1]
    with pytest.raises(CsvFormatError, match=f"bin counts of channel {channel} are inconsistent"):
        read_plane_histograms(table_with(tmp_path, column, value))


def test_rows_with_another_channel_count_are_rejected(tmp_path):
    planes = [record("a/1.tif"), record("a/2.tif"), record("a/3.tif")]
    path = write_plane_histograms(planes, tmp_path / "t.csv")
    edit_table(path, {"channel_count": "3"}, row_index=1)
    with pytest.raises(CsvFormatError) as error:
        read_plane_histograms(path)
    message = str(error.value)
    assert "Its images have different numbers of channels" in message
    assert "(2 channel(s): a/1.tif, a/3.tif; 3 channel(s): a/2.tif)" in message
    assert "Every image of an experiment must have the same channels" in message


def test_unexpected_channel_columns_are_rejected(tmp_path):
    path = write_plane_histograms([record(bin_count=2)], tmp_path / "t.csv")
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("Histogram_BinningMax_C1", "Histogram_BinningMax_C2"), "utf-8")
    with pytest.raises(CsvFormatError, match="unexpected per-channel column names"):
        read_plane_histograms(path)


@pytest.mark.parametrize("version", ["2", "1"])
def test_tables_with_one_row_per_channel_are_refused(tmp_path, version):
    # A table written by the Milestone 2 code (schema version 2).
    path = shutil.copy(DATA / "v2_uint16.csv", tmp_path / "plane_histograms.csv")
    if version == "1":
        with open(path, newline="", encoding="utf-8") as fh:
            header, *rows = list(csv.reader(fh))
        for row in rows:
            row[header.index("schema_version")] = "1"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows([header, *rows])
    with pytest.raises(CsvFormatError) as error:
        read_plane_histograms(path)
    assert str(error.value) == (
        f"{path}: This plane_histograms.csv was written with schema version {version}, one row "
        "per channel. Run the preprocessing brick again on this experiment: it rewrites the "
        "table with the channels in columns (schema version 4)."
    )


@pytest.mark.parametrize(
    "column, value, implied",
    [
        (
            "plane_id",
            "control/bio_rep1/tech_rep1/other.tif::z=0,t=0",
            "control/bio_rep1/tech_rep1/img.tif::z=0,t=0",
        ),
        ("plane_id", "control/bio_rep1/tech_rep1/img.tif::z=1,t=0", None),
        ("image_file", "other.tif", "img.tif"),
        ("bit_depth", "8", "16"),
    ],
)
def test_derived_plane_columns_must_match_the_row(tmp_path, column, value, implied):
    # plane_id, image_file and bit_depth follow from source_relative_path, z, t and dtype;
    # a row that disagrees is refused, not silently recomputed (preprocessing#39).
    with pytest.raises(CsvFormatError, match=f"line 2: its {column} is {value!r}") as error:
        read_plane_histograms(table_with(tmp_path, column, value))
    if implied is not None:
        assert repr(implied) in str(error.value)
