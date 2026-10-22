import csv

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
from morphobricks_normalization.csv_io import METADATA_COLUMNS, CsvFormatError

rng = np.random.default_rng(7)


def record(
    path="control/bio_rep1/tech_rep1/img.tif",
    labels=("control", "bio_rep1", "tech_rep1"),
    pixels=None,
    bin_count=None,
    **coords,
):
    pixels = rng.integers(0, 4000, (16, 16)).astype(np.uint16) if pixels is None else pixels
    return PlaneRecord(
        path, *labels, histogram=compute_plane_histogram(pixels, bin_count), **coords
    )


def assert_same(a, b):
    for field in (
        "source_relative_path",
        "condition",
        "biological_repeat",
        "technical_repeat",
        "z",
        "c",
        "t",
        "position_id",
        "plane_id",
        "image_file",
    ):
        assert getattr(a, field) == getattr(b, field)
    ha, hb = a.histogram, b.histogram
    assert ha.dtype == hb.dtype and ha.bin_count == hb.bin_count
    assert (ha.binning_min, ha.binning_max) == (hb.binning_min, hb.binning_max)
    assert (ha.total_pixel_count, ha.valid_pixel_count) == (
        hb.total_pixel_count,
        hb.valid_pixel_count,
    )
    np.testing.assert_array_equal(ha.counts, hb.counts)


def test_integer_round_trip(tmp_path):
    records = [record(), record("single/single/single/b.tif", ("single",) * 3, z=0, c=2, t=5)]
    path = write_plane_histograms(records, tmp_path / "plane_histograms.csv")
    for a, b in zip(records, read_plane_histograms(path), strict=True):
        assert_same(a, b)


def test_float_round_trip_keeps_full_precision(tmp_path):
    pixels = rng.normal(0, 1, (16, 16)) * np.pi
    pixels[0, 0] = np.nan
    original = [
        record(pixels=pixels, bin_count=128),
        record("x/a.tif", ("x", "single", "single"), pixels=np.full(4, np.nan), bin_count=128),
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
    assert header[:8] == [
        "position_id",
        "image_file",
        "z",
        "c",
        "t",
        "condition",
        "biological_repeat",
        "technical_repeat",
    ]
    assert header[len(METADATA_COLUMNS) :] == [
        "histogram_bin_00000",
        "histogram_bin_00001",
        "histogram_bin_00002",
    ]


def test_plane_id_always_has_all_coordinates(tmp_path):
    assert record().plane_id == "control/bio_rep1/tech_rep1/img.tif::z=0,c=0,t=0"
    assert record(z=3, c=1).plane_id == "control/bio_rep1/tech_rep1/img.tif::z=3,c=1,t=0"
    # An axis the image does not have is coordinate 0, also when given as None.
    assert (record(z=None).z, record(z=None).plane_id) == (0, record().plane_id)
    assert make_plane_id("img.tif", z=None, t=2) == "img.tif::z=0,c=0,t=2"


def test_mixed_bin_counts_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="same bin count"):
        write_plane_histograms([record(bin_count=10), record(bin_count=20)], tmp_path / "t.csv")


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


def table_with(tmp_path, column, value):
    """Write a one-row table, then set ``column`` of its row to ``value``."""
    # plane_id holds commas, so the row is edited with the csv module.
    path = write_plane_histograms([record(bin_count=2)], tmp_path / "t.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        header, row = list(csv.reader(fh))
    row[METADATA_COLUMNS.index(column)] = value
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([header, row])
    return path


def test_blank_coordinates_are_rejected(tmp_path):
    # Version 1 tables left absent axes blank; version 2 always writes them.
    with pytest.raises(CsvFormatError, match="line 2: invalid literal"):
        read_plane_histograms(table_with(tmp_path, "z", ""))


def test_unknown_schema_version_is_rejected(tmp_path):
    with pytest.raises(CsvFormatError, match="schema version"):
        read_plane_histograms(table_with(tmp_path, "schema_version", "99"))
