import csv
import re
from pathlib import Path

import numpy as np
import pytest

from morphobricks_normalization import (
    ChannelMeasurementError,
    ChannelMeasurementFailure,
    PlaneRecord,
    compute_plane_histogram,
    csv_io,
    make_plane_id,
    read_plane_histogram_table,
    read_plane_histograms,
    write_plane_histograms,
)
from morphobricks_normalization.csv_io import (
    METADATA_COLUMNS,
    CsvFormatError,
    bin_column,
    header_channels,
    table_header,
)

rng = np.random.default_rng(7)
DATA = Path(__file__).parent / "data"
# Contract §6.1 (N42 as amended by ADR 0003 HV2): <Category>_<Feature>_<channel ID>[_<parameter>].
TOKEN = r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*"
PER_CHANNEL_NAME = re.compile(rf"[A-Z][A-Za-z]*_[A-Z][A-Za-z]*_{TOKEN}")
IDS = ("DNA", "ACTIN")  # source order C0, C1; sorted they are ACTIN, DNA
UNIVERSE = ("DNA", "ACTIN", "PHASE")


def record(
    path="control/bio_rep1/tech_rep1/img.tif",
    labels=("control", "bio_rep1", "tech_rep1"),
    channels=None,
    bin_count=None,
    position_id="p",
    ids=None,
    **coords,
):
    if channels is None:
        channels = [rng.integers(0, 4000, (16, 16)).astype(np.uint16) for _ in range(2)]
    if ids is None:
        ids = IDS[: len(channels)] if len(channels) <= 2 else UNIVERSE[: len(channels)]
    histograms = [compute_plane_histogram(pixels, bin_count) for pixels in channels]
    return PlaneRecord(
        path, *labels, histograms=histograms, position_id=position_id, channels=ids, **coords
    )


def write(records, path, channel_ids=UNIVERSE, **kwargs):
    return write_plane_histograms(records, path, channel_ids=channel_ids, **kwargs)


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
        "channels",
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
    path = write(records, tmp_path / "plane_histograms.csv")
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
    path = write(original, tmp_path / "t.csv")
    for a, b in zip(original, read_plane_histograms(path), strict=True):
        assert_same(a, b)


@pytest.mark.parametrize("label", ["NA", "nan", "None", "001", "TRUE", "1e3", "a,b", 'say "hi"'])
def test_labels_are_kept_as_text(tmp_path, label):
    original = record(f"{label}/img.tif", (label, "single", "single"))
    (back,) = read_plane_histograms(write([original], tmp_path / "t.csv"))
    assert back.condition == label and back.source_relative_path == f"{label}/img.tif"


def test_header_follows_the_contract(tmp_path):
    path = write([record(bin_count=3)], tmp_path / "t.csv")
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
    assert "c" not in header and "failed_channels" not in header
    # HV1: one presence column right after the plane columns.
    assert header[len(METADATA_COLUMNS)] == "channels"
    per_channel = header[len(METADATA_COLUMNS) + 1 :]
    expected = []
    for c in ("ACTIN", "DNA", "PHASE"):  # every ID of the experiment, sorted; features, bins
        expected += [
            f"Intensity_MinIntensity_{c}",
            f"Intensity_MaxIntensity_{c}",
            f"Histogram_ValidPixelCount_{c}",
            f"Histogram_BinningMin_{c}",
            f"Histogram_BinningMax_{c}",
            f"Histogram_UnderflowCount_{c}",
            f"Histogram_UnderflowMean_{c}",
            f"Histogram_UnderflowStd_{c}",
            f"Histogram_OverflowCount_{c}",
            f"Histogram_OverflowMean_{c}",
            f"Histogram_OverflowStd_{c}",
        ]
        expected += [f"Histogram_Count_{c}_{i:05d}" for i in range(3)]
    assert per_channel == expected
    assert all(PER_CHANNEL_NAME.fullmatch(name) for name in per_channel)
    assert header == table_header(["PHASE", "DNA", "ACTIN"], 3)


def test_rows_hold_every_channel_side_by_side_by_id(tmp_path):
    channels = [np.array([[3, 5], [5, 9]], np.uint16), np.array([[0, 0], [1, 2]], np.uint16)]
    path = write([record(channels=channels, bin_count=8, z=2)], tmp_path / "t.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        (row,) = csv.DictReader(fh)
    assert row["plane_id"] == "control/bio_rep1/tech_rep1/img.tif::z=2,t=0"
    assert (row["dtype"], row["bit_depth"], row["channel_count"]) == ("uint16", "16", "2")
    assert (row["total_pixel_count"], row["bin_count"], row["schema_version"]) == ("4", "8", "5")
    # Source order in the presence column: C0 is DNA, C1 is ACTIN.
    assert row["channels"] == "DNA;ACTIN"
    assert (row["Intensity_MinIntensity_DNA"], row["Intensity_MaxIntensity_DNA"]) == ("3", "9")
    assert (row["Histogram_BinningMin_ACTIN"], row["Histogram_BinningMax_ACTIN"]) == ("0", "2")
    assert row["Histogram_ValidPixelCount_ACTIN"] == "4"
    # The min-max range leaves no pixel outside the bins (contract §1.4).
    assert [
        row[f"Histogram_{tail}{what}_DNA"]
        for tail in ("Underflow", "Overflow")
        for what in ("Count", "Mean", "Std")
    ] == ["0", "0.0", "0.0"] * 2
    assert [row[bin_column("DNA", i)] for i in range(8)] == ["1", "0", "2", "0", "0", "0", "1", "0"]
    assert [row[bin_column("ACTIN", i)] for i in range(3)] == ["2", "1", "1"]
    # PHASE is in the experiment but absent from this plane: every cell blank, never zero.
    assert all(row[name] == "" for name in row if name.endswith("_PHASE") or "_PHASE_" in name)


def test_plane_id_has_both_coordinates_and_no_channel(tmp_path):
    assert record().plane_id == "control/bio_rep1/tech_rep1/img.tif::z=0,t=0"
    assert record(z=3, t=1).plane_id == "control/bio_rep1/tech_rep1/img.tif::z=3,t=1"
    # An axis the image does not have is coordinate 0, also when given as None.
    assert (record(z=None).z, record(z=None).plane_id) == (0, record().plane_id)
    assert make_plane_id("img.tif", z=None, t=2) == "img.tif::z=0,t=2"


def test_mixed_bin_counts_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="same bin count"):
        write([record(bin_count=10), record(bin_count=20)], tmp_path / "t.csv")


def test_images_with_different_channels_are_written_with_blank_absent_cells(tmp_path):
    # ADR 0003 HV1 replaces N43: rows may have different channels.
    one = [rng.integers(0, 10, (4, 4)).astype(np.uint16)]
    planes = [record("a/1.tif"), record("a/2.tif", channels=one, ids=("PHASE",))]
    table = read_plane_histogram_table(write(planes, tmp_path / "t.csv"))
    assert table.channel_ids == ("ACTIN", "DNA", "PHASE")
    assert [r.channels for r in table.records] == [("DNA", "ACTIN"), ("PHASE",)]
    for a, b in zip(planes, table.records, strict=True):
        assert_same(a, b)


def test_a_record_channel_outside_the_experiment_is_refused(tmp_path):
    with pytest.raises(ValueError, match=r"not in the experiment's channel IDs.*PAX_488"):
        write([record(ids=("DNA", "PAX_488"))], tmp_path / "t.csv")
    assert not (tmp_path / "t.csv").exists()


@pytest.mark.parametrize("ids", [[], ["DNA", "DNA"], ["0"], [0, 1], "DNA"])
def test_the_experiment_channel_ids_must_be_ids(tmp_path, ids):
    with pytest.raises(ValueError):
        write([record()], tmp_path / "t.csv", channel_ids=ids)


def test_planes_of_one_image_with_different_channels_are_refused(tmp_path):
    planes = [record("a/1.tif", t=0), record("a/1.tif", ids=("ACTIN", "DNA"), t=1)]
    with pytest.raises(ValueError, match=r"a/1\.tif \(DNA;ACTIN in .*ACTIN;DNA in"):
        write(planes, tmp_path / "t.csv")


def test_a_failed_measurement_writes_nothing(tmp_path):
    # HV3: the table is not written, and the run is told which plane and channel failed.
    path = tmp_path / "t.csv"
    failure = ChannelMeasurementFailure(record("a/2.tif").plane_id, "ACTIN", "read error")
    with pytest.raises(ChannelMeasurementError) as error:
        write([record("a/1.tif"), record("a/2.tif")], path, failures=[failure])
    assert error.value.failures == (failure,) and error.value.removed is None
    assert "plane a/2.tif::z=0,t=0, channel ACTIN: read error" in str(error.value)
    assert "plane_histograms.csv was not written" in str(error.value)
    assert "earlier run" not in str(error.value)
    assert list(tmp_path.iterdir()) == []


def test_a_failed_measurement_removes_an_earlier_table_so_none_looks_current(tmp_path):
    # Open question 5 of PR #90: an earlier run's table left in place would read as this
    # run's, so the writer removes it and says so; nothing else in the folder is touched.
    path = write([record()], tmp_path / "t.csv")
    other = tmp_path / "other.csv"
    other.write_text("kept", encoding="utf-8")
    failure = ChannelMeasurementFailure(record("a/2.tif").plane_id, "ACTIN", "read error")
    with pytest.raises(ChannelMeasurementError) as error:
        write([record("a/1.tif"), record("a/2.tif")], path, failures=[failure])
    assert error.value.removed == str(path) and error.value.failures == (failure,)
    assert f"The table of an earlier run, {path}, was removed" in str(error.value)
    assert [p.name for p in tmp_path.iterdir()] == ["other.csv"]
    assert other.read_text(encoding="utf-8") == "kept"


def test_the_channels_of_a_plane_share_type_bins_and_size():
    a = compute_plane_histogram(np.arange(16, dtype=np.uint16).reshape(4, 4))
    two = {"position_id": "p", "channels": ("DNA", "ACTIN")}
    with pytest.raises(ValueError, match="channel ACTIN has pixel type uint8, but channel DNA"):
        PlaneRecord(
            "p.tif", "c", "r", "t", (a, compute_plane_histogram(np.zeros(16, np.uint8))), **two
        )
    with pytest.raises(ValueError, match="channel ACTIN has bin count 4"):
        PlaneRecord(
            "p.tif", "c", "r", "t", (a, compute_plane_histogram(np.zeros(16, a.dtype), 4)), **two
        )
    with pytest.raises(ValueError, match="channel ACTIN has number of pixels 9"):
        PlaneRecord(
            "p.tif", "c", "r", "t", (a, compute_plane_histogram(np.zeros(9, a.dtype))), **two
        )
    with pytest.raises(ValueError, match="has no channels"):
        PlaneRecord("p.tif", "c", "r", "t", (), position_id="p", channels=())
    single = PlaneRecord("p.tif", "c", "r", "t", a, position_id="p", channels=("DNA",))
    assert single.histograms == (a,) and single.channel_count == 1
    assert single.channel_histogram("DNA") is a
    with pytest.raises(ValueError, match="channel ACTIN is absent from it"):
        single.channel_histogram("ACTIN")
    with pytest.raises(ValueError, match="never by its position"):
        single.channel_histogram(0)


@pytest.mark.parametrize(
    "ids, match",
    [
        (("DNA",), "2 histogram\\(s\\) but 1 channel ID"),
        (("DNA", "DNA"), "listed more than once"),
        (("DNA", "C 1"), "is not a canonical channel ID"),
        ((0, 1), "never by its position"),
        ("DNA", "expected a list of channel IDs"),
    ],
)
def test_a_record_needs_one_channel_id_per_histogram(ids, match):
    a = compute_plane_histogram(np.arange(16, dtype=np.uint16))
    with pytest.raises(ValueError, match=match):
        PlaneRecord("p.tif", "c", "r", "t", (a, a), position_id="p", channels=ids)


def test_channels_have_no_default_and_are_keyword_only():
    histogram = compute_plane_histogram(np.arange(6))
    with pytest.raises(TypeError, match="channels"):
        PlaneRecord("c/a.tif", "c", "r", "t", histogram, position_id="p")


def test_failed_write_leaves_previous_file_untouched(tmp_path, monkeypatch):
    path = write([record()], tmp_path / "t.csv")
    before = path.read_bytes()
    calls = iter([False, True])

    def failing_row(rec, ids):
        if next(calls):
            raise RuntimeError("simulated failure")
        return original_row(rec, ids)

    original_row = csv_io._row
    monkeypatch.setattr(csv_io, "_row", failing_row)
    with pytest.raises(RuntimeError):
        # Two different planes: one plane twice is refused before writing (issue #43).
        write([record("a/1.tif"), record("a/2.tif")], path)
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
    path = write([record(bin_count=2)], tmp_path / "t.csv")
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
        ("Histogram_ValidPixelCount_ACTIN", "1"),
        ("Histogram_Count_DNA_00001", "999"),
        ("Intensity_MaxIntensity_ACTIN", "1"),
    ],
)
def test_inconsistent_channel_columns_are_rejected(tmp_path, column, value):
    channel = "ACTIN" if "ACTIN" in column else "DNA"
    with pytest.raises(CsvFormatError, match=f"bin counts of channel {channel} are inconsistent"):
        read_plane_histograms(table_with(tmp_path, column, value))


@pytest.mark.parametrize(
    "column, value",
    [
        ("Histogram_ValidPixelCount_PHASE", "0"),  # a fabricated zero
        ("Histogram_Count_PHASE_00000", "0"),
        ("Intensity_MinIntensity_PHASE", "nan"),  # a fabricated NaN
        ("Histogram_UnderflowMean_PHASE", "0.0"),
    ],
)
def test_a_value_in_an_absent_channel_is_refused(tmp_path, column, value):
    with pytest.raises(CsvFormatError, match=f"line 2: channel PHASE is absent .*{column}"):
        read_plane_histograms(table_with(tmp_path, column, value))


def test_a_listed_channel_with_blank_cells_is_refused(tmp_path):
    # Listing a channel claims a measured histogram; blank counts are not one (HV3).
    path = write([record(bin_count=2)], tmp_path / "t.csv")
    columns = csv_io.channel_columns("ACTIN", 2)
    edit_table(path, dict.fromkeys(columns, ""))
    with pytest.raises(CsvFormatError, match="line 2"):
        read_plane_histograms(path)


@pytest.mark.parametrize(
    "value, match",
    [
        ("", "channels cell is blank"),
        ("DNA;DNA", "listed more than once"),
        ("DNA;PAX", "lists PAX, which the header does not have"),
        ("DNA", "channel_count is '2', but its channels cell lists 1"),
        ("DNA ;ACTIN", "is not a canonical channel ID"),
        ("DNA,ACTIN", "is not a canonical channel ID"),
    ],
)
def test_a_bad_presence_cell_is_refused(tmp_path, value, match):
    with pytest.raises(CsvFormatError, match=f"line 2: .*{match}"):
        read_plane_histograms(table_with(tmp_path, "channels", value))


def test_listing_an_absent_channel_as_present_is_refused(tmp_path):
    # PHASE's cells are blank, so claiming it present (with a matching count) is refused.
    path = table_with(tmp_path, "channels", "DNA;ACTIN;PHASE")
    edit_table(path, {"channel_count": "3"})
    with pytest.raises(CsvFormatError, match="line 2"):
        read_plane_histograms(path)


def test_rows_of_one_image_with_different_channels_are_refused(tmp_path):
    planes = [record("a/1.tif", t=0), record("a/1.tif", t=1)]
    path = write(planes, tmp_path / "t.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        header, *rows = list(csv.reader(fh))
    # Swap the presence list of the second plane and its column groups to match: each row is
    # valid on its own, but one image cannot have two channel orders.
    rows[1][header.index("channels")] = "ACTIN;DNA"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([header, *rows])
    with pytest.raises(CsvFormatError, match="planes list different channels"):
        read_plane_histograms(path)


@pytest.mark.parametrize(
    "old, new",
    [
        ("Histogram_BinningMax_ACTIN", "Histogram_BinningMax_ACTINX"),  # broken group
        (",channels,", ",presence,"),  # no presence column
    ],
)
def test_unexpected_channel_columns_are_rejected(tmp_path, old, new):
    path = write([record(bin_count=2)], tmp_path / "t.csv")
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(old, new, 1), "utf-8")
    with pytest.raises(CsvFormatError, match="does not have the per-channel columns of schema"):
        read_plane_histograms(path)


def test_header_channels_follow_the_id_token_rule():
    # HV2: IDs come from Histogram_ValidPixelCount_<ID>, the bin index from the last _NNNNN,
    # even when an ID contains "_" or ends in five digits.
    ids = ["PAX_488", "DNA", "DNA_00001", "A_B_C"]
    assert header_channels(table_header(ids, 4)) == (tuple(sorted(ids)), 4)
    header = table_header(ids, 4)
    assert bin_column("DNA", 1) in header and bin_column("DNA_00001", 1) in header
    assert header.index(bin_column("DNA", 1)) < header.index("Histogram_ValidPixelCount_DNA_00001")
    unsorted = table_header(["B"], 2) + csv_io.channel_columns("A", 2)
    assert header_channels(unsorted) is None
    missing = [name for name in table_header(["A"], 2) if name != "Histogram_BinningMin_A"]
    assert header_channels(missing) is None


def test_a_version_4_table_is_refused_naming_version_5_and_the_fix(tmp_path):
    # HV4: a hard cut; the table is recomputed, never converted.
    path = tmp_path / "plane_histograms.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(
            [
                [*METADATA_COLUMNS, "Histogram_ValidPixelCount_C0"],
                ["p", "img.tif", "0", "0", "c", "r", "t", "img.tif", "img.tif::z=0,t=0"]
                + ["uint8", "8", "1", "4", "2", "4", "4"],
            ]
        )
    with pytest.raises(CsvFormatError) as error:
        read_plane_histograms(path)
    message = str(error.value)
    assert message == (
        f"{path}: This plane_histograms.csv was written with schema version 4, its channels "
        "named by their index in the image (C0, C1, ...). This version reads schema version 5 "
        "only, with every channel named by its canonical channel ID from the experiment "
        "description; old tables are not converted. Run the preprocessing brick again on this "
        "experiment: it recomputes the table (schema version 5)."
    )


def test_no_converter_from_version_4_exists():
    names = {name for name in dir(csv_io) if not name.startswith("__")}
    assert not {n for n in names if "convert" in n.lower() or "v4" in n.lower()}


@pytest.mark.parametrize("version", ["2", "1"])
def test_tables_with_one_row_per_channel_are_refused(tmp_path, version):
    # Only the schema_version value is read: the layout of such a table is not (and a table
    # of these versions is not converted); the message names the fix.
    path = tmp_path / "plane_histograms.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(
            [
                ["image_file", "c", "dtype", "schema_version", "histogram_bin_00000"],
                ["img.tif", "0", "uint16", version, "5"],
            ]
        )
    with pytest.raises(CsvFormatError) as error:
        read_plane_histograms(path)
    assert str(error.value).startswith(
        f"{path}: This plane_histograms.csv was written with schema version {version}, one row "
        "per channel. This version reads schema version 5 only"
    )
    assert str(error.value).endswith("it recomputes the table (schema version 5).")


def test_writing_is_byte_deterministic_and_shards_share_one_header(tmp_path):
    planes = [record(f"a/{i}.tif") for i in range(4)]
    one = write(planes, tmp_path / "one.csv").read_bytes()
    assert write(planes, tmp_path / "two.csv").read_bytes() == one
    # Two shards given the same experiment IDs have the same header; their rows, merged in
    # path order, give the serial table byte for byte.
    first = write(planes[2:], tmp_path / "s1.csv").read_bytes().splitlines(keepends=True)
    second = write(planes[:2], tmp_path / "s2.csv").read_bytes().splitlines(keepends=True)
    assert first[0] == second[0] == one.splitlines(keepends=True)[0]
    assert b"".join([second[0], *second[1:], *first[1:]]) == one


def test_a_table_without_a_known_version_or_the_columns_is_refused_without_a_fix_guess(tmp_path):
    path = tmp_path / "plane_histograms.csv"
    path.write_text("image_file,c\nimg.tif,0\n", encoding="utf-8")
    with pytest.raises(CsvFormatError, match="does not have the plane-histogram columns"):
        read_plane_histograms(path)
    path.write_text("image_file,c,schema_version\nimg.tif,0,9\n", encoding="utf-8")
    with pytest.raises(CsvFormatError, match="does not have the plane-histogram columns"):
        read_plane_histograms(path)


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


@pytest.mark.parametrize("value", [None, "", "  ", 7])
def test_a_record_without_a_position_id_is_refused(value):
    histogram = compute_plane_histogram(np.arange(6))
    with pytest.raises(ValueError, match=r"c/a\.tif.*no position_id.*preprocessing brick"):
        PlaneRecord("c/a.tif", "c", "r", "t", histogram, position_id=value, channels=("DNA",))


def test_position_id_has_no_default_and_is_keyword_only():
    histogram = compute_plane_histogram(np.arange(6))
    with pytest.raises(TypeError, match="position_id"):
        PlaneRecord("c/a.tif", "c", "r", "t", histogram, channels=("DNA",))
    with pytest.raises(TypeError):
        PlaneRecord("c/a.tif", "c", "r", "t", histogram, 0, 0, "p", channels=("DNA",))


def test_a_row_with_a_blank_position_id_is_refused_not_filled(tmp_path):
    path = write([record()], tmp_path / "t.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        header, *rows = list(csv.reader(fh))
    for row in rows:
        row[header.index("position_id")] = ""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([header, *rows])
    with pytest.raises(CsvFormatError, match=r"line 2: .*no position_id"):
        read_plane_histograms(path)


def test_position_id_round_trips_as_written(tmp_path):
    original = record(position_id="NA")
    (back,) = read_plane_histograms(write([original], tmp_path / "t.csv"))
    assert back.position_id == "NA"
