"""Tables of earlier schema versions are refused by the version-5 reader (ADR 0003 HV4).

A table is recomputed, never converted: the reader names version 5 and the fix.
"""

import csv

import numpy as np
import pytest

from morphobricks_normalization import (
    PlaneRecord,
    compute_plane_histogram,
    read_plane_histograms,
    write_plane_histograms,
)
from morphobricks_normalization.csv_io import CsvFormatError

C0 = ("C0",)


def as_version_3(path):
    """Rewrite a version-5 table as version 3: no tail columns, schema_version 3."""
    with open(path, newline="", encoding="utf-8") as fh:
        header, *rows = list(csv.reader(fh))
    keep = [i for i, name in enumerate(header) if "flow" not in name]
    version = header.index("schema_version")
    for row in rows:
        row[version] = "3"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([[r[i] for i in keep] for r in [header, *rows]])
    return path


def test_version_3_is_refused(tmp_path):
    records = [
        PlaneRecord(
            "c/a.tif",
            "c",
            "r",
            "t",
            compute_plane_histogram(np.arange(6)),
            position_id="p",
            channels=C0,
        )
    ]
    path = as_version_3(
        write_plane_histograms(records, tmp_path / "plane_histograms.csv", channel_ids=C0)
    )
    with pytest.raises(CsvFormatError) as error:
        read_plane_histograms(path)
    assert str(error.value) == (
        f"{path}: This plane_histograms.csv was written with schema version 3, its channels "
        "named by index and without the underflow and overflow columns. This version reads "
        "schema version 5 only, with every channel named by its canonical channel ID from the "
        "experiment description; old tables are not converted. Run the preprocessing brick "
        "again on this experiment: it recomputes the table (schema version 5)."
    )


def test_a_version_3_header_without_rows_is_refused_by_its_columns(tmp_path):
    # No row, so no schema_version value to read: the missing tail columns refuse it.
    records = [
        PlaneRecord(
            "c/a.tif",
            "c",
            "r",
            "t",
            compute_plane_histogram(np.arange(6)),
            position_id="p",
            channels=C0,
        )
    ]
    path = as_version_3(
        write_plane_histograms(records, tmp_path / "plane_histograms.csv", channel_ids=C0)
    )
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text(lines[0], encoding="utf-8")
    with pytest.raises(CsvFormatError, match="does not have the per-channel columns of schema"):
        read_plane_histograms(path)


@pytest.mark.parametrize(
    "version, match",
    [("3", "schema version 3, its channels named by index"), ("4", "schema version 4, its")],
)
def test_a_row_of_an_earlier_version_in_a_version_5_table_is_refused(tmp_path, version, match):
    records = [
        PlaneRecord(
            f"c/{name}.tif",
            "c",
            "r",
            "t",
            compute_plane_histogram(np.arange(6)),
            position_id="p",
            channels=C0,
        )
        for name in ("a", "b")
    ]
    path = write_plane_histograms(records, tmp_path / "t.csv", channel_ids=C0)
    with open(path, newline="", encoding="utf-8") as fh:
        header, *rows = list(csv.reader(fh))
    rows[1][header.index("schema_version")] = version  # the second row, not the first
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([header, *rows])
    with pytest.raises(CsvFormatError, match=match):
        read_plane_histograms(path)
