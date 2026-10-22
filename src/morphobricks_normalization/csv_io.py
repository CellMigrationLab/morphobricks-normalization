"""Read and write the ``plane_histograms.csv`` table.

One row per plane. The columns before the histogram follow the shared
MorphoBricks contract (see ``docs/contract.md``); then come ``bin_count`` and
``schema_version``, and one column per bin (``histogram_bin_00000`` ...).

Labels are always read back as text, so folder names such as ``NA``, ``001``
or ``TRUE`` keep their exact spelling. Float values are written at full
precision and read back unchanged.
"""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .dtypes import bit_depth, check_supported, is_integer
from .histogram import PlaneHistogram
from .records import PlaneRecord

# Version 2: z, c and t are always written (0 for an axis the image lacks).
SCHEMA_VERSION = "2"
OUTPUT_FILENAME = "plane_histograms.csv"

METADATA_COLUMNS = (
    "position_id",
    "image_file",
    "z",
    "c",
    "t",
    "condition",
    "biological_repeat",
    "technical_repeat",
    "source_relative_path",
    "plane_id",
    "dtype",
    "bit_depth",
    "pixel_min",
    "pixel_max",
    "total_pixel_count",
    "valid_pixel_count",
    "binning_min",
    "binning_max",
    "bin_count",
    "schema_version",
)
BIN_COLUMN_PREFIX = "histogram_bin_"


class CsvFormatError(ValueError):
    """Raised when a file does not follow the plane-histogram CSV format."""


def bin_column(index: int) -> str:
    """Column name of bin ``index``, e.g. ``histogram_bin_00042``."""
    return f"{BIN_COLUMN_PREFIX}{index:05d}"


def write_plane_histograms(records: Iterable[PlaneRecord], path: str | Path) -> Path:
    """Write records to a CSV file, replacing it only once writing succeeded.

    Parameters
    ----------
    records
        Plane records; all must use the same bin count.
    path
        Destination file.

    Returns
    -------
    pathlib.Path
        The written file.

    Raises
    ------
    ValueError
        If there are no records or their bin counts differ.
    """
    records = list(records)
    if not records:
        raise ValueError("There are no planes to write.")
    bin_counts = {r.histogram.bin_count for r in records}
    if len(bin_counts) != 1:
        raise ValueError(
            f"All planes in one table must use the same bin count; found {sorted(bin_counts)}."
        )
    path = Path(path)
    n_bins = bin_counts.pop()
    header = [*METADATA_COLUMNS, *(bin_column(i) for i in range(n_bins))]
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for record in records:
                writer.writerow(_row(record))
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def read_plane_histograms(path: str | Path) -> list[PlaneRecord]:
    """Read a plane-histogram CSV written by :func:`write_plane_histograms`.

    Raises
    ------
    CsvFormatError
        If the header, schema version or a row does not match the format.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise CsvFormatError(f"{path} is empty.") from None
        n_meta = len(METADATA_COLUMNS)
        if tuple(header[:n_meta]) != METADATA_COLUMNS:
            raise CsvFormatError(f"{path} does not have the plane-histogram columns.")
        n_bins = len(header) - n_meta
        if header[n_meta:] != [bin_column(i) for i in range(n_bins)]:
            raise CsvFormatError(f"{path} has unexpected histogram column names.")
        return [_record(row, n_meta, n_bins, path, line) for line, row in enumerate(reader, 2)]


def _row(record: PlaneRecord) -> list[str]:
    hist = record.histogram
    return [
        _text(record.position_id),
        record.image_file,
        _text(record.z),
        _text(record.c),
        _text(record.t),
        record.condition,
        record.biological_repeat,
        record.technical_repeat,
        record.source_relative_path,
        record.plane_id,
        hist.dtype.name,
        str(bit_depth(hist.dtype)),
        _text(hist.pixel_min),
        _text(hist.pixel_max),
        str(hist.total_pixel_count),
        str(hist.valid_pixel_count),
        _text(hist.binning_min),
        _text(hist.binning_max),
        str(hist.bin_count),
        SCHEMA_VERSION,
        *map(str, hist.counts.tolist()),
    ]


def _record(row: list[str], n_meta: int, n_bins: int, path, line: int) -> PlaneRecord:
    if len(row) != n_meta + n_bins:
        raise CsvFormatError(f"{path}, line {line}: expected {n_meta + n_bins} values.")
    meta = dict(zip(METADATA_COLUMNS, row[:n_meta], strict=True))
    if meta["schema_version"] != SCHEMA_VERSION:
        raise CsvFormatError(
            f"{path}, line {line}: schema version {meta['schema_version']!r} is not supported "
            f"(this version reads {SCHEMA_VERSION!r})."
        )
    try:
        dtype = check_supported(meta["dtype"])
        number = int if is_integer(dtype) else float
        histogram = PlaneHistogram(
            dtype=dtype,
            bin_count=int(meta["bin_count"]),
            binning_min=_value(meta["binning_min"], number),
            binning_max=_value(meta["binning_max"], number),
            counts=np.array(row[n_meta:], dtype=np.int64),
            total_pixel_count=int(meta["total_pixel_count"]),
            valid_pixel_count=int(meta["valid_pixel_count"]),
        )
        record = PlaneRecord(
            source_relative_path=meta["source_relative_path"],
            condition=meta["condition"],
            biological_repeat=meta["biological_repeat"],
            technical_repeat=meta["technical_repeat"],
            histogram=histogram,
            z=int(meta["z"]),
            c=int(meta["c"]),
            t=int(meta["t"]),
            position_id=meta["position_id"] or None,
        )
    except ValueError as error:
        raise CsvFormatError(f"{path}, line {line}: {error}") from error
    if histogram.bin_count != n_bins or int(histogram.counts.sum()) != histogram.valid_pixel_count:
        raise CsvFormatError(f"{path}, line {line}: the bin counts are inconsistent.")
    return record


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _value(text: str, number: type):
    return None if text == "" else number(text)
