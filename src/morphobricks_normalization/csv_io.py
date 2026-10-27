"""Read and write the ``plane_histograms.csv`` table.

One row per XYC plane: every channel of one image at one Z and T position
(contract §6.2, N41). The plane columns come first: the shared MorphoBricks
columns (see ``docs/contract.md``), then ``dtype``, ``bit_depth``,
``channel_count``, ``total_pixel_count``, ``bin_count`` and ``schema_version``.
Then, for each channel ``i`` in turn, its columns named as in CellProfiler
(N42): ``Intensity_MinIntensity_C<i>``, ``Intensity_MaxIntensity_C<i>``,
``Histogram_ValidPixelCount_C<i>``, ``Histogram_BinningMin_C<i>``,
``Histogram_BinningMax_C<i>`` and one column per bin
(``Histogram_Count_C<i>_00000`` ...).

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
from .records import PlaneRecord, different_channel_counts

# Version 3: one row per XYC plane, the channels in columns (N41). Versions 1
# and 2 had one row per plane and channel, with a ``c`` column.
SCHEMA_VERSION = "3"
OUTPUT_FILENAME = "plane_histograms.csv"

METADATA_COLUMNS = (
    "position_id",
    "image_file",
    "z",
    "t",
    "condition",
    "biological_repeat",
    "technical_repeat",
    "source_relative_path",
    "plane_id",
    "dtype",
    "bit_depth",
    "channel_count",
    "total_pixel_count",
    "bin_count",
    "schema_version",
)
# Per-channel columns before the bins, as <Category>_<Feature>; the channel
# token is appended by channel_column.
CHANNEL_FEATURES = (
    "Intensity_MinIntensity",
    "Intensity_MaxIntensity",
    "Histogram_ValidPixelCount",
    "Histogram_BinningMin",
    "Histogram_BinningMax",
)
BIN_COLUMN_PREFIX = "Histogram_Count"

# Rows of an earlier version are recognised by these columns of their header.
_EARLIER_LAYOUT = ("c", "schema_version")


class CsvFormatError(ValueError):
    """Raised when a file does not follow the plane-histogram CSV format."""


def channel_column(feature: str, channel: int) -> str:
    """Column name of a per-channel feature, e.g. ``Intensity_MinIntensity_C0``."""
    return f"{feature}_C{channel}"


def bin_column(channel: int, index: int) -> str:
    """Column name of bin ``index`` of ``channel``, e.g. ``Histogram_Count_C1_00042``."""
    return f"{BIN_COLUMN_PREFIX}_C{channel}_{index:05d}"


def channel_columns(channel: int, bin_count: int) -> list[str]:
    """Every column of ``channel`` in table order: the features, then the bins."""
    return [
        *(channel_column(feature, channel) for feature in CHANNEL_FEATURES),
        *(bin_column(channel, i) for i in range(bin_count)),
    ]


def write_plane_histograms(records: Iterable[PlaneRecord], path: str | Path) -> Path:
    """Write records to a CSV file, replacing it only once writing succeeded.

    Parameters
    ----------
    records
        Plane records; all must have the same number of channels (N43) and use
        the same bin count.
    path
        Destination file.

    Returns
    -------
    pathlib.Path
        The written file.

    Raises
    ------
    ValueError
        If there are no records, or their channel counts or bin counts differ.
    """
    records = list(records)
    if not records:
        raise ValueError("There are no planes to write.")
    channel_counts = {r.channel_count for r in records}
    if len(channel_counts) != 1:
        raise ValueError(
            different_channel_counts(
                [(r.source_relative_path, r.channel_count) for r in records], "These images"
            )
        )
    bin_counts = {r.histograms[0].bin_count for r in records}
    if len(bin_counts) != 1:
        raise ValueError(
            f"All planes in one table must use the same bin count; found {sorted(bin_counts)}."
        )
    path = Path(path)
    n_channels, n_bins = channel_counts.pop(), bin_counts.pop()
    header = [*METADATA_COLUMNS]
    for channel in range(n_channels):
        header.extend(channel_columns(channel, n_bins))
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
        If the header, schema version or a row does not match the format; if the
        table was written with schema version 1 or 2 (one row per channel); or if
        its rows have different numbers of channels.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise CsvFormatError(f"{path} is empty.") from None
        n_meta = len(METADATA_COLUMNS)
        if tuple(header[:n_meta]) != METADATA_COLUMNS:
            if all(name in header for name in _EARLIER_LAYOUT):
                first = next(reader, None)
                version = "1 or 2"
                if first is not None and len(first) == len(header):
                    version = first[header.index("schema_version")] or version
                raise CsvFormatError(_earlier_version_message(path, version))
            raise CsvFormatError(f"{path} does not have the plane-histogram columns.")
        n_channels, n_bins = _channel_layout(header[n_meta:], path)
        records, other_counts = [], []
        for line, row in enumerate(reader, 2):
            record = _record(row, n_channels, n_bins, path, line)
            if isinstance(record, PlaneRecord):
                records.append(record)
            else:
                other_counts.append(record)
        if other_counts:
            listed = [(r.source_relative_path, r.channel_count) for r in records] + other_counts
            raise CsvFormatError(f"{path}: " + different_channel_counts(listed, "Its images"))
        return records


def _channel_layout(columns: list[str], path) -> tuple[int, int]:
    # The bins of channel 0 give the bin count; the header must then hold the
    # same columns for a whole number of channels, in order.
    prefix = f"{BIN_COLUMN_PREFIX}_C0_"
    n_bins = sum(name.startswith(prefix) for name in columns)
    width = len(CHANNEL_FEATURES) + n_bins
    n_channels = len(columns) // width
    expected = [name for c in range(n_channels) for name in channel_columns(c, n_bins)]
    if n_bins == 0 or n_channels == 0 or columns != expected:
        raise CsvFormatError(f"{path} has unexpected per-channel column names.")
    return n_channels, n_bins


def _row(record: PlaneRecord) -> list[str]:
    first = record.histograms[0]
    row = [
        _text(record.position_id),
        record.image_file,
        _text(record.z),
        _text(record.t),
        record.condition,
        record.biological_repeat,
        record.technical_repeat,
        record.source_relative_path,
        record.plane_id,
        first.dtype.name,
        str(bit_depth(first.dtype)),
        str(record.channel_count),
        str(first.total_pixel_count),
        str(first.bin_count),
        SCHEMA_VERSION,
    ]
    for hist in record.histograms:
        row.extend(
            (
                _text(hist.pixel_min),
                _text(hist.pixel_max),
                str(hist.valid_pixel_count),
                _text(hist.binning_min),
                _text(hist.binning_max),
                *map(str, hist.counts.tolist()),
            )
        )
    return row


def _record(
    row: list[str], n_channels: int, n_bins: int, path, line: int
) -> PlaneRecord | tuple[str, int]:
    """Read one row; a row with another channel count is returned as ``(path, count)``."""
    n_meta = len(METADATA_COLUMNS)
    width = len(CHANNEL_FEATURES) + n_bins
    if len(row) != n_meta + n_channels * width:
        raise CsvFormatError(f"{path}, line {line}: expected {n_meta + n_channels * width} values.")
    meta = dict(zip(METADATA_COLUMNS, row[:n_meta], strict=True))
    if meta["schema_version"] != SCHEMA_VERSION:
        if meta["schema_version"] in ("1", "2"):
            raise CsvFormatError(_earlier_version_message(path, meta["schema_version"]))
        raise CsvFormatError(
            f"{path}, line {line}: schema version {meta['schema_version']!r} is not supported "
            f"(this version reads {SCHEMA_VERSION!r})."
        )
    try:
        channel_count = int(meta["channel_count"])
        if channel_count != n_channels:
            return meta["source_relative_path"], channel_count
        dtype = check_supported(meta["dtype"])
        number = int if is_integer(dtype) else float
        histograms = []
        for channel in range(n_channels):
            values = row[n_meta + channel * width : n_meta + (channel + 1) * width]
            pixel_min, pixel_max, valid, lo, hi = values[: len(CHANNEL_FEATURES)]
            histogram = PlaneHistogram(
                dtype=dtype,
                bin_count=int(meta["bin_count"]),
                binning_min=_value(lo, number),
                binning_max=_value(hi, number),
                counts=np.array(values[len(CHANNEL_FEATURES) :], dtype=np.int64),
                total_pixel_count=int(meta["total_pixel_count"]),
                valid_pixel_count=int(valid),
            )
            extrema = (_value(pixel_min, number), _value(pixel_max, number))
            if (
                histogram.bin_count != n_bins
                or int(histogram.counts.sum()) != histogram.valid_pixel_count
                or extrema != (histogram.pixel_min, histogram.pixel_max)
            ):
                raise CsvFormatError(
                    f"{path}, line {line}: the bin counts of channel {channel} are inconsistent."
                )
            histograms.append(histogram)
        return PlaneRecord(
            source_relative_path=meta["source_relative_path"],
            condition=meta["condition"],
            biological_repeat=meta["biological_repeat"],
            technical_repeat=meta["technical_repeat"],
            histograms=histograms,
            z=int(meta["z"]),
            t=int(meta["t"]),
            position_id=meta["position_id"] or None,
        )
    except CsvFormatError:
        raise
    except ValueError as error:
        raise CsvFormatError(f"{path}, line {line}: {error}") from error


def _earlier_version_message(path, version: str) -> str:
    # The message of contract §9.1, after the file's path.
    return (
        f"{path}: This plane_histograms.csv was written with schema version {version}, one row "
        "per channel. Run the preprocessing brick again on this experiment: it rewrites the "
        f"table with the channels in columns (schema version {SCHEMA_VERSION})."
    )


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _value(text: str, number: type):
    return None if text == "" else number(text)
