"""Read and write the ``plane_histograms.csv`` table.

One row per XYC plane: every channel of one image at one Z and T position
(contract §6.2, N41). The plane columns come first: the shared MorphoBricks
columns (see ``docs/contract.md``), then ``dtype``, ``bit_depth``,
``channel_count``, ``total_pixel_count``, ``bin_count`` and ``schema_version``.
Then, for each channel ``i`` in turn, its columns named as in CellProfiler
(N42): ``Intensity_MinIntensity_C<i>``, ``Intensity_MaxIntensity_C<i>``,
``Histogram_ValidPixelCount_C<i>``, ``Histogram_BinningMin_C<i>``,
``Histogram_BinningMax_C<i>``, the underflow and overflow columns of schema
version 4 (``Histogram_UnderflowCount_C<i>``, ``Histogram_UnderflowSum_C<i>``,
``Histogram_UnderflowSumOfSquares_C<i>`` and the same three for ``Overflow``,
contract §1.4) and one column per bin (``Histogram_Count_C<i>_00000`` ...).

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
from .histogram import EMPTY_TAIL, HistogramTail, PlaneHistogram
from .records import PlaneRecord, different_channel_counts

# Version 4: the underflow and overflow columns of the robust histogram range
# (N56). Version 3 put one row per XYC plane, the channels in columns (N41);
# versions 1 and 2 had one row per plane and channel, with a ``c`` column.
SCHEMA_VERSION = "4"
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
    "Histogram_UnderflowCount",
    "Histogram_UnderflowSum",
    "Histogram_UnderflowSumOfSquares",
    "Histogram_OverflowCount",
    "Histogram_OverflowSum",
    "Histogram_OverflowSumOfSquares",
)
BIN_COLUMN_PREFIX = "Histogram_Count"
# The per-channel features of schema version 3, before the tails, to recognise its tables.
_VERSION_3_FEATURES = CHANNEL_FEATURES[:5]

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
        If the header, schema version or a row does not match the format, including
        a row whose ``plane_id``, ``image_file`` or ``bit_depth`` differs from what its
        path, coordinates and pixel type give; if the table was written with an
        earlier schema version (1 and 2 had one row per channel, 3 no underflow and
        overflow columns); or if its rows have different numbers of channels.
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
        if _channel_layout(header[n_meta:], _VERSION_3_FEATURES) is not None:
            raise CsvFormatError(_earlier_version_message(path, "3"))
        layout = _channel_layout(header[n_meta:], CHANNEL_FEATURES)
        if layout is None:
            raise CsvFormatError(f"{path} has unexpected per-channel column names.")
        n_channels, n_bins = layout
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


def _channel_layout(columns: list[str], features: tuple[str, ...]) -> tuple[int, int] | None:
    # The bins of channel 0 give the bin count; the header must then hold the
    # same columns for a whole number of channels, in order. None otherwise.
    prefix = f"{BIN_COLUMN_PREFIX}_C0_"
    n_bins = sum(name.startswith(prefix) for name in columns)
    width = len(features) + n_bins
    n_channels = len(columns) // width
    expected = [
        name
        for c in range(n_channels)
        for name in (
            *(channel_column(feature, c) for feature in features),
            *(bin_column(c, i) for i in range(n_bins)),
        )
    ]
    if n_bins == 0 or n_channels == 0 or columns != expected:
        return None
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
                *_tail_text(hist.underflow),
                *_tail_text(hist.overflow),
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
        if meta["schema_version"] in ("1", "2", "3"):
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
            pixel_min, pixel_max, valid, lo, hi = values[:5]
            underflow, overflow = _tail(values[5:8]), _tail(values[8:11])
            histogram = PlaneHistogram(
                dtype=dtype,
                bin_count=int(meta["bin_count"]),
                binning_min=_value(lo, number),
                binning_max=_value(hi, number),
                counts=np.array(values[len(CHANNEL_FEATURES) :], dtype=np.int64),
                total_pixel_count=int(meta["total_pixel_count"]),
                valid_pixel_count=int(valid),
                pixel_min=_value(pixel_min, number),
                pixel_max=_value(pixel_max, number),
                underflow=underflow,
                overflow=overflow,
            )
            in_bins = int(histogram.counts.sum())
            if (
                histogram.bin_count != n_bins
                or in_bins + underflow.count + overflow.count != histogram.valid_pixel_count
                or not _consistent_range(histogram)
            ):
                raise CsvFormatError(
                    f"{path}, line {line}: the bin counts of channel {channel} are inconsistent."
                )
            histograms.append(histogram)
        record = PlaneRecord(
            source_relative_path=meta["source_relative_path"],
            condition=meta["condition"],
            biological_repeat=meta["biological_repeat"],
            technical_repeat=meta["technical_repeat"],
            histograms=histograms,
            z=int(meta["z"]),
            t=int(meta["t"]),
            position_id=meta["position_id"] or None,
        )
        # These columns are written from the others, never read into the record, so a
        # row whose stored values disagree would otherwise pass with another identity.
        for column, implied in (
            ("plane_id", record.plane_id),
            ("image_file", record.image_file),
            ("bit_depth", str(bit_depth(dtype))),
        ):
            if meta[column] != implied:
                raise CsvFormatError(
                    f"{path}, line {line}: its {column} is {meta[column]!r}, but its other "
                    f"columns give {implied!r}. The table was changed after it was written; "
                    "run the preprocessing brick again on this experiment to rewrite it."
                )
        return record
    except CsvFormatError:
        raise
    except ValueError as error:
        raise CsvFormatError(f"{path}, line {line}: {error}") from error


def _tail_text(tail: HistogramTail) -> tuple[str, str, str]:
    return str(tail.count), repr(float(tail.sum)), repr(float(tail.sum_of_squares))


def _tail(texts: list[str]) -> HistogramTail:
    count, total, squares = int(texts[0]), float(texts[1]), float(texts[2])
    if count < 0:
        raise ValueError(f"a pixel count cannot be negative; got {count}.")
    if count == 0:
        if total != 0 or squares != 0:
            raise ValueError("an empty underflow or overflow must have zero sums.")
        return EMPTY_TAIL
    return HistogramTail(count, total, squares)


def _consistent_range(histogram: PlaneHistogram) -> bool:
    """Check that a binning limit differs from the extremum only where a tail lies beyond it."""
    if histogram.binning_min is None or histogram.pixel_min is None:
        return (
            histogram.binning_min is None
            and histogram.binning_max is None
            and histogram.pixel_min is None
            and histogram.pixel_max is None
            and not histogram.has_tails
        )
    low = (histogram.pixel_min == histogram.binning_min) == histogram.underflow.is_empty
    high = (histogram.pixel_max == histogram.binning_max) == histogram.overflow.is_empty
    ordered = (
        histogram.pixel_min <= histogram.binning_min <= histogram.binning_max <= histogram.pixel_max
    )
    return low and high and ordered


def _earlier_version_message(path, version: str) -> str:
    # The messages of contract §9.1, after the file's path.
    if version == "3":
        return (
            f"{path}: This plane_histograms.csv was written with schema version 3, without the "
            "underflow and overflow columns. Run the preprocessing brick again on this "
            "experiment: it rewrites the table with these columns (schema version "
            f"{SCHEMA_VERSION})."
        )
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
