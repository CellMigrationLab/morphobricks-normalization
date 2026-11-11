"""Read and write the ``plane_histograms.csv`` table, schema version 5 (ADR 0003).

One row per XYC plane: every channel of one image at one Z and T position
(contract §6.2, N41). The plane columns come first: the shared MorphoBricks
columns (see ``docs/contract.md``), then ``dtype``, ``bit_depth``,
``channel_count``, ``total_pixel_count``, ``bin_count`` and ``schema_version``,
then the presence column ``channels``: the canonical channel IDs of the plane, in
source order, ``;``-separated (HV1). Then, for every channel ID of the experiment
(taken from the description and given to the writer), sorted, its columns named
as in CellProfiler with the ID as the channel token (HV2):
``Intensity_MinIntensity_<ID>``, ``Intensity_MaxIntensity_<ID>``,
``Histogram_ValidPixelCount_<ID>``, ``Histogram_BinningMin_<ID>``,
``Histogram_BinningMax_<ID>``, the underflow and overflow columns
(``Histogram_UnderflowCount_<ID>``, ``Histogram_UnderflowMean_<ID>``,
``Histogram_UnderflowStd_<ID>`` and the same three for ``Overflow``,
contract §1.4) and one column per bin (``Histogram_Count_<ID>_00000`` ...).

A channel a plane does not list is *absent*: all its cells are blank, and the
reader refuses a value there (absent is not zero and not failed). A failed
measurement is not a state of the table: the writer refuses to write it at all
(HV3). A table of an earlier schema version is refused with the fix (HV4); it is
never converted.

An ID may contain ``_`` (``PAX_488``), so column names are never split on
``_``: the reader takes the IDs from the parameter-less
``Histogram_ValidPixelCount_<ID>`` columns and requires the exact layout above.

Labels are always read back as text, so folder names such as ``NA``, ``001``
or ``TRUE`` keep their exact spelling. Float values are written at full
precision and read back unchanged.
"""

from __future__ import annotations

import csv
import itertools
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .channels import check_channel_ids
from .dtypes import bit_depth, check_supported, is_integer
from .histogram import EMPTY_TAIL, HistogramTail, PlaneHistogram
from .records import (
    ChannelMeasurementError,
    ChannelMeasurementFailure,
    PlaneRecord,
    check_measurements,
    check_universe,
    duplicate_planes,
    same_channels_per_image,
)

# Version 5: channels named by canonical channel ID, with the presence column (ADR 0003).
# Version 4 added the underflow and overflow columns (N56); version 3 put one row per XYC
# plane with the channels in columns by C index (N41); versions 1 and 2 had one row per
# plane and channel, with a ``c`` column.
SCHEMA_VERSION = "5"
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
# The presence column (HV1), right after the plane columns.
CHANNELS_COLUMN = "channels"
CHANNEL_SEPARATOR = ";"
# Per-channel columns before the bins, as <Category>_<Feature>; the channel
# token is appended by channel_column.
CHANNEL_FEATURES = (
    "Intensity_MinIntensity",
    "Intensity_MaxIntensity",
    "Histogram_ValidPixelCount",
    "Histogram_BinningMin",
    "Histogram_BinningMax",
    "Histogram_UnderflowCount",
    "Histogram_UnderflowMean",
    "Histogram_UnderflowStd",
    "Histogram_OverflowCount",
    "Histogram_OverflowMean",
    "Histogram_OverflowStd",
)
BIN_COLUMN_PREFIX = "Histogram_Count"
# The parameter-less feature whose columns name the table's channel IDs (HV2).
_ID_FEATURE = "Histogram_ValidPixelCount"
# Tables of these versions are refused by their ``schema_version`` value alone, with a message
# that names the fix; no reader of their layouts is kept (they would have to be recomputed).
_EARLIER_VERSIONS = ("1", "2", "3", "4")


class CsvFormatError(ValueError):
    """Raised when a file does not follow the plane-histogram CSV format."""


class PlaneHistogramTable(NamedTuple):
    """A ``plane_histograms.csv`` read back: the header's channel IDs and the rows.

    Attributes
    ----------
    channel_ids
        Every channel ID of the header (the experiment's), sorted.
    records
        One :class:`~morphobricks_normalization.records.PlaneRecord` per row, in file
        order; each holds the channels its row lists as present.
    """

    channel_ids: tuple[str, ...]
    records: list[PlaneRecord]


def channel_column(feature: str, channel: str) -> str:
    """Column name of a per-channel feature, e.g. ``Intensity_MinIntensity_DNA``."""
    return f"{feature}_{channel}"


def bin_column(channel: str, index: int) -> str:
    """Column name of bin ``index`` of ``channel``, e.g. ``Histogram_Count_DNA_00042``."""
    return f"{BIN_COLUMN_PREFIX}_{channel}_{index:05d}"


def channel_columns(channel: str, bin_count: int) -> list[str]:
    """Every column of ``channel`` in table order: the features, then the bins."""
    return [
        *(channel_column(feature, channel) for feature in CHANNEL_FEATURES),
        *(bin_column(channel, i) for i in range(bin_count)),
    ]


def table_header(channel_ids: Iterable[str], bin_count: int) -> list[str]:
    """Return the header of a table of these channel IDs (sorted here) and bin count.

    Every writer of a parallel shard gives the experiment's full list, so shards share
    one header (ADR 0003, details).
    """
    header = [*METADATA_COLUMNS, CHANNELS_COLUMN]
    for channel in sorted(check_channel_ids(channel_ids, "The table's channel IDs")):
        header.extend(channel_columns(channel, bin_count))
    return header


def header_channels(header: list[str]) -> tuple[tuple[str, ...], int] | None:
    """Return the channel IDs (sorted) and bin count of a version-5 header, or None.

    The IDs are the tokens of the ``Histogram_ValidPixelCount_<ID>`` columns (HV2); the
    header must then be exactly :func:`table_header` of those IDs and some bin count of
    at least 1, so a bin index is the last ``_NNNNN`` of its column.

    Parameters
    ----------
    header
        The header row.

    Returns
    -------
    tuple or None
        ``(channel_ids, bin_count)``, or None when the header is not a version-5 header.
    """
    n_meta = len(METADATA_COLUMNS) + 1
    if tuple(header[: n_meta - 1]) != METADATA_COLUMNS or header[n_meta - 1 : n_meta] != [
        CHANNELS_COLUMN
    ]:
        return None
    prefix = f"{_ID_FEATURE}_"
    found = [name[len(prefix) :] for name in header[n_meta:] if name.startswith(prefix)]
    try:
        ids = check_channel_ids(found)
    except ValueError:
        return None
    columns = len(header) - n_meta
    if not ids or columns % len(ids):
        return None
    n_bins = columns // len(ids) - len(CHANNEL_FEATURES)
    if n_bins < 1 or header != table_header(ids, n_bins):
        return None
    return tuple(sorted(ids)), n_bins


def write_plane_histograms(
    records: Iterable[PlaneRecord],
    path: str | Path,
    *,
    channel_ids: Iterable[str],
    failures: Iterable[ChannelMeasurementFailure] = (),
) -> Path:
    """Write records to a CSV file, replacing it only once writing succeeded.

    Parameters
    ----------
    records
        Plane records, in the order to write; all use the same bin count. Their
        channels may differ (ADR 0003 HV1).
    path
        Destination file.
    channel_ids
        Every channel ID of the experiment, from the confirmed description; the header
        lists them sorted. Give every shard of a run the same list, so shards share one
        header. Each record's channels must be among them.
    failures
        The run's failed channel measurements (HV3). When there is any, nothing is
        written, and a file already at ``path`` (an earlier run's table) is removed, so
        that no out-of-date table is left that looks current (contract §6.2).

    Returns
    -------
    pathlib.Path
        The written file.

    Raises
    ------
    ChannelMeasurementError
        When ``failures`` is not empty, naming each failing plane and channel and the
        earlier table it removed (``removed``); checked before anything else.
    ValueError
        If there are no records, two records have the same ``plane_id``, their bin
        counts differ, ``channel_ids`` is not a list of distinct channel IDs, a record
        has a channel that is not in ``channel_ids``, or the planes of one image list
        different channels.
    """
    _refuse_failures(failures, Path(path))
    universe = check_channel_ids(channel_ids, "The table's channel IDs")
    if not universe:
        raise ValueError("Give the experiment's channel IDs (channel_ids); the list is empty.")
    records = list(records)
    if not records:
        raise ValueError("There are no planes to write.")
    bin_counts = {r.histograms[0].bin_count for r in records}
    if len(bin_counts) != 1:
        raise ValueError(
            f"All planes in one table must use the same bin count; found {sorted(bin_counts)}."
        )
    check_universe(records, set(universe))
    duplicate_planes(records, "The planes to write")
    same_channels_per_image(records, "The planes to write")
    path = Path(path)
    ids = tuple(sorted(universe))
    header = table_header(ids, bin_counts.pop())
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for record in records:
                writer.writerow(_row(record, ids))
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def _refuse_failures(failures: Iterable[ChannelMeasurementFailure], path: Path) -> None:
    """Raise for failed measurements, after removing an earlier table at ``path`` (HV3).

    An earlier run's table left in place would read as this run's, so it is removed and
    the error names it (``ChannelMeasurementError.removed``).
    """
    try:
        check_measurements(failures)
    except ChannelMeasurementError as error:
        if not path.is_file():
            raise
        path.unlink()
        raise ChannelMeasurementError(error.failures, removed=str(path)) from None


def read_plane_histograms(path: str | Path) -> list[PlaneRecord]:
    """Read the rows of a plane-histogram CSV written by :func:`write_plane_histograms`.

    The records of :func:`read_plane_histogram_table`; use that function for the
    header's channel IDs as well.

    Raises
    ------
    CsvFormatError
        As :func:`read_plane_histogram_table`.
    """
    return read_plane_histogram_table(path).records


def read_plane_histogram_table(path: str | Path) -> PlaneHistogramTable:
    """Read a plane-histogram CSV: its channel IDs and its rows.

    Raises
    ------
    CsvFormatError
        If the header, schema version or a row does not match the format, including
        a row whose ``plane_id``, ``image_file`` or ``bit_depth`` differs from what its
        path, coordinates and pixel type give, whose ``channels`` is blank, repeats an
        ID, names an ID the header does not have or does not have ``channel_count``
        entries, which has a value in a cell of a channel it does not list (absent), or
        whose statistics break the rules of :mod:`~morphobricks_normalization.integrity`;
        if a row says the table was written with an earlier schema version (1 to 4),
        which is refused by that value alone, without reading the layout, with the
        message naming version 5 and how to recompute it (HV4); if it has no rows, two
        rows with the same ``plane_id``, or planes of one image with different channels.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise CsvFormatError(f"{path} is empty.") from None
        first = next(reader, None)
        if "schema_version" in header and first is not None:
            index = header.index("schema_version")
            if index < len(first) and first[index] in _EARLIER_VERSIONS:
                raise CsvFormatError(_earlier_version_message(path, first[index]))
        reader = itertools.chain([] if first is None else [first], reader)
        if tuple(header[: len(METADATA_COLUMNS)]) != METADATA_COLUMNS:
            raise CsvFormatError(f"{path} does not have the plane-histogram columns.")
        layout = header_channels(header)
        if layout is None:
            raise CsvFormatError(
                f"{path} does not have the per-channel columns of schema version "
                f"{SCHEMA_VERSION} (the '{CHANNELS_COLUMN}' column, then for every channel ID "
                "in sorted order its features and bins). Run the preprocessing brick again on "
                "this experiment to rewrite the table."
            )
        ids, n_bins = layout
        records, lines = [], {}
        for line, row in enumerate(reader, 2):
            record = _record(row, ids, n_bins, path, line)
            if record.plane_id in lines:
                raise CsvFormatError(_duplicate_message(path, record.plane_id, lines, line))
            lines[record.plane_id] = line
            records.append(record)
        if not records:
            raise CsvFormatError(
                f"{path} has the plane-histogram columns but no rows, so it holds no planes. The "
                "table is incomplete; run the preprocessing brick again on this experiment."
            )
        try:
            same_channels_per_image(records, f"{path}: its rows")
        except ValueError as error:
            raise CsvFormatError(str(error)) from None
        return PlaneHistogramTable(ids, records)


def _row(record: PlaneRecord, ids: tuple[str, ...]) -> list[str]:
    """Return the cells of one row: the plane columns, then every channel of ``ids``."""
    first = record.histograms[0]
    row = [
        record.position_id,
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
        CHANNEL_SEPARATOR.join(record.channels),
    ]
    width = len(CHANNEL_FEATURES) + first.bin_count
    for channel in ids:
        if not record.has_channel(channel):
            row.extend([""] * width)  # absent: blank, never a zero histogram (HV1)
            continue
        hist = record.channel_histogram(channel)
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
    row: list[str], ids: tuple[str, ...], n_bins: int, path: str | Path, line: int
) -> PlaneRecord:
    """Read one row into a record of the channels it lists."""
    n_meta = len(METADATA_COLUMNS) + 1
    width = len(CHANNEL_FEATURES) + n_bins
    if len(row) != n_meta + len(ids) * width:
        raise CsvFormatError(f"{path}, line {line}: expected {n_meta + len(ids) * width} values.")
    meta = dict(zip((*METADATA_COLUMNS, CHANNELS_COLUMN), row[:n_meta], strict=True))
    if meta["schema_version"] != SCHEMA_VERSION:
        if meta["schema_version"] in _EARLIER_VERSIONS:
            raise CsvFormatError(_earlier_version_message(path, meta["schema_version"]))
        raise CsvFormatError(
            f"{path}, line {line}: schema version {meta['schema_version']!r} is not supported "
            f"(this version reads {SCHEMA_VERSION!r})."
        )
    try:
        present = _present_channels(meta, ids)
        dtype = check_supported(meta["dtype"])
        if int(meta["bin_count"]) != n_bins:
            raise ValueError(f"its bin_count is {meta['bin_count']}, but it has {n_bins} bins.")
        by_channel = {}
        for k, channel in enumerate(ids):
            values = row[n_meta + k * width : n_meta + (k + 1) * width]
            if channel not in present:
                _check_absent(values, channel, n_bins)
                continue
            try:
                total = meta["total_pixel_count"]
                by_channel[channel] = _histogram(values, dtype, n_bins, total, channel)
            except ValueError as error:
                raise CsvFormatError(
                    f"{path}, line {line}: the bin counts of channel {channel} are "
                    f"inconsistent: {error} The table was changed after it was written; run "
                    "the preprocessing brick again on this experiment to rewrite it."
                ) from error
        record = PlaneRecord(
            source_relative_path=meta["source_relative_path"],
            condition=meta["condition"],
            biological_repeat=meta["biological_repeat"],
            technical_repeat=meta["technical_repeat"],
            histograms=[by_channel[channel] for channel in present],
            z=int(meta["z"]),
            t=int(meta["t"]),
            position_id=meta["position_id"],
            channels=present,
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


def _present_channels(meta: dict[str, str], ids: tuple[str, ...]) -> tuple[str, ...]:
    """Return the row's ``channels``: present IDs in source order, checked against the header."""
    text = meta[CHANNELS_COLUMN]
    if text == "":
        raise ValueError(
            f"its {CHANNELS_COLUMN} cell is blank: every plane has at least one channel, listed "
            "by canonical channel ID."
        )
    present = check_channel_ids(text.split(CHANNEL_SEPARATOR), f"its {CHANNELS_COLUMN} cell")
    unknown = [channel for channel in present if channel not in ids]
    if unknown:
        raise ValueError(
            f"its {CHANNELS_COLUMN} cell lists {', '.join(unknown)}, which the header does not "
            f"have (it has {', '.join(ids)})."
        )
    if meta["channel_count"] != str(len(present)):
        raise ValueError(
            f"its channel_count is {meta['channel_count']!r}, but its {CHANNELS_COLUMN} cell "
            f"lists {len(present)} channel(s) ({text})."
        )
    return present


def _check_absent(values: list[str], channel: str, n_bins: int) -> None:
    """Refuse any value in the cells of a channel the row does not list (HV1)."""
    filled = [
        name
        for name, value in zip(channel_columns(channel, n_bins), values, strict=True)
        if value != ""
    ]
    if filled:
        shown = ", ".join(filled[:3]) + (f" and {len(filled) - 3} more" if len(filled) > 3 else "")
        raise ValueError(
            f"channel {channel} is absent from this plane (its {CHANNELS_COLUMN} cell does not "
            f"list it), but {shown} {'holds a value' if len(filled) == 1 else 'hold values'}. "
            "An absent channel has blank cells: it is not measured, and is never a zero or NaN "
            "histogram. The table was changed after it was written; run the preprocessing "
            "brick again on this experiment to rewrite it."
        )


def _histogram(
    values: list[str], dtype: np.dtype, n_bins: int, total: str, channel: str
) -> PlaneHistogram:
    """Read one channel's columns; the histogram checks its own invariants.

    A blank intensity limit of a channel with valid pixels is refused here, so
    that the message names its column (#47).
    """
    number = int if is_integer(dtype) else float
    pixel_min, pixel_max, valid, lo, hi = values[:5]
    blank = [
        channel_column(feature, channel)
        for feature, value in zip(CHANNEL_FEATURES[:5], values[:5], strict=True)
        if value == "" and feature != _ID_FEATURE
    ]
    if blank and int(valid) > 0:
        raise ValueError(
            f"{', '.join(blank)} {'is' if len(blank) == 1 else 'are'} blank, but the channel has "
            f"{int(valid)} valid pixel(s): a plane with valid pixels needs its four finite "
            "intensity limits, and none is filled in from another."
        )
    try:
        counts = np.array(values[len(CHANNEL_FEATURES) :], dtype=np.int64)
    except OverflowError:
        raise ValueError("a bin count is larger than 2**63 - 1.") from None
    return PlaneHistogram(
        dtype=dtype,
        bin_count=n_bins,
        binning_min=_value(lo, number),
        binning_max=_value(hi, number),
        counts=counts,
        total_pixel_count=int(total),
        valid_pixel_count=int(valid),
        pixel_min=_value(pixel_min, number),
        pixel_max=_value(pixel_max, number),
        underflow=_tail(values[5:8]),
        overflow=_tail(values[8:11]),
    )


def _duplicate_message(path: str | Path, plane_id: str, lines: dict[str, int], line: int) -> str:
    return (
        f"{path}, line {line}: plane {plane_id} is already in line {lines[plane_id]}. Every "
        "plane (an image at one Z and T position) has one row; a repeated row would count its "
        "pixels twice. Run the preprocessing brick again on this experiment to rewrite the table."
    )


def _tail_text(tail: HistogramTail) -> tuple[str, str, str]:
    return str(tail.count), repr(float(tail.mean)), repr(float(tail.std))


def _tail(texts: list[str]) -> HistogramTail:
    tail = HistogramTail(int(texts[0]), float(texts[1]), float(texts[2]))
    return EMPTY_TAIL if tail.is_empty else tail


def _earlier_version_message(path: str | Path, version: str) -> str:
    """Return the refusal of a table of an earlier schema version (contract §9.1).

    A table is recomputed, never converted: the channel IDs come from the confirmed
    description (HV4, ADR 0002 AR10).
    """
    if version == "4":
        what = "its channels named by their index in the image (C0, C1, ...)"
    elif version == "3":
        what = "its channels named by index and without the underflow and overflow columns"
    else:
        what = "one row per channel"
    return (
        f"{path}: This plane_histograms.csv was written with schema version {version}, {what}. "
        f"This version reads schema version {SCHEMA_VERSION} only, with every channel named by "
        "its canonical channel ID from the experiment description; old tables are not "
        "converted. Run the preprocessing brick again on this experiment: it recomputes the "
        f"table (schema version {SCHEMA_VERSION})."
    )


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _value(text: str, number: type) -> object:
    return None if text == "" else number(text)
