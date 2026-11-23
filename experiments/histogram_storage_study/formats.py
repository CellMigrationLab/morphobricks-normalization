"""The storage formats compared, each a writer and a reader of plane records (not for merge).

Every reader returns ``PlaneRecord`` objects whose ``PlaneHistogram`` is built with the
repository's own type, so every invariant of ``integrity.py`` is checked on read as the CSV
reader does. The plane columns and the 11 per-channel feature cells are the CSV's exact text
(``csv_io``'s formatting: ``repr`` floats), so only the bin counts change representation.

=================  ==============================================================================
``csv_dense``      today: ``write_plane_histograms`` / ``read_plane_histogram_table`` (schema v5)
``csv_sparse``     schema v5 with one ``Histogram_Counts_<ID>`` cell per channel instead of the
                   bin columns, in the sparse grammar of ADR 0004 (``index:count;...``, zero
                   bins left out, ``empty`` for none); no new dependency
``parquet_long``   one long table (plane, channel, bin, count), zero bins left out, pyarrow, zstd;
                   the plane and feature cells as CSV text in the file's key-value metadata
``npz_trimmed``    per channel plane the counts from its first to its last occupied bin, all
                   concatenated (int64) with start and length arrays, ``np.savez_compressed``
``zarr_trimmed``   the same three arrays in a zarr v3 group (default zstd codec)
=================  ==============================================================================

For exact integer histograms (one bin per value, min-max range) bin ``i`` holds the value
``Histogram_BinningMin + i``, so (plane, bin, count) is the (plane, value, count) table.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np

from morphobricks_normalization import (
    PlaneHistogram,
    PlaneRecord,
    read_plane_histogram_table,
    write_plane_histograms,
)
from morphobricks_normalization.csv_io import (
    CHANNEL_FEATURES,
    CHANNELS_COLUMN,
    METADATA_COLUMNS,
    SCHEMA_VERSION,
    _tail,
    _tail_text,
    _text,
    _value,
    channel_column,
)
from morphobricks_normalization.dtypes import bit_depth, check_supported, is_integer

csv.field_size_limit(2**31 - 1)
FORMATS = ("csv_dense", "csv_sparse", "parquet_long", "npz_trimmed", "zarr_trimmed")
SUFFIX = {
    "csv_dense": ".csv",
    "csv_sparse": ".csv",
    "parquet_long": ".parquet",
    "npz_trimmed": ".npz",
    "zarr_trimmed": ".zarr",
}
SPARSE_FEATURE = "Histogram_Counts"


# ----------------------------------------------------------------------------- shared text cells
def _plane_cells(record: PlaneRecord) -> list[str]:
    first = record.histograms[0]
    return [
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
        ";".join(record.channels),
    ]


def _feature_cells(hist: PlaneHistogram) -> list[str]:
    return [
        _text(hist.pixel_min),
        _text(hist.pixel_max),
        str(hist.valid_pixel_count),
        _text(hist.binning_min),
        _text(hist.binning_max),
        *_tail_text(hist.underflow),
        *_tail_text(hist.overflow),
    ]


def _meta_header(ids: tuple[str, ...]) -> list[str]:
    head = [*METADATA_COLUMNS, CHANNELS_COLUMN]
    for channel in ids:
        head += [channel_column(f, channel) for f in CHANNEL_FEATURES]
    return head


def _meta_rows(records: list[PlaneRecord], ids: tuple[str, ...]) -> list[list[str]]:
    rows = []
    for record in records:
        row = _plane_cells(record)
        for channel in ids:
            if record.has_channel(channel):
                row += _feature_cells(record.channel_histogram(channel))
            else:
                row += [""] * len(CHANNEL_FEATURES)
        rows.append(row)
    return rows


def _meta_text(records: list[PlaneRecord], ids: tuple[str, ...]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_meta_header(ids))
    writer.writerows(_meta_rows(records, ids))
    return buffer.getvalue()


def _parse_meta(text: str) -> tuple[tuple[str, ...], list[list[str]]]:
    rows = list(csv.reader(io.StringIO(text)))
    header = rows[0]
    n = len(METADATA_COLUMNS) + 1
    ids = tuple(header[k][len("Intensity_MinIntensity_") :] for k in range(n, len(header), 11))
    return ids, rows[1:]


def _histogram(features: list[str], counts: np.ndarray, meta: dict[str, str]) -> PlaneHistogram:
    dtype = check_supported(meta["dtype"])
    number = int if is_integer(dtype) else float
    pixel_min, pixel_max, valid, lo, hi = features[:5]
    return PlaneHistogram(
        dtype=dtype,
        bin_count=int(meta["bin_count"]),
        binning_min=_value(lo, number),
        binning_max=_value(hi, number),
        counts=counts,
        total_pixel_count=int(meta["total_pixel_count"]),
        valid_pixel_count=int(valid),
        pixel_min=_value(pixel_min, number),
        pixel_max=_value(pixel_max, number),
        underflow=_tail(features[5:8]),
        overflow=_tail(features[8:11]),
    )


def _records_from(
    ids: tuple[str, ...], rows: list[list[str]], counts_of
) -> list[PlaneRecord]:
    """Build records from meta rows; ``counts_of(row_index, channel, n_bins)`` gives counts."""
    n = len(METADATA_COLUMNS) + 1
    records = []
    for i, row in enumerate(rows):
        meta = dict(zip((*METADATA_COLUMNS, CHANNELS_COLUMN), row[:n], strict=True))
        if meta["schema_version"] != SCHEMA_VERSION:
            raise ValueError("schema version")
        present = tuple(meta[CHANNELS_COLUMN].split(";"))
        n_bins = int(meta["bin_count"])
        hists = {}
        for k, channel in enumerate(ids):
            if channel in present:
                features = row[n + 11 * k : n + 11 * (k + 1)]
                hists[channel] = _histogram(features, counts_of(i, channel, n_bins), meta)
        record = PlaneRecord(
            source_relative_path=meta["source_relative_path"],
            condition=meta["condition"],
            biological_repeat=meta["biological_repeat"],
            technical_repeat=meta["technical_repeat"],
            histograms=[hists[c] for c in present],
            z=int(meta["z"]),
            t=int(meta["t"]),
            position_id=meta["position_id"],
            channels=present,
        )
        if record.plane_id != meta["plane_id"]:
            raise ValueError("plane_id")
        records.append(record)
    return records


def _entries(records: list[PlaneRecord], ids: tuple[str, ...]):
    """(row index, channel index, histogram) of every present channel, in table order."""
    for i, record in enumerate(records):
        for k, channel in enumerate(ids):
            if record.has_channel(channel):
                yield i, k, record.channel_histogram(channel)


# ----------------------------------------------------------------------------------- csv_dense
def write_csv_dense(records, path, ids):
    write_plane_histograms(records, path, channel_ids=ids)


def read_csv_dense(path):
    return read_plane_histogram_table(path).records


# ---------------------------------------------------------------------------------- csv_sparse
def sparse_text(counts: np.ndarray) -> str:
    """ADR 0004's sparse grammar: ``index:count`` pairs joined by ``;``, or ``empty``."""
    index = np.flatnonzero(counts)
    if index.size == 0:
        return "empty"
    return ";".join(map("{}:{}".format, index.tolist(), counts[index].tolist()))


def parse_sparse(text: str, n_bins: int) -> np.ndarray:
    """Dense int64 counts from the sparse grammar, refusing what ADR 0004's reader refuses."""
    counts = np.zeros(n_bins, dtype=np.int64)
    if text == "empty":
        return counts
    pairs = np.array(text.replace(":", ";").split(";"), dtype=np.int64).reshape(-1, 2)
    index, values = pairs[:, 0], pairs[:, 1]
    if (
        index[0] < 0
        or index[-1] >= n_bins
        or (np.diff(index) <= 0).any()
        or (values <= 0).any()
    ):
        raise ValueError("sparse counts: index out of range, not increasing, or count <= 0")
    counts[index] = values
    return counts


def write_csv_sparse(records, path, ids):
    header = [*METADATA_COLUMNS, CHANNELS_COLUMN]
    for channel in ids:
        header += [channel_column(f, channel) for f in CHANNEL_FEATURES]
        header.append(channel_column(SPARSE_FEATURE, channel))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for record in records:
            row = _plane_cells(record)
            for channel in ids:
                if record.has_channel(channel):
                    hist = record.channel_histogram(channel)
                    row += _feature_cells(hist)
                    row.append(sparse_text(hist.counts))
                else:
                    row += [""] * (len(CHANNEL_FEATURES) + 1)
            writer.writerow(row)


def read_csv_sparse(path):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        n = len(METADATA_COLUMNS) + 1
        ids = tuple(header[k][len("Intensity_MinIntensity_") :] for k in range(n, len(header), 12))
        rows, sparse = [], []
        for row in reader:
            meta, cells = row[:n], []
            texts = []
            for k in range(len(ids)):
                block = row[n + 12 * k : n + 12 * (k + 1)]
                cells += block[:11]
                texts.append(block[11])
            rows.append(meta + cells)
            sparse.append(texts)
    index = {c: k for k, c in enumerate(ids)}
    return _records_from(ids, rows, lambda i, c, nb: parse_sparse(sparse[i][index[c]], nb))


# -------------------------------------------------------------------------------- parquet_long
def write_parquet_long(records, path, ids, compression="zstd"):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows, chans, bins, counts = [], [], [], []
    for i, k, hist in _entries(records, ids):
        index = np.flatnonzero(hist.counts)
        rows.append(np.full(index.size, i, dtype=np.int32))
        chans.append(np.full(index.size, k, dtype=np.int32))
        bins.append(index.astype(np.int32))
        counts.append(hist.counts[index])
    plane_ids = pa.array([r.plane_id for r in records], type=pa.string())
    table = pa.table(
        {
            "plane_id": pa.DictionaryArray.from_arrays(np.concatenate(rows), plane_ids),
            "channel": pa.DictionaryArray.from_arrays(
                np.concatenate(chans), pa.array(list(ids), type=pa.string())
            ),
            "bin": np.concatenate(bins),
            "count": np.concatenate(counts).astype(np.int64),
        }
    )
    meta = {b"plane_histograms_meta": _meta_text(records, ids).encode()}
    table = table.replace_schema_metadata(meta)
    pq.write_table(table, path, compression=compression)


def read_parquet_long(path):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    ids, rows = _parse_meta(table.schema.metadata[b"plane_histograms_meta"].decode())
    plane = table.column("plane_id").combine_chunks()
    channel = table.column("channel").combine_chunks()
    row_index = plane.indices.to_numpy()
    chan_names = channel.dictionary.to_pylist()
    chan_index = channel.indices.to_numpy()
    bins = table.column("bin").to_numpy()
    counts = table.column("count").to_numpy()
    plane_names = plane.dictionary.to_pylist()
    if plane_names != [row[8] for row in rows]:
        raise ValueError("plane_id dictionary does not match the plane rows")
    key = row_index.astype(np.int64) * len(chan_names) + chan_index
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], key.size]
    where = {int(key[s]): (s, e) for s, e in zip(starts, ends, strict=True)}
    by_name = {c: k for k, c in enumerate(chan_names)}

    def counts_of(i, c, nb):
        dense = np.zeros(nb, dtype=np.int64)
        span = where.get(i * len(chan_names) + by_name[c])
        if span is not None:
            s, e = span
            b = bins[s:e]
            if (np.diff(b) <= 0).any() or b[0] < 0 or b[-1] >= nb or (counts[s:e] <= 0).any():
                raise ValueError("long table: bins out of range or not increasing")
            dense[b] = counts[s:e]
        return dense

    return _records_from(ids, rows, counts_of)


# ---------------------------------------------------------------------------- trimmed (npz/zarr)
def _trimmed(records, ids):
    starts, lengths, parts = [], [], []
    for _, _, hist in _entries(records, ids):
        index = np.flatnonzero(hist.counts)
        first, last = (int(index[0]), int(index[-1])) if index.size else (0, -1)
        starts.append(first)
        lengths.append(last - first + 1)
        parts.append(hist.counts[first : last + 1])
    return (
        np.concatenate(parts).astype(np.int64),
        np.array(starts, dtype=np.int64),
        np.array(lengths, dtype=np.int64),
    )


def _records_from_trimmed(ids, rows, counts, starts, lengths):
    n = len(METADATA_COLUMNS) + 1
    order = {}
    j = 0
    for i, row in enumerate(rows):
        present = set(row[n - 1].split(";"))
        for channel in ids:
            if channel in present:
                order[(i, channel)] = j
                j += 1
    offsets = np.r_[0, np.cumsum(lengths)]
    if offsets[-1] != counts.size or len(order) != starts.size:
        raise ValueError("trimmed arrays do not match the plane rows")

    def counts_of(i, c, nb):
        k = order[(i, c)]
        dense = np.zeros(nb, dtype=np.int64)
        s, length = int(starts[k]), int(lengths[k])
        if s < 0 or s + length > nb:
            raise ValueError("trimmed counts outside the bins")
        dense[s : s + length] = counts[offsets[k] : offsets[k] + length]
        return dense

    return _records_from(ids, rows, counts_of)


def write_npz_trimmed(records, path, ids):
    counts, starts, lengths = _trimmed(records, ids)
    meta = np.array(_meta_text(records, ids))
    with open(path, "wb") as fh:
        np.savez_compressed(fh, counts=counts, starts=starts, lengths=lengths, meta=meta)


def read_npz_trimmed(path):
    with np.load(path, allow_pickle=False) as z:
        ids, rows = _parse_meta(str(z["meta"]))
        return _records_from_trimmed(ids, rows, z["counts"], z["starts"], z["lengths"])


def write_zarr_trimmed(records, path, ids):
    import zarr

    counts, starts, lengths = _trimmed(records, ids)
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    for name, array in (("counts", counts), ("starts", starts), ("lengths", lengths)):
        arr = group.create_array(
            name, shape=array.shape, dtype=array.dtype, chunks=(min(max(array.size, 1), 1 << 20),)
        )
        arr[...] = array
    group.attrs["plane_histograms_meta"] = json.dumps(_meta_text(records, ids))


def read_zarr_trimmed(path):
    import zarr

    group = zarr.open_group(str(path), mode="r")
    ids, rows = _parse_meta(json.loads(group.attrs["plane_histograms_meta"]))
    arrays = [np.asarray(group[name][...]) for name in ("counts", "starts", "lengths")]
    return _records_from_trimmed(ids, rows, *arrays)


WRITERS = {
    "csv_dense": write_csv_dense,
    "csv_sparse": write_csv_sparse,
    "parquet_long": write_parquet_long,
    "npz_trimmed": write_npz_trimmed,
    "zarr_trimmed": write_zarr_trimmed,
}
READERS = {
    "csv_dense": read_csv_dense,
    "csv_sparse": read_csv_sparse,
    "parquet_long": read_parquet_long,
    "npz_trimmed": read_npz_trimmed,
    "zarr_trimmed": read_zarr_trimmed,
}


def size_of(path: Path) -> tuple[int, int]:
    """Bytes on disk (apparent size) and number of files."""
    path = Path(path)
    if path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file()]
        return sum(p.stat().st_size for p in files), len(files)
    return path.stat().st_size, 1


def same_records(a: list[PlaneRecord], b: list[PlaneRecord]) -> list[str]:
    """Differences between two record lists, every histogram field compared exactly (bits)."""
    out = []
    if len(a) != len(b):
        return [f"{len(a)} vs {len(b)} records"]
    for ra, rb in zip(a, b, strict=True):
        for name in ("source_relative_path", "condition", "biological_repeat",
                     "technical_repeat", "z", "t", "position_id", "channels"):
            if getattr(ra, name) != getattr(rb, name):
                out.append(f"{ra.plane_id}: {name}")
        for c in ra.channels:
            ha, hb = ra.channel_histogram(c), rb.channel_histogram(c)
            for name in ("dtype", "bin_count", "total_pixel_count", "valid_pixel_count"):
                if getattr(ha, name) != getattr(hb, name):
                    out.append(f"{ra.plane_id} {c}: {name}")
            for name in ("binning_min", "binning_max", "pixel_min", "pixel_max"):
                va, vb = getattr(ha, name), getattr(hb, name)
                if type(va) is not type(vb) or repr(va) != repr(vb):
                    out.append(f"{ra.plane_id} {c}: {name} {va!r} {vb!r}")
            if ha.counts.dtype != hb.counts.dtype or not np.array_equal(ha.counts, hb.counts):
                out.append(f"{ra.plane_id} {c}: counts")
            for tail in ("underflow", "overflow"):
                ta, tb = getattr(ha, tail), getattr(hb, tail)
                if (ta.count, repr(ta.mean), repr(ta.std)) != (tb.count, repr(tb.mean), repr(tb.std)):
                    out.append(f"{ra.plane_id} {c}: {tail}")
    return out
