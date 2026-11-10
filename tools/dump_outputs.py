"""Dump the brick's outputs for a fixed set of valid inputs, and print their SHA-256.

Used to show that a change leaves every output for valid inputs bitwise
unchanged: run it on the base commit and on the change, in the same
environment, and compare the printed digests (or ``diff`` the dumps).

The dump covers histograms of simulated planes of every supported pixel type
(min-max and robust range, whole planes and blocks, with range checks), the
``plane_histograms.csv`` text written from them and the records read back,
pooled statistics, image summaries, intensity checks and fitted parameters for
every method and scope. Floats are written with :meth:`float.hex`, so any
change of a single bit changes the digest.

With ``--corpus`` the dump is the generated corpus instead: histograms of every pixel
type over extreme, subnormal, one-ulp, narrow, ordinary and wide ranges, at many bin counts,
whole and in blocks, min-max and robust. It uses only the public API that exists on
older commits, so the same corpus runs on the base commit (``PYTHONPATH=<base>/src``) and
on the change; the digests must be equal.

Usage::

    python tools/dump_outputs.py [--out DUMP] [--corpus]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import numpy as np

from morphobricks_normalization import (
    HistogramSettings,
    NormalizationSettings,
    PlaneHistogram,
    PlaneRecord,
    PooledHistogram,
    compute_block_histogram,
    compute_plane_histogram,
    fit,
    histogram_with_range_check,
    image_summaries,
    intensity_checks,
    read_plane_histograms,
    write_plane_histograms,
)

SEED = 20261103
ROBUST = HistogramSettings(histogram_range="robust")
# Quantiles that leave pixels of these small planes in both tails.
TIGHT = HistogramSettings(histogram_range="robust", robust_quantiles=(2.0, 97.0))
BIN_COUNTS = (16, 256)
CORPUS_BIN_COUNTS = (2, 3, 4, 5, 7, 16, 64, 256, 1024)
CORPUS_INTEGER_BIN_COUNTS = (1, 3, 16, 256)
SMALLEST_SUBNORMAL = 5e-324
SMALLEST_NORMAL = float(np.finfo(np.float64).tiny)
FLOAT64_MAX = float(np.finfo(np.float64).max)
METHODS = ("minmax", "percentile", "zscore", "robust_zscore")


def text(value: object) -> str:
    """Return a bit-exact text for numbers, arrays and containers.

    Parameters
    ----------
    value
        Any value of an output.

    Returns
    -------
    str
        Floats as :meth:`float.hex`, arrays with their dtype and every element.
    """
    if isinstance(value, float | np.floating):
        return float(value).hex()
    if isinstance(value, np.ndarray):
        return f"{value.dtype.str}{list(value.shape)}[{','.join(map(text, value.tolist()))}]"
    if isinstance(value, list | tuple):
        return "(" + ",".join(text(item) for item in value) + ")"
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{text(item)}" for key, item in value.items()) + "}"
    return repr(value)


def histogram_text(hist: PlaneHistogram) -> str:
    """Return every stored field and derived property of a histogram.

    Parameters
    ----------
    hist
        The histogram.

    Returns
    -------
    str
        One line.
    """
    fields = (
        hist.dtype.str,
        hist.bin_count,
        hist.binning_min,
        hist.binning_max,
        hist.counts,
        hist.total_pixel_count,
        hist.valid_pixel_count,
        hist.pixel_min,
        hist.pixel_max,
        (hist.underflow.count, hist.underflow.sum, hist.underflow.sum_of_squares),
        (hist.overflow.count, hist.overflow.sum, hist.overflow.sum_of_squares),
        hist.representatives,
        hist.max_representative_error,
        hist.is_exact,
    )
    return text(fields)


def planes() -> Iterator[tuple[str, np.ndarray]]:
    """Yield named simulated planes of every supported pixel type.

    Yields
    ------
    tuple of str and numpy.ndarray
        A name and a 2D plane.
    """
    rng = np.random.default_rng(SEED)
    for dtype in ("uint8", "uint16", "int8", "int16", "uint32", "int32", "int64", "uint64"):
        info = np.iinfo(dtype)
        narrow = rng.integers(max(info.min, -3), min(info.max, 9), (32, 48), endpoint=True)
        yield f"{dtype}-narrow", narrow.astype(dtype)
        wide = rng.integers(info.min, info.max, (32, 48), endpoint=True, dtype=dtype)
        yield f"{dtype}-wide", wide
        yield f"{dtype}-constant", np.full((8, 8), info.max // 3, dtype)
    for dtype in ("float32", "float64"):
        plane = rng.normal(100.0, 15.0, (32, 48)).astype(dtype)
        plane[0, :5] = (-1e4, 1e4, -3e3, 7e3, 5e2)
        yield f"{dtype}-normal", plane
        holes = plane.copy()
        holes[1, :6] = (np.nan, np.inf, -np.inf, np.nan, np.nan, np.nan)
        yield f"{dtype}-non-finite", holes
        yield f"{dtype}-constant", np.full((8, 8), 2.5, dtype)
        yield f"{dtype}-all-nan", np.full((4, 4), np.nan, dtype)


def histograms(name: str, plane: np.ndarray) -> Iterator[tuple[str, PlaneHistogram]]:
    """Yield every histogram of one plane, and dump the range checks on the way.

    Parameters
    ----------
    name
        The plane's name.
    plane
        The plane.

    Yields
    ------
    tuple of str and PlaneHistogram
        A label and a histogram.
    """
    blocks = [plane[:7], plane[7:]]
    for bins in BIN_COUNTS:
        for label, settings in (("minmax", None), ("robust", ROBUST), ("tight", TIGHT)):
            whole = compute_plane_histogram(plane, bins, settings=settings)
            yield f"{name}/{bins}/{label}/plane", whole
            split = compute_block_histogram(lambda: blocks, plane.dtype, bins, settings=settings)
            yield f"{name}/{bins}/{label}/blocks", split
            checked, check = histogram_with_range_check(
                lambda: blocks, plane.dtype, bins, settings=settings
            )
            yield f"{name}/{bins}/{label}/checked", checked
            yield f"{name}/{bins}/{label}/check={text(check and asdict(check))}", checked


def pooled_text(hists: list[PlaneHistogram]) -> str:
    """Return the pooled statistics of some histograms, or the error they raise.

    Parameters
    ----------
    hists
        Histograms of one pixel type and bin count.

    Returns
    -------
    str
        One line.
    """
    try:
        pooled = PooledHistogram(hists)
        values = (
            pooled.pixel_count,
            pooled.counts,
            pooled.mean,
            pooled.std,
            pooled.median,
            pooled.mad,
        )
        quantiles = [pooled.percentile(q) for q in (0, 0.01, 1, 50, 99.8, 100)]
        linear = [pooled.linear_percentile(q) for q in (0, 1, 50, 99.8, 100)]
        return text((values, quantiles, linear, asdict(pooled.summary())))
    except ValueError as error:
        return f"{type(error).__name__}: {error}"


def fit_text(records: list[PlaneRecord]) -> Iterator[str]:
    """Yield the fitted parameters of every method and scope, or the error raised.

    Parameters
    ----------
    records
        Records of one experiment.

    Yields
    ------
    str
        One line per fit.
    """
    for method in METHODS:
        for scope, pooling in (("image", "pixels"), ("biological_repeat", "conditions")):
            settings = NormalizationSettings(method=method, scope=scope, pooling=pooling)
            try:
                parameters = fit(records, settings, group="g")
                issues = [asdict(issue) for issue in parameters.issues]
                yield f"fit {method} {scope}: {text(parameters.to_columns())} {text(issues)}"
            except ValueError as error:
                yield f"fit {method} {scope}: {type(error).__name__}: {error}"


def experiment_lines(dtype: str, hists: list[PlaneHistogram], folder: Path) -> Iterator[str]:
    """Write records of one pixel type, read them back, and yield what derives from them.

    Parameters
    ----------
    dtype
        The records' pixel type.
    hists
        One-channel histograms of that type, all with one bin count.
    folder
        A temporary folder for the CSV file.

    Yields
    ------
    str
        Lines of the dump.
    """
    records = [
        PlaneRecord(
            f"c{i % 2}/r{i % 3}/img{i}.tif",
            f"c{i % 2}",
            f"r{i % 3}",
            "t",
            [h],
            t=i % 2,
            position_id=f"img{i}",
        )
        for i, h in enumerate(hists)
    ]
    path = write_plane_histograms(records, folder / f"{dtype}.csv")
    yield f"csv {dtype} sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}"
    back = read_plane_histograms(path)
    for record in back:
        yield f"read {record.plane_id} {histogram_text(record.histograms[0])}"
    yield f"pooled {dtype} {pooled_text([r.histograms[0] for r in back])}"
    summaries = image_summaries(back)
    for summary in summaries:
        yield f"summary {text(summary.to_row())}"
    issues = intensity_checks(summaries)
    yield f"checks {dtype} {text([asdict(issue) for issue in issues])}"
    yield from fit_text(back)


def dump_lines(folder: Path) -> Iterator[str]:
    """Yield every line of the dump.

    Parameters
    ----------
    folder
        A temporary folder for the CSV files.

    Yields
    ------
    str
        Lines of the dump.
    """
    experiments: dict[tuple[str, int, str], list[PlaneHistogram]] = {}
    for name, plane in planes():
        for label, hist in histograms(name, plane):
            yield f"{label} {histogram_text(hist)}"
            bins, kind, how = label.split("/")[1:4]
            if how == "plane" and not hist.is_empty:
                experiments.setdefault((plane.dtype.name, int(bins), kind), []).append(hist)
    for (dtype, bins, kind), hists in experiments.items():
        yield from experiment_lines(f"{dtype}-{bins}-{kind}", hists, folder)


def one_ulp(x: float) -> tuple[float, float]:
    """Return ``x`` and the next float64 above it, or below it at the maximum.

    Parameters
    ----------
    x
        A finite float64.

    Returns
    -------
    tuple of float
        ``(x, next float)`` in increasing order.
    """
    with np.errstate(over="ignore"):  # the next float above the maximum is infinity
        up = float(np.nextafter(x, np.inf))
    return (x, up) if np.isfinite(up) else (float(np.nextafter(x, 0.0)), x)


def corpus_ranges() -> list[tuple[float, float]]:
    """Return the float64 ranges of the corpus, from one ulp to the whole float64 range.

    Returns
    -------
    list of tuple of float
        ``(lo, hi)`` with ``lo < hi``.
    """
    tiny, sub, big = SMALLEST_NORMAL, SMALLEST_SUBNORMAL, FLOAT64_MAX
    pairs = [
        (0.0, sub),
        (0.0, 2 * sub),
        (0.0, 3 * sub),
        (-sub, sub),
        (-sub, 0.0),
        (1e-320, 2e-320),
        (0.0, 1e-310),
        (0.0, tiny),
        (-tiny, tiny),
        (tiny, 2 * tiny),
        (0.0, 1e-306),
        (0.0, 1e-100),
        (0.0, 1.0),
        (-1.0, 1.0),
        (0.0, 255.0),
        (-1e4, 1e4),
        (1e-30, 1e30),
        (-1e30, 1e30),
        (0.0, 1e300),
        (-1e308, 1e308),
        (0.0, big),
        (-big, 0.0),
        (-big, big),
    ]
    for x in (sub, 1e-320, tiny, 1e-300, 0.1, 1.0, -2.0, 3.7, 1e10, 1e300, 1e308, -big, big):
        pairs.append(one_ulp(x))
    pairs.append((1.0, 1.0 + 4 * 2.0**-52))
    return pairs


def corpus_plane(lo: float, hi: float, dtype: str) -> np.ndarray:
    """Return a plane whose finite range is ``[lo, hi]``, with values spread inside it.

    Parameters
    ----------
    lo, hi
        The float64 range, ``lo < hi``.
    dtype
        ``"float32"`` or ``"float64"``; for float32 the range is cast and may collapse.

    Returns
    -------
    numpy.ndarray
        A 2D plane of ``dtype`` with the endpoints, their neighbours and an even spread.
    """
    fractions = np.linspace(0.0, 1.0, 40)
    values = np.clip(lo * (1 - fractions) + hi * fractions, lo, hi)
    near = [lo, hi, float(np.nextafter(lo, hi)), float(np.nextafter(hi, lo))]
    flat = np.concatenate([values, near, [0.0 if lo <= 0.0 <= hi else lo]])
    with np.errstate(over="ignore"):
        cast = flat.astype(dtype)
    cast = cast[np.isfinite(cast)]
    return np.resize(cast, (8, max(cast.size // 8, 1))) if cast.size >= 8 else cast.reshape(1, -1)


def corpus_planes() -> Iterator[tuple[str, np.ndarray]]:
    """Yield the corpus planes: float ranges of both widths, and extreme integer planes.

    Yields
    ------
    tuple of str and numpy.ndarray
        A name and a 2D plane.
    """
    for lo, hi in corpus_ranges():
        for dtype in ("float64", "float32"):
            yield f"{dtype}[{lo.hex()},{hi.hex()}]", corpus_plane(lo, hi, dtype)
    rng = np.random.default_rng(SEED)
    for dtype in ("uint8", "uint16", "int8", "int16", "uint32", "int32", "int64", "uint64"):
        info = np.iinfo(dtype)
        for low, high in ((info.min, info.min + 1), (info.max - 1, info.max), (info.min, info.max)):
            extremes = np.array([low, high, low, high, (low + high) // 2], dtype=dtype)
            yield f"{dtype}[{low},{high}]", np.resize(extremes, (4, 5))
        mid = rng.integers(max(info.min, -5), min(info.max, 40), (6, 7), endpoint=True)
        yield f"{dtype}-small", mid.astype(dtype)


def corpus_histogram_lines(
    name: str, plane: np.ndarray, bins: int, label: str, settings: HistogramSettings | None
) -> Iterator[str]:
    """Yield the lines of one plane read whole and in two blocks, or the error raised.

    Parameters
    ----------
    name
        The plane's name.
    plane
        The plane.
    bins
        Bin count.
    label
        Name of the range setting.
    settings
        The range setting, or None for min-max.

    Yields
    ------
    str
        A line for the whole plane and a line for the blocks.
    """
    half = max(plane.shape[0] // 2, 1)
    blocks = [plane[:half], plane[half:]]
    for how in ("plane", "blocks"):
        try:
            if how == "plane":
                hist = compute_plane_histogram(plane, bins, settings=settings)
            else:
                hist = compute_block_histogram(lambda: blocks, plane.dtype, bins, settings=settings)
            yield f"{name}/{bins}/{label}/{how} {histogram_text(hist)}"
        except ValueError as error:
            yield f"{name}/{bins}/{label}/{how} {type(error).__name__}: {error}"


def corpus_lines() -> Iterator[str]:
    """Yield the histogram lines of the corpus.

    Yields
    ------
    str
        One line per plane, bin count, range setting and way of reading the plane.
    """
    for name, plane in corpus_planes():
        counts = CORPUS_BIN_COUNTS if plane.dtype.kind == "f" else CORPUS_INTEGER_BIN_COUNTS
        for bins in counts:
            for label, settings in (("minmax", None), ("robust", ROBUST), ("tight", TIGHT)):
                yield from corpus_histogram_lines(name, plane, bins, label, settings)


def main(argv: list[str] | None = None) -> int:
    """Write the dump and print its SHA-256.

    Parameters
    ----------
    argv
        Command-line arguments; ``--out`` names the dump file, ``--corpus`` selects the corpus.

    Returns
    -------
    int
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="where to write the dump (default: not kept)")
    parser.add_argument(
        "--corpus",
        action="store_true",
        help="dump the generated corpus of narrow and extreme ranges",
    )
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as folder:
        lines = corpus_lines() if args.corpus else dump_lines(Path(folder))
        dump = "".join(line + "\n" for line in lines).encode("utf-8")
    if args.out is not None:
        args.out.write_bytes(dump)
    lines = dump.count(b"\n")
    print(f"{hashlib.sha256(dump).hexdigest()}  ({lines} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
