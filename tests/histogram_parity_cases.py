"""Inputs of the v0.7.0 histogram parity table of tests/test_histogram_parity.py.

``tests/data/histogram_v070_parity.json`` freezes, as one SHA-256 per plane (over all its
bin counts, both ranges and both block splits), every field v0.7.0 (commit ``c6a9f9d``) gave
for these planes: the ``PlaneHistogram`` (limits, counts, pixel counts, both tails), its
``representatives`` and ``max_representative_error`` as raw float64 bits, its integer bin
limits, ``is_exact`` and the ``RangeCheck``. The per-bin work of integer planes moved from
Python loops to NumPy int64 after v0.7.0; every one of these fields must stay bitwise equal.
The cases are the differential check of the preprocessing study
``experiments/histogram_numba_study/diffcheck.py`` (preprocessing PR #318), made smaller:
the ten pixel types, NaN and +/-inf, subnormals, constant, empty and dtype-extreme planes,
bin-edge float grids, several bin counts, the min-max and robust ranges, whole planes and
three blocks, plus 64-bit ranges on both sides of the int64 limits of ``integer_bins``. To
record the table again::

    git worktree add /tmp/v070 c6a9f9d
    PYTHONPATH=/tmp/v070/src <venv>/bin/python tests/histogram_parity_cases.py \\
        > tests/data/histogram_v070_parity.json
    git worktree remove /tmp/v070

The script prints the path of the ``histogram`` module it used on standard error: check that
it is the worktree's.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import struct
import sys
from collections.abc import Iterator
from typing import Any

import numpy as np

from morphobricks_normalization import histogram as H
from morphobricks_normalization.dtypes import is_integer
from morphobricks_normalization.ranges import HistogramSettings

BASE_COMMIT = "c6a9f9d"
SEED = 12345
DTYPES = (
    "uint8",
    "int8",
    "uint16",
    "int16",
    "uint32",
    "int32",
    "uint64",
    "int64",
    "float32",
    "float64",
)
SHAPES = ((1,), (7, 5), (120, 97))
BIN_COUNTS = (None, 1, 3, 256, 500, 10_000)
KINDS = 6
# 64-bit planes beyond 2**51 (full range, near the minimum or the maximum) take the exact
# Fraction bound of every bin, unchanged since v0.7.0 and slow with many bins: those planes
# get up to 500 bins; the 64-bit planes below 2**51 keep every bin count.
BEYOND_FLOAT_KINDS = (0, 3, 5)
BEYOND_FLOAT_BIN_COUNTS = (1, 3, 256, 500)


def _bits(x: Any) -> str:
    """Return ``x`` as text that tells every float bit and every array byte apart."""
    if isinstance(x, np.ndarray):
        return f"{x.dtype.str}:{x.shape}:{hashlib.sha256(x.tobytes()).hexdigest()}"
    if isinstance(x, float | np.floating):
        return f"{type(x).__name__}:{struct.pack('<d', float(x)).hex()}"
    return f"{type(x).__name__}:{x!r}"


def fingerprint(histogram: H.PlaneHistogram, check: Any) -> str:
    """Return every field of a histogram (and its range check) as text."""
    h = histogram
    parts = [
        h.dtype.str,
        repr(h.bin_count),
        _bits(h.binning_min),
        _bits(h.binning_max),
        _bits(h.counts),
        repr(h.total_pixel_count),
        repr(h.valid_pixel_count),
        _bits(h.pixel_min),
        _bits(h.pixel_max),
        *(_bits(v) for t in (h.underflow, h.overflow) for v in (t.count, t.mean, t.std)),
        _bits(h.representatives),
        _bits(h.max_representative_error),
        repr(h.is_exact),
    ]
    if is_integer(h.dtype) and not h.is_empty:
        flat = list(itertools.chain.from_iterable(h.integer_bin_limits()))
        values = np.array(flat)  # int64 or uint64; object (as text) only beyond both
        parts.append(_bits(values) if values.dtype != object else " ".join(map(str, flat)))
        parts.append(repr(sorted(t.__name__ for t in set(map(type, flat)))))
    if check is not None:
        parts.extend(f"{k}={_bits(v)}" for k, v in sorted(vars(check).items()))
    return "|".join(parts)


def _integer_plane(rng: np.random.Generator, d: np.dtype, shape: tuple, kind: int) -> np.ndarray:
    info = np.iinfo(d)
    lo, hi = [
        (info.min, info.max),
        (0, 10),
        (5, 5),
        (int(info.min), int(info.min) + 3),
        (max(info.min, -1000), min(info.max, 70000)),
        (info.max - 2, info.max),
    ][kind]
    return rng.integers(lo, hi, size=shape, dtype=d, endpoint=True)


def _float_plane(rng: np.random.Generator, d: np.dtype, shape: tuple, kind: int) -> np.ndarray:
    scale = (1, 1e-310, 0, 1e300, 1e5, 1)[kind]  # normal, subnormal, constant, overflowing
    with np.errstate(over="ignore"):  # float32 * 1e300 is inf, a non-finite pixel on purpose
        a = (rng.standard_normal(shape) * scale).astype(d)
    # No plane mixes -0.0 and +0.0: which of the two NumPy's min and max return then depends
    # on the platform (seen on Windows and macOS runners), so its bits are not a fixed
    # reference. Adding +0.0 turns -0.0 into +0.0 and changes no other value.
    a = a + d.type(0.0)
    if kind == 5:
        a.flat[::7] = np.nan
        a.flat[1::11] = np.inf
        a.flat[2::13] = -np.inf
    if kind == 4:
        a.flat[0] = np.finfo(d).max
        a.flat[-1] = np.finfo(d).min
    if kind == 2 and a.size > 3:
        a.flat[:3] = np.nan
    return a


def planes() -> Iterator[tuple[str, np.ndarray, tuple]]:
    """Yield ``(case ID, plane, bin counts)``."""
    rng = np.random.default_rng(SEED)
    for name in DTYPES:
        d = np.dtype(name)
        make = _integer_plane if d.kind in "iu" else _float_plane
        for shape in SHAPES:
            for kind in range(KINDS):
                label = f"{name}_{'x'.join(map(str, shape))}_k{kind}"
                bins = BIN_COUNTS
                if d.itemsize == 8 and d.kind in "iu" and kind in BEYOND_FLOAT_KINDS:
                    bins = BEYOND_FLOAT_BIN_COUNTS
                yield label, make(rng, d, shape, kind), bins
        yield f"{name}_empty", np.zeros(0, d), BIN_COUNTS
    yield "float32_all_nan", np.full((4, 4), np.nan, np.float32), BIN_COUNTS
    negative_zero = np.full((5, 3), -0.0)
    negative_zero.flat[::4] = np.nan
    yield "float64_negative_zero", negative_zero, BIN_COUNTS
    # Values on or next to bin edges: integer-valued floats over [0, n] and k * step grids.
    for name in ("float32", "float64"):
        for n in (3, 256, 500, 10_000):
            for lo, hi in ((0.0, float(n)), (-1.5, 7.25), (1e-3, 1.0), (-1e6, 3e6), (0.1, 0.7)):
                grid = lo + (hi - lo) * np.arange(n + 1) / n
                edges = np.concatenate(
                    [grid, np.nextafter(grid, -np.inf), np.nextafter(grid, np.inf)]
                )
                a = np.clip(edges, lo, hi).astype(name)
                a = np.concatenate([a, np.array([np.nan, np.inf, -np.inf], name)])
                yield f"{name}_edges_{n}_{lo}_{hi}", a, (n,)
    # 64-bit ranges around the int64 limits of integer_bins: wider than 2**24 values with
    # int64 bin starts; below 2**51 but too wide for int64 starts with 65,536 bins (not with
    # 500); bin count times width just below and at 2**62; at 2**51 and beyond.
    for name, low, high, bin_counts in (
        ("int64", -(2**40), 2**40, (None, 3, 500)),
        ("uint64", 2**45, 2**45 + 2**33, (None, 3, 500)),
        ("int64", -(2**50), 2**50, (None, 3, 500)),
        ("int64", -(2**51) + 1, 2**51 - 1, (None, 3, 500)),
        ("int64", 0, 2**50 - 2, (4096,)),
        ("int64", 0, 2**50 - 1, (4096,)),
        ("uint64", 2**51 - 5, 2**51 + 5, (None, 3, 500)),
        ("uint64", 2**63, 2**64 - 1, (3, 500)),
    ):
        a = rng.integers(low, high, size=(31, 29), dtype=name, endpoint=True)
        a.flat[:2] = (low, high)
        yield f"{name}_range_{low}_{high}", a, bin_counts


def variants(plane: np.ndarray, bin_counts: tuple) -> Iterator[tuple[int | None, bool, int]]:
    """Yield ``(bin count, robust, blocks)`` for every allowed bin count of a plane."""
    for bins in bin_counts:
        try:
            H.validate_bin_count(
                plane.dtype, H.default_bin_count(plane.dtype) if bins is None else bins
            )
        except ValueError:
            continue
        for robust in (False, True):
            for blocks in (1, 3):
                yield bins, robust, blocks


def outputs(plane: np.ndarray, bins: int | None, robust: bool, blocks: int) -> str:
    """Return both histograms of a variant and its range check as text."""
    parts = np.array_split(plane.ravel(), blocks)
    settings = HistogramSettings(histogram_range="robust") if robust else None
    histogram = H.compute_block_histogram(lambda: iter(parts), plane.dtype, bins, settings=settings)
    checked, check = H.histogram_with_range_check(
        lambda: iter(parts), plane.dtype, bins, settings=settings
    )
    variant = f"bins={bins} robust={robust} blocks={blocks}"
    return f"{variant}#{fingerprint(histogram, None)}#{fingerprint(checked, check)}"


def table() -> dict[str, list]:
    """Return ``[variants, SHA-256 of all their outputs]`` for every plane."""
    found = {}
    for label, plane, bin_counts in planes():
        texts = [outputs(plane, *variant) for variant in variants(plane, bin_counts)]
        found[label] = [len(texts), hashlib.sha256("\n".join(texts).encode()).hexdigest()]
    return found


if __name__ == "__main__":
    print(H.__file__, file=sys.stderr)
    record = {"base_commit": BASE_COMMIT, "numpy": np.__version__, "planes": table()}
    json.dump(record, sys.stdout, indent=0, sort_keys=True)
    sys.stdout.write("\n")
