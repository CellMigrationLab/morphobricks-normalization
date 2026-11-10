"""Pooled means, standard deviations and z-score fits never depend on CPU threads (#67).

``np.dot`` let the BLAS library split a long sum across its threads, so the last bits of a
z-score fit changed with ``OPENBLAS_NUM_THREADS``/``OMP_NUM_THREADS`` (OpenBLAS threads a dot
product of more than 10,000 elements). The sums are now correctly rounded sums of the products.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from morphobricks_normalization import (
    HistogramSettings,
    NormalizationSettings,
    PlaneRecord,
    PooledHistogram,
    compute_plane_histogram,
    fit,
)
from morphobricks_normalization.pooling import WeightedHistogram
from morphobricks_normalization.sums import weighted_sum

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
TIGHT = HistogramSettings(histogram_range="robust", robust_quantiles=(2.0, 97.0))


def _records(settings: HistogramSettings | None = None) -> list[PlaneRecord]:
    """Planes whose pooled list holds far more than 10,000 distinct values."""
    rng = np.random.default_rng(67)
    records = []
    for i in range(60):
        plane = rng.gamma(2.0, 300.0 + 7 * i, size=(64, 64))
        hist = compute_plane_histogram(plane, 256, settings=settings)
        records.append(
            PlaneRecord(f"c/r/img{i}.tif", "c", f"r{i % 2}", "t", [hist], position_id="p")
        )
    return records


def _numbers() -> dict[str, str]:
    """Every sum that once went through ``np.dot``, as bit-exact text."""
    plain = PooledHistogram(_records())
    tails = PooledHistogram(_records(TIGHT))
    assert plain.values.size > 10_000 and tails.tails
    weighted = WeightedHistogram([plain, tails], [1.0, 0.37])
    numbers = {
        "mean": plain.mean,
        "std": plain.std,
        "tails mean": tails.mean,
        "tails std": tails.std,
        "weighted mean": weighted.mean,
        "weighted std": weighted.std,
    }
    for scope in ("image", "biological_repeat"):
        parameters = fit(_records(), NormalizationSettings(method="zscore", scope=scope))
        numbers[f"fit {scope}"] = repr(parameters.to_columns())
    return {k: v.hex() if isinstance(v, float) else v for k, v in numbers.items()}


def _exact(weights: np.ndarray, values: np.ndarray) -> float:
    """The correctly rounded sum of the float64 products, with exact fractions."""
    products = np.multiply(weights, values, dtype=np.float64)
    return float(sum(Fraction(p) for p in products.tolist()))


def test_pooled_sums_are_correctly_rounded() -> None:
    """The mean and spread sums equal the exact sums of their products, rounded once."""
    pooled = PooledHistogram(_records())
    total = pooled.pixel_count
    assert pooled.mean == _exact(pooled.counts, pooled.values) / total
    deviations = pooled.values - pooled.mean
    assert pooled.std == float(np.sqrt(_exact(pooled.counts, deviations * deviations) / total))


def test_weighted_sum_is_exactly_rounded_and_order_free() -> None:
    """``weighted_sum`` is the correctly rounded sum, whatever the order of the terms."""
    rng = np.random.default_rng(3)
    counts = rng.integers(1, 10**6, 50_000)
    values = rng.normal(0.0, 1e4, 50_000)
    assert weighted_sum(counts, values) == _exact(counts, values)
    order = rng.permutation(counts.size)
    assert weighted_sum(counts[order], values[order]) == weighted_sum(counts, values)


def test_fits_are_bitwise_equal_with_one_and_many_threads(tmp_path: Path) -> None:
    """Every pooled sum and z-score fit has the same bits with 1 and 4 BLAS/OpenMP threads."""
    script = tmp_path / "fit_bits.py"  # not numbers.py, which would hide the standard module
    script.write_text(
        "import json, sys\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from test_sums import _numbers\n"
        "print(json.dumps(_numbers()))\n",
        encoding="utf-8",
    )
    results = []
    for threads in ("1", "4"):
        environment = {**os.environ, **dict.fromkeys(THREAD_VARIABLES, threads)}
        done = subprocess.run(
            [sys.executable, str(script)],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert done.returncode == 0, done.stderr
        results.append(json.loads(done.stdout))
    assert results[0] == results[1]
    assert results[0] == _numbers()


def test_non_finite_sums_stay_nan_or_inf_as_with_np_dot() -> None:
    """``math.fsum`` raises on ``inf - inf`` and overflow; the sums keep np.dot's nan and inf."""
    big = np.finfo(np.float64).max
    with np.errstate(over="ignore", invalid="ignore"):
        assert np.isnan(weighted_sum(np.array([1, 1]), np.array([np.inf, -np.inf])))
        assert np.isnan(weighted_sum(np.array([4, 4]), np.array([big, -big])))
        assert weighted_sum(np.array([1, 1]), np.array([big, big * 0.9])) == np.inf
        assert weighted_sum(np.array([2, 1]), np.array([-np.inf, 1.0])) == -np.inf
        assert np.isnan(weighted_sum(np.array([1, 1]), np.array([np.nan, 1.0])))
    assert weighted_sum(np.array([], dtype=np.int64), np.array([])) == 0.0


def test_huge_planes_are_refused_by_name_not_by_fsum() -> None:
    """A float64 plane whose sums overflow gets the z-score refusal, not a bare fsum error."""
    big = np.finfo(np.float64).max
    for plane in (np.array([[big, -big], [big, -big]]), np.array([[big, big * 0.9], [1.0, 2.0]])):
        with np.errstate(over="ignore", invalid="ignore"):
            try:
                fit(plane, NormalizationSettings(method="zscore"))
            except ValueError as error:
                assert "z-score normalization cannot be used" in str(error)
            else:
                raise AssertionError("expected the z-score refusal")
