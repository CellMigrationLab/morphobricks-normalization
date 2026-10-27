"""Small simulated experiments for tests.

Fluorescence-like images: background plus blob-shaped cells, with shot and
read noise, and brightness differences between conditions and repeats. Kept
small (64 x 64) so the whole suite runs in seconds. The full-size study lives
in ``experiments/binning_study``.
"""

from __future__ import annotations

import functools

import numpy as np

from morphobricks_normalization import SINGLE, PlaneRecord, compute_plane_histogram


def simulate_plane(rng, dtype, shape=(64, 64), brightness=1.0, n_cells=6):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    signal = np.full(shape, 60.0)
    for _ in range(n_cells):
        cy, cx = rng.uniform(0, shape[0]), rng.uniform(0, shape[1])
        radius = rng.uniform(3, 8)
        peak = 900 * brightness * rng.lognormal(0, 0.4)
        signal += peak * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * radius**2))
    photons = rng.poisson(signal).astype(np.float64) + rng.normal(0, 3, shape)
    if np.dtype(dtype).kind == "f":
        return (photons - 60.0).astype(dtype)  # background-subtracted, can be negative
    return np.clip(np.rint(photons + 100), 0, 65535).astype(dtype)


def simulate_experiment(
    dtype, seed=0, conditions=("control", "treated"), n_bio=2, n_tech=2, n_images=2, shape=(64, 64)
):
    """Return ``[(labels, relative_path, pixels), ...]`` for a full hierarchy."""
    rng = np.random.default_rng(seed)
    planes = []
    for k, condition in enumerate(conditions):
        for b in range(1, n_bio + 1):
            bio_factor = rng.lognormal(0, 0.3)
            for t in range(1, n_tech + 1):
                tech_factor = rng.lognormal(0, 0.1)
                for i in range(1, n_images + 1):
                    labels = (condition, f"bio_rep{b}", f"tech_rep{t}")
                    path = "/".join(labels) + f"/img_{i:02d}.tif"
                    pixels = simulate_plane(
                        rng,
                        dtype,
                        shape,
                        (1 + 0.6 * k) * bio_factor * tech_factor,
                        n_cells=int(rng.integers(3, 10)),
                    )
                    planes.append((labels, path, pixels))
    return planes


def make_records(planes, bin_count=None):
    return [
        PlaneRecord(path, *labels, histogram=compute_plane_histogram(pixels, bin_count))
        for labels, path, pixels in planes
    ]


def direct_stats(pixels, percentiles):
    """Reference statistics computed directly on pooled pixels."""
    x = np.asarray(pixels, dtype=np.float64)
    x = x[np.isfinite(x)]
    median = np.percentile(x, 50, method="inverted_cdf")
    return {
        "mean": x.mean(),
        "std": x.std(),
        "median": median,
        "mad": np.percentile(np.abs(x - median), 50, method="inverted_cdf"),
        "minimum": x.min(),
        "maximum": x.max(),
        "percentiles": {q: np.percentile(x, q, method="inverted_cdf") for q in percentiles},
    }


def simulate_design(
    seed=0,
    design=None,
    n_tech=2,
    n_images=2,
    n_channels=2,
    n_times=1,
    shape=(32, 32),
    dtype=np.uint16,
):
    """Return ``[(labels, path, c, t, pixels), ...]`` for multi-channel, multi-time images.

    ``design`` maps each condition to its biological-repeat labels (default:
    two dated repeats in ``control`` and ``treated``). Brightness differs between
    conditions, repeats and channels, so pooling the wrong planes changes the
    parameters.
    """
    rng = np.random.default_rng(seed)
    if design is None:
        design = {c: ("2024_05_01", "2024_05_02") for c in ("control", "treated")}
    planes = []
    for k, (condition, repeats) in enumerate(design.items()):
        for repeat in repeats:
            repeat_factor = rng.lognormal(0, 0.3)
            for tech in range(1, n_tech + 1):
                for i in range(1, n_images + 1):
                    labels = (condition, repeat, f"tech_rep{tech}" if n_tech > 1 else SINGLE)
                    path = "/".join(labels) + f"/img_{i:02d}.tif"
                    for c in range(n_channels):
                        for t in range(n_times):
                            brightness = (1 + 0.6 * k) * repeat_factor * (1 + c) * (1 + 0.2 * t)
                            pixels = simulate_plane(rng, dtype, shape, brightness)
                            planes.append((labels, path, c, t, pixels))
    return planes


def make_design_records(planes, bin_count=None):
    return [
        PlaneRecord(path, *labels, histogram=compute_plane_histogram(pixels, bin_count), c=c, t=t)
        for labels, path, c, t, pixels in planes
    ]


@functools.cache
def cached_design(seed=0, bin_count=None, **kwargs):
    """``(planes, records)`` of :func:`simulate_design`, computed once per argument set.

    Exact 16-bit histograms are slow to build, so parametrized tests share them.
    Callers must not modify the returned arrays.
    """
    planes = simulate_design(seed=seed, **kwargs)
    return planes, make_design_records(planes, bin_count)
