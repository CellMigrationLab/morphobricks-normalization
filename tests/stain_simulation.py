"""Synthetic stained images from the Beer-Lambert law, with their exact error bounds.

``I = white * 10 ** (-C @ M)`` per pixel, with the rows of ``M`` unit optical-density vectors.
Rounding to uint8 moves each value by at most 0.5, so the optical density of a rounded value
``r`` (at least 1) is off by at most ``log10(r / (r - 0.5))`` per channel (the larger of the
two one-sided bounds, ``log10((r + 0.5) / r)`` being smaller). Concentrations are
``OD @ inv(M)``, so stain ``j`` is off by at most ``sum_c bound_c * |inv(M)[c, j]|``.
"""

import numpy as np

WHITE = (240.0, 235.0, 245.0)
# Unit-free optical-density directions (R, G, B absorbance), normalised where used.
HEMATOXYLIN = (0.65, 0.70, 0.29)  # Ruifrok-Johnston
EOSIN = (0.07, 0.99, 0.11)  # Ruifrok-Johnston
DAB = (0.27, 0.57, 0.78)  # Ruifrok-Johnston
PSR = (0.10, 0.80, 0.55)  # picrosirius red: absorbs green and blue, passes red
COUNTERSTAIN = HEMATOXYLIN


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def unit_matrix(*vectors):
    """Rows: the unit vectors; with two, the third is their normalised cross product."""
    rows = [unit(v) for v in vectors]
    if len(rows) == 2:
        rows.append(unit(np.cross(rows[0], rows[1])))
    return np.array(rows)


def concentrations(seed, shape, stains, high=0.6):
    """Non-negative concentrations, uniform in [0, high], one per stain."""
    return np.random.default_rng(seed).uniform(0.0, high, (*shape, stains))


def beer_lambert(amounts, matrix, white=WHITE):
    """Float intensities of concentrations ``amounts`` (..., k) with the first k rows of M."""
    od = amounts @ matrix[: amounts.shape[-1]]
    return np.asarray(white) * 10.0 ** (-od)


def to_uint8(intensities):
    assert intensities.min() >= 1.5 and intensities.max() <= 254.5, "no floor, no saturation"
    return np.rint(intensities).astype(np.uint8)


def concentration_bound(rounded, matrix):
    """Per-pixel, per-stain bound of the concentration error caused by uint8 rounding."""
    r = rounded.astype(np.float64)
    od_bound = np.log10(r / (r - 0.5))
    # float32 output and float64 arithmetic: a few float32 ulps of values below about 2.
    return od_bound @ np.abs(np.linalg.inv(matrix)) + 1e-6
