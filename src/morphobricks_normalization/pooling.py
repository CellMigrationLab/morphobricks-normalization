"""Group statistics from per-plane histograms.

Planes are pooled by combining pixel counts, never by averaging per-plane
statistics. Because every plane has its own bin limits, each bin becomes a
(representative value, count) pair; the pairs of all planes in a group are
merged and sorted, and the statistics are computed from that weighted list.

Definitions
    mean, std   Count-weighted mean and population standard deviation (divide
                by N) of the representative values.
    percentile  Inverse-CDF definition: the smallest representative value such
                that at least q% of pixels are at or below it (numpy
                ``method="inverted_cdf"``). The median is the 50th percentile.
    mad         Median absolute deviation from the median, unscaled.
    min, max    Exact: taken from the planes' stored extrema.

Accuracy
    Every pixel is within ``max_representative_error`` of the value standing in
    for it, so no percentile (including the median) can be off by more than that
    bound. The MAD can be off by at most twice the bound. With exact integer
    histograms (one value per bin) every statistic is exact.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .histogram import PlaneHistogram
from .records import PlaneRecord

DEFAULT_PERCENTILES = (0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9)


class NoValidPixelsError(ValueError):
    """Raised when a group contains no finite pixels at all."""


@dataclass(frozen=True)
class PooledStatistics:
    """Summary statistics of a group of planes.

    ``percentiles`` maps each requested percentile (in %) to its value.
    ``max_representative_error`` bounds the error of every percentile.
    """

    n_planes: int
    n_excluded_planes: int
    pixel_count: int
    minimum: float
    maximum: float
    mean: float
    std: float
    median: float
    mad: float
    percentiles: dict[float, float]
    max_representative_error: float


class PooledHistogram:
    """Pixel counts of several planes merged into one weighted value list.

    Parameters
    ----------
    histograms
        Plane histograms, or :class:`PlaneRecord` objects carrying them. Planes
        without finite pixels are skipped and counted in ``n_excluded_planes``.

    Raises
    ------
    NoValidPixelsError
        If none of the planes has a finite pixel.
    """

    def __init__(self, histograms: Iterable[PlaneHistogram | PlaneRecord]):
        values, weights = [], []
        self.n_planes = 0
        self.n_excluded_planes = 0
        self.minimum = np.inf
        self.maximum = -np.inf
        self.max_representative_error = 0.0
        for item in histograms:
            hist = item.histogram if isinstance(item, PlaneRecord) else item
            if hist.is_empty:
                self.n_excluded_planes += 1
                continue
            self.n_planes += 1
            occupied = hist.counts > 0
            values.append(hist.representatives[occupied])
            weights.append(hist.counts[occupied])
            self.minimum = min(self.minimum, float(hist.pixel_min))
            self.maximum = max(self.maximum, float(hist.pixel_max))
            self.max_representative_error = max(
                self.max_representative_error, hist.max_representative_error
            )
        if self.n_planes == 0:
            raise NoValidPixelsError(
                "None of the selected planes contains finite pixel values, so no statistics "
                "can be calculated for this group."
            )
        # Planes binned the same way share representative values; merging them
        # keeps the list short (a few thousand entries for typical integer data).
        merged, inverse = np.unique(np.concatenate(values), return_inverse=True)
        self.values = merged
        self.counts = np.bincount(inverse, weights=np.concatenate(weights)).astype(np.int64)
        self.pixel_count = int(self.counts.sum())
        self._cumulative = np.cumsum(self.counts)

    @property
    def mean(self) -> float:
        """Count-weighted mean."""
        return float(np.dot(self.counts, self.values) / self.pixel_count)

    @property
    def std(self) -> float:
        """Count-weighted population standard deviation."""
        deviations = self.values - self.mean
        return float(np.sqrt(np.dot(self.counts, deviations * deviations) / self.pixel_count))

    def percentile(self, q: float) -> float:
        """Inverse-CDF percentile, ``q`` in percent (0-100)."""
        if not 0 <= q <= 100:
            raise ValueError(f"Percentiles must be between 0 and 100; got {q}.")
        return float(self.values[_inverse_cdf_index(self._cumulative, self.pixel_count, q)])

    @property
    def median(self) -> float:
        """50th percentile."""
        return self.percentile(50)

    @property
    def mad(self) -> float:
        """Median absolute deviation from the median (unscaled)."""
        deviations = np.abs(self.values - self.median)
        order = np.argsort(deviations, kind="stable")
        cumulative = np.cumsum(self.counts[order])
        return float(deviations[order][_inverse_cdf_index(cumulative, self.pixel_count, 50)])

    def summary(self, percentiles: Sequence[float] = DEFAULT_PERCENTILES) -> PooledStatistics:
        """Return the standard set of statistics for this group."""
        return PooledStatistics(
            n_planes=self.n_planes,
            n_excluded_planes=self.n_excluded_planes,
            pixel_count=self.pixel_count,
            minimum=self.minimum,
            maximum=self.maximum,
            mean=self.mean,
            std=self.std,
            median=self.median,
            mad=self.mad,
            percentiles={float(q): self.percentile(q) for q in percentiles},
            max_representative_error=self.max_representative_error,
        )


def pooled_statistics(
    histograms: Iterable[PlaneHistogram | PlaneRecord],
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
) -> PooledStatistics:
    """Pool a group of planes and return its summary statistics."""
    return PooledHistogram(histograms).summary(percentiles)


def _inverse_cdf_index(cumulative: np.ndarray, total: int, q: float) -> int:
    rank = max(1, int(np.ceil(q / 100 * total)))
    return int(np.searchsorted(cumulative, rank, side="left"))
