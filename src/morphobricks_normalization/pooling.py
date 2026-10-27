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
    linear      NumPy's default linear-interpolation percentile, used only by
                the RGB method (contract §3.3).
    mad        Median absolute deviation from the median, unscaled.
    min, max    Exact: taken from the planes' stored extrema.

Accuracy
    Every pixel is within ``max_representative_error`` of the value standing in
    for it, so no percentile (including the median) can be off by more than that
    bound. The MAD can be off by at most twice the bound. With exact integer
    histograms (one value per bin) every statistic is exact.
"""

from __future__ import annotations

import math
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
        Plane histograms, or :class:`PlaneRecord` objects carrying one histogram
        per channel. Planes without finite pixels are skipped and counted in
        ``n_excluded_planes``.
    channel
        For records, the channel to pool (zero-based); it may be left out when
        every record has one channel. Plane histograms are used as given.

    Raises
    ------
    NoValidPixelsError
        If none of the planes has a finite pixel.
    ValueError
        If a record has more than one channel and ``channel`` is not given, or
        has no channel ``channel``.
    """

    def __init__(
        self, histograms: Iterable[PlaneHistogram | PlaneRecord], channel: int | None = None
    ):
        values, weights = [], []
        self.n_planes = 0
        self.n_excluded_planes = 0
        self.minimum = np.inf
        self.maximum = -np.inf
        self.max_representative_error = 0.0
        for item in histograms:
            hist = _channel_histogram(item, channel) if isinstance(item, PlaneRecord) else item
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

    def linear_percentile(self, q: float) -> float:
        """Percentile with NumPy's default linear interpolation, ``q`` in percent (0-100).

        The value ``np.percentile(pixels, q)`` gives for the pooled pixels,
        bit for bit when every histogram is exact (one value per bin); used by
        the RGB method (contract §3.3). The fluorescence methods use
        :meth:`percentile`.
        """
        if not 0 <= q <= 100:
            raise ValueError(f"Percentiles must be between 0 and 100; got {q}.")
        return _linear_percentile(self.values, self._cumulative, self.pixel_count, q)

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


class WeightedHistogram:
    """Several pooled histograms added with a weight each (contract §4.3).

    Every pixel of ``parts[i]`` counts ``weights[i]`` times. The statistics are
    those of :class:`PooledHistogram` with weighted counts: the percentile is
    the smallest representative whose cumulative weight reaches ``q / 100`` of
    the total. With all weights 1 the numbers equal those of a
    :class:`PooledHistogram` of the same planes exactly.

    Parameters
    ----------
    parts
        The histograms to add, for example one per condition, all of the same
        channel.
    weights
        One positive, finite weight per part.
    """

    def __init__(self, parts: Sequence[PooledHistogram], weights: Sequence[float]):
        if len(parts) == 0 or len(parts) != len(weights):
            raise ValueError("Expected one weight for each of at least one pooled histogram.")
        if not all(np.isfinite(w) and w > 0 for w in weights):
            raise ValueError(f"Weights must be positive and finite; got {list(weights)}.")
        self.n_planes = sum(part.n_planes for part in parts)
        self.n_excluded_planes = sum(part.n_excluded_planes for part in parts)
        self.pixel_count = sum(part.pixel_count for part in parts)
        self.minimum = min(part.minimum for part in parts)
        self.maximum = max(part.maximum for part in parts)
        self.max_representative_error = max(part.max_representative_error for part in parts)
        merged, inverse = np.unique(
            np.concatenate([part.values for part in parts]), return_inverse=True
        )
        scaled = np.concatenate(
            [part.counts * float(w) for part, w in zip(parts, weights, strict=True)]
        )
        self.values = merged
        self.weights = np.bincount(inverse, weights=scaled)
        self._cumulative = np.cumsum(self.weights)
        self.total_weight = float(self._cumulative[-1])

    @property
    def mean(self) -> float:
        """Weighted mean."""
        return float(np.dot(self.weights, self.values) / self.total_weight)

    @property
    def std(self) -> float:
        """Weighted population standard deviation."""
        deviations = self.values - self.mean
        return float(np.sqrt(np.dot(self.weights, deviations * deviations) / self.total_weight))

    def percentile(self, q: float) -> float:
        """Weighted inverse-CDF percentile, ``q`` in percent (0-100)."""
        if not 0 <= q <= 100:
            raise ValueError(f"Percentiles must be between 0 and 100; got {q}.")
        return float(self.values[_weighted_index(self._cumulative, self.total_weight, q)])

    def linear_percentile(self, q: float) -> float:
        """Weighted linear-interpolation percentile, ``q`` in percent (0-100) (N49).

        NumPy's linear rule with the cumulative weights as positions: the
        values are placed as if each were repeated by its weight, and the
        total weight ``W`` takes the place of the pixel count, so the result
        depends on the scale of the weights (``fit`` gives each pixel of the
        largest condition weight 1). With all weights 1 the result equals
        :meth:`PooledHistogram.linear_percentile` of the same planes exactly.
        """
        if not 0 <= q <= 100:
            raise ValueError(f"Percentiles must be between 0 and 100; got {q}.")
        return _linear_percentile(self.values, self._cumulative, self.total_weight, q)

    @property
    def median(self) -> float:
        """Weighted 50th percentile."""
        return self.percentile(50)

    @property
    def mad(self) -> float:
        """Weighted median absolute deviation from the weighted median (unscaled)."""
        deviations = np.abs(self.values - self.median)
        order = np.argsort(deviations, kind="stable")
        cumulative = np.cumsum(self.weights[order])
        return float(deviations[order][_weighted_index(cumulative, cumulative[-1], 50)])


def pooled_statistics(
    histograms: Iterable[PlaneHistogram | PlaneRecord],
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
    *,
    channel: int | None = None,
) -> PooledStatistics:
    """Pool a group of planes and return its summary statistics.

    ``channel`` selects the channel of records, as for :class:`PooledHistogram`.
    """
    return PooledHistogram(histograms, channel).summary(percentiles)


def _channel_histogram(record: PlaneRecord, channel: int | None) -> PlaneHistogram:
    if channel is None:
        if record.channel_count != 1:
            raise ValueError(
                f"Plane {record.plane_id} has {record.channel_count} channels; give the "
                f"channel to pool, from 0 to {record.channel_count - 1}."
            )
        channel = 0
    return record.channel_histogram(channel)


def _inverse_cdf_index(cumulative: np.ndarray, total: int, q: float) -> int:
    rank = max(1, int(np.ceil(q / 100 * total)))
    return int(np.searchsorted(cumulative, rank, side="left"))


def _linear_percentile(values: np.ndarray, cumulative: np.ndarray, total: float, q: float) -> float:
    # NumPy's "linear" method (Hyndman & Fan type 7), step by step as
    # numpy.lib's _quantile and _lerp compute it, so the result is bit-identical
    # to np.percentile on the pixels: the virtual index h = (N - 1) * (q / 100)
    # lies between the sorted positions k and k + 1, and the interpolation
    # starts from the nearer of the two values.
    h = (total - 1) * (q / 100)
    if h >= total - 1:
        return float(values[-1])
    k = math.floor(h)
    gamma = h - k
    # The value at zero-based sorted position i is the first whose cumulative
    # count exceeds i.
    last = len(values) - 1
    below, above = (
        values[min(int(np.searchsorted(cumulative, i, side="right")), last)] for i in (k, k + 1)
    )
    difference = above - below
    if gamma >= 0.5:
        return float(above - difference * (1 - gamma))
    return float(below + difference * gamma)


def _weighted_index(cumulative: np.ndarray, total: float, q: float) -> int:
    # With whole-number weights, "cumulative >= q/100 * total" picks the same
    # entry as the integer rank of _inverse_cdf_index, so unit weights give
    # PooledHistogram's numbers exactly.
    index = int(np.searchsorted(cumulative, q / 100 * total, side="left"))
    return min(index, len(cumulative) - 1)
