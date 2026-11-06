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

Tails (contract §1.4)
    Pixels outside a robust histogram range are kept as underflow and overflow
    counts and sums. For percentiles, the median and the MAD each tail stands
    as one (mean, count) pair among the bins; the mean and standard deviation
    use the tails' exact sums instead. Histograms without tails give the
    numbers they gave before, bit for bit.

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
from .integrity import INT64_MAX
from .records import PlaneRecord
from .sums import weighted_sum

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
        has no channel ``channel``; or if the planes hold more than ``2**63 - 1``
        finite pixels together, more than the int64 counts can hold.

    Notes
    -----
    The counts are added as integers, so ``pixel_count`` is exactly the sum of
    the planes' valid pixel counts, however large.
    """

    def __init__(
        self, histograms: Iterable[PlaneHistogram | PlaneRecord], channel: int | None = None
    ) -> None:
        values, weights = [], []
        bin_values, bin_weights = [], []
        # (count, sum, sum of squares) of every non-empty tail.
        self.tails: list[tuple[float, float, float]] = []
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
            bin_values.append(values[-1])
            bin_weights.append(weights[-1])
            for tail in (hist.underflow, hist.overflow):
                if not tail.is_empty:
                    values.append(np.array([tail.mean]))
                    weights.append(np.array([tail.count], dtype=np.int64))
                    self.tails.append((tail.count, tail.sum, tail.sum_of_squares))
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
        if sum(int(w.sum()) for w in weights) > INT64_MAX:
            raise ValueError(
                "These planes hold more than 2**63 - 1 finite pixels together, more than the "
                "pixel counts can hold. Pool fewer planes per group."
            )
        # Planes binned the same way share representative values; merging them
        # keeps the list short (a few thousand entries for typical integer data).
        self.values, self.counts = _merge(values, weights)
        self.pixel_count = int(self.counts.sum())
        self._cumulative = np.cumsum(self.counts)
        if self.tails:
            # The bins alone, for the mean and standard deviation (the tails use their sums).
            self.bin_values, self.bin_counts = _merge(bin_values, bin_weights)
        else:
            self.bin_values, self.bin_counts = self.values, self.counts

    @property
    def mean(self) -> float:
        """Count-weighted mean; pixels outside the bins count with their exact values."""
        if self.tails:
            return _mean_with_tails(self.bin_values, self.bin_counts, self.tails, self.pixel_count)
        return weighted_sum(self.counts, self.values) / self.pixel_count

    @property
    def std(self) -> float:
        """Count-weighted population standard deviation; tails count with their exact sums."""
        if self.tails:
            return _std_with_tails(
                self.bin_values, self.bin_counts, self.tails, self.pixel_count, self.mean
            )
        deviations = self.values - self.mean
        squares = weighted_sum(self.counts, deviations * deviations)
        return float(np.sqrt(squares / self.pixel_count))

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


def _merge(values: list[np.ndarray], counts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Merge (value, int64 count) pairs: sorted distinct values and their exact count sums."""
    merged, inverse = np.unique(np.concatenate(values), return_inverse=True)
    total = np.zeros(merged.size, dtype=np.int64)
    # Integer addition, never float64 weights: counts above 2**53 stay exact.
    np.add.at(total, inverse.ravel(), np.concatenate(counts))
    return merged, total


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

    Notes
    -----
    The weighted counts are float64, as the weights are fractions in general,
    so they carry float64 rounding (exact for whole counts below ``2**53``);
    ``pixel_count`` adds the parts' exact integer counts.
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
        # Tails scaled by their part's weight, for the mean and standard deviation.
        self.tails = [
            (count * float(w), total * float(w), squares * float(w))
            for part, w in zip(parts, weights, strict=True)
            for count, total, squares in part.tails
        ]
        if self.tails:
            merged, inverse = np.unique(
                np.concatenate([part.bin_values for part in parts]), return_inverse=True
            )
            scaled = np.concatenate(
                [part.bin_counts * float(w) for part, w in zip(parts, weights, strict=True)]
            )
            self.bin_values = merged
            self.bin_weights = np.bincount(inverse, weights=scaled, minlength=merged.size)
        else:
            self.bin_values, self.bin_weights = self.values, self.weights

    @property
    def mean(self) -> float:
        """Weighted mean; pixels outside the bins count with their exact values."""
        if self.tails:
            return _mean_with_tails(
                self.bin_values, self.bin_weights, self.tails, self.total_weight
            )
        return weighted_sum(self.weights, self.values) / self.total_weight

    @property
    def std(self) -> float:
        """Weighted population standard deviation; tails count with their exact sums."""
        if self.tails:
            return _std_with_tails(
                self.bin_values, self.bin_weights, self.tails, self.total_weight, self.mean
            )
        deviations = self.values - self.mean
        squares = weighted_sum(self.weights, deviations * deviations)
        return float(np.sqrt(squares / self.total_weight))

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


def _mean_with_tails(
    values: np.ndarray, weights: np.ndarray, tails: list[tuple[float, float, float]], total: float
) -> float:
    sums = [s for _, s, _ in tails]
    if not all(math.isfinite(s) for s in sums):
        # math.fsum would raise a bare ValueError or OverflowError from the standard library.
        raise ValueError(_non_finite_tail_message(sums))
    return (weighted_sum(weights, values) + math.fsum(sums)) / total


def _non_finite_tail_message(sums: list[float]) -> str:
    bad = ", ".join(
        f"tail {number} of {len(sums)} has sum {s!r}"
        for number, s in enumerate(sums, 1)
        if not math.isfinite(s)
    )
    opposing = any(s == math.inf for s in sums) and any(s == -math.inf for s in sums)
    reason = (
        "infinite sums of opposite sign have no mean" if opposing else "a sum that is not finite"
    )
    return (
        f"The mean cannot be pooled: {bad} ({reason}). The pixels outside the robust range "
        "are too large for float64 sums. Use a method that does not need the mean, or "
        "another range."
    )


def _std_with_tails(
    values: np.ndarray,
    weights: np.ndarray,
    tails: list[tuple[float, float, float]],
    total: float,
    mean: float,
) -> float:
    deviations = values - mean
    squares = weighted_sum(weights, deviations * deviations)
    for count, s, s2 in tails:
        # Each tail adds its spread around its own mean and its mean's distance
        # from the overall mean; the spread never goes negative through rounding.
        tail_mean = s / count
        squares += max(s2 - s * tail_mean, 0.0) + count * (tail_mean - mean) ** 2
    return float(np.sqrt(squares / total))


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
