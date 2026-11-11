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
        For records, the canonical ID of the channel to pool; it may be left out when
        every record has one channel. Every record must have that channel: only the
        same channel ID is pooled, and a plane without it is refused, never skipped.
        Plane histograms are used as given.

    Raises
    ------
    NoValidPixelsError
        If none of the planes has a finite pixel.
    ValueError
        If a record has more than one channel and ``channel`` is not given, or
        does not have channel ``channel`` (it is absent there); or if the planes
        hold more than ``2**63 - 1`` finite pixels together, more than the int64
        counts can hold.

    Notes
    -----
    The counts are added as integers, so ``pixel_count`` is exactly the sum of
    the planes' valid pixel counts, however large.
    """

    def __init__(
        self, histograms: Iterable[PlaneHistogram | PlaneRecord], channel: str | None = None
    ) -> None:
        values, weights = [], []
        bin_values, bin_weights = [], []
        # (count, mean, standard deviation) of every non-empty tail.
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
                    self.tails.append((tail.count, tail.mean, tail.std))
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
            # The bins alone, for the mean and standard deviation (the tails use their moments).
            self.bin_values, self.bin_counts = _merge(bin_values, bin_weights)
        else:
            self.bin_values, self.bin_counts = self.values, self.counts

    @property
    def mean(self) -> float:
        """Count-weighted mean; pixels outside the bins count by their tail's mean."""
        if self.tails:
            return _mean_with_tails(self.bin_values, self.bin_counts, self.tails, self.pixel_count)
        return _mean_with_tails(self.values, self.counts, [], self.pixel_count)

    @property
    def std(self) -> float:
        """Count-weighted population standard deviation; tails count with their moments."""
        if self.tails:
            return _std_with_tails(
                self.bin_values, self.bin_counts, self.tails, self.pixel_count, self.mean
            )
        return _std_with_tails(self.values, self.counts, [], self.pixel_count, self.mean)

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
        # Tails with their count scaled by their part's weight (mean and spread unchanged).
        self.tails = [
            (count * float(w), mean, std)
            for part, w in zip(parts, weights, strict=True)
            for count, mean, std in part.tails
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
        """Weighted mean; pixels outside the bins count by their tail's mean."""
        if self.tails:
            return _mean_with_tails(
                self.bin_values, self.bin_weights, self.tails, self.total_weight
            )
        return _mean_with_tails(self.values, self.weights, [], self.total_weight)

    @property
    def std(self) -> float:
        """Weighted population standard deviation; tails count with their moments."""
        if self.tails:
            return _std_with_tails(
                self.bin_values, self.bin_weights, self.tails, self.total_weight, self.mean
            )
        return _std_with_tails(self.values, self.weights, [], self.total_weight, self.mean)

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
    channel: str | None = None,
) -> PooledStatistics:
    """Pool a group of planes and return its summary statistics.

    ``channel`` selects the channel of records, as for :class:`PooledHistogram`.
    """
    return PooledHistogram(histograms, channel).summary(percentiles)


def _mean_with_tails(
    values: np.ndarray, weights: np.ndarray, tails: list[tuple[float, float, float]], total: float
) -> float:
    """Return the weighted mean of the bins and of each tail's mean, weighted by its count.

    Without tails this is the plain weighted mean, bit for bit (a sum that overflows gives
    ``inf`` or ``nan``, which the methods refuse by name). A tail's mean is finite, so when
    the products of the tails overflow although every value is finite (pixels near the
    float64 limit at both ends), the sum is redone with every value divided by a power of
    two near the largest, which is exact, and the finite mean is found.
    """
    if not tails:
        return weighted_sum(weights, values) / total
    weights = np.concatenate([np.asarray(weights, dtype=np.float64), [t[0] for t in tails]])
    values = np.concatenate([values, [t[1] for t in tails]])
    with np.errstate(over="ignore", invalid="ignore"):  # an overflow is redone below
        mean = weighted_sum(weights, values) / total
    if math.isfinite(mean) or not np.isfinite(values).all():
        return mean
    scale = _power_of_two_near(float(np.max(np.abs(values))))
    return weighted_sum(weights, values / scale) / total * scale


def _std_with_tails(
    values: np.ndarray,
    weights: np.ndarray,
    tails: list[tuple[float, float, float]],
    total: float,
    mean: float,
) -> float:
    """Return the population standard deviation of the bins (and tails) around ``mean`` (#33).

    Each tail adds its own spread, ``count * std**2``, and its mean's distance from ``mean``
    (centred moments: no cancellation). The squares are first summed as they are, which
    keeps every finite result bit for bit. When that overflows although the mean and every
    value are finite (``[-1e200, 1e200]`` has standard deviation ``1e200`` but squares of
    ``1e400``), the sum is redone with every value divided by a power of two near the
    largest, which is exact, so the finite spread is found.
    """
    try:
        with np.errstate(over="ignore", invalid="ignore"):  # an overflow is redone below
            std = _std_scaled(values, weights, tails, total, mean, 1.0)
    except OverflowError:  # float ** 2 raises where NumPy gives inf
        std = math.inf
    if math.isfinite(std) or not (math.isfinite(mean) and np.isfinite(values).all()):
        return std
    largest = max(float(np.max(np.abs(values), initial=0.0)), abs(mean))
    largest = max([largest] + [max(abs(m), s) for _, m, s in tails])
    return _std_scaled(values, weights, tails, total, mean, _power_of_two_near(largest))


def _power_of_two_near(largest: float) -> float:
    """Return the power of two at or just below ``largest``; values divided by it are below 2."""
    return math.ldexp(1.0, math.frexp(largest)[1] - 1)


def _std_scaled(
    values: np.ndarray,
    weights: np.ndarray,
    tails: list[tuple[float, float, float]],
    total: float,
    mean: float,
    scale: float,
) -> float:
    """Return the standard deviation computed in units of ``scale``, a power of two (1: as is)."""
    mean = mean / scale
    deviations = values / scale - mean
    squares = weighted_sum(weights, deviations * deviations)
    for count, tail_mean, tail_std in tails:
        spread, offset = tail_std / scale, tail_mean / scale - mean
        squares += count * (spread * spread) + count * (offset * offset)
    return float(np.sqrt(squares / total)) * scale


def _channel_histogram(record: PlaneRecord, channel: str | None) -> PlaneHistogram:
    if channel is None:
        if record.channel_count != 1:
            raise ValueError(
                f"Plane {record.plane_id} has {record.channel_count} channels "
                f"({', '.join(record.channels)}); give the ID of the channel to pool."
            )
        return record.histograms[0]
    return record.channel_histogram(channel)


def _inverse_cdf_index(cumulative: np.ndarray, total: int, q: float) -> int:
    rank = max(1, int(np.ceil(q / 100 * total)))
    return int(np.searchsorted(cumulative, rank, side="left"))


def _lerp(below: float, above: float, gamma: float) -> float:
    """NumPy's ``_lerp``: start from the nearer endpoint; finite endpoints never overflow."""
    with np.errstate(over="ignore"):  # an overflowing gap is handled just below
        difference = above - below
    if not math.isfinite(difference) and math.isfinite(below) and math.isfinite(above):
        # The weighted mean of two finite values is representable; form it from the
        # halved endpoints so the gap cannot overflow.
        half = above * 0.5 - below * 0.5
        if gamma >= 0.5:
            return float(above - half * (2 * (1 - gamma)))
        return float(below + half * (2 * gamma))
    if gamma >= 0.5:
        return float(above - difference * (1 - gamma))
    return float(below + difference * gamma)


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
    return _lerp(below, above, gamma)


def _weighted_index(cumulative: np.ndarray, total: float, q: float) -> int:
    # With whole-number weights, "cumulative >= q/100 * total" picks the same
    # entry as the integer rank of _inverse_cdf_index, so unit weights give
    # PooledHistogram's numbers exactly.
    index = int(np.searchsorted(cumulative, q / 100 * total, side="left"))
    return min(index, len(cumulative) - 1)
