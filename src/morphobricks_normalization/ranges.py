"""Histogram range settings and the narrow-range check (contract §1.2.2, §1.4, N54–N60).

Histograms that are not exact (float planes, and integer planes with more
values than bins) are binned in equal parts of a range. By default the range is
the plane's exact minimum to maximum, so one extreme pixel, for example a hot
pixel, can squeeze every other pixel into a few bins (issue #12). Two tools
address this:

- **Robust range** (``histogram_range="robust"``): each channel's bins span
  the plane's quantiles at ``robust_quantiles`` (default 0.01 % and 99.99 %),
  and the pixels outside are kept as underflow and overflow counts and sums
  (:class:`~morphobricks_normalization.histogram.HistogramTail`).
- **Range check** (:class:`RangeCheck`): for every plane and channel whose
  histogram is not exact, how many bins the central range (1st to 99.8th
  percentile) covers, how many pixels lie outside the robust limits and how
  many bins the central range would cover with the robust range. A central
  range of fewer than ``min_central_bins`` bins is reported by
  :func:`range_check_issues`, one warning per channel, for the preprocessing
  brick to show (N60).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._version import __version__
from .issues import NARROW_HISTOGRAM_RANGE, WARNING, Issue

HISTOGRAM_RANGES = ("minmax", "robust")
DEFAULT_ROBUST_QUANTILES = (0.01, 99.99)
DEFAULT_MIN_CENTRAL_BINS = 100
# The central range of the check: the default percentiles of the percentile method
# (settings.DEFAULT_PERCENTILE, not imported: settings imports this module's users).
CENTRAL_PERCENTILES = (1.0, 99.8)

_KEYS = ("library_version", "histogram_range", "robust_quantiles", "min_central_bins")
# Planes named in one warning before the rest are counted.
_MAX_LISTED = 5


@dataclass(frozen=True)
class HistogramSettings:
    """How histograms that are not exact are binned, and when their range is too narrow.

    Stored once per run by the brick that computes the statistics, with
    :meth:`to_dict`, and read back with :meth:`from_dict`.

    Attributes
    ----------
    histogram_range
        ``"minmax"`` (default): bins span the plane's exact minimum to maximum.
        ``"robust"``: bins span the plane's quantiles at ``robust_quantiles``, and
        the pixels outside are counted as underflow and overflow. Exact
        histograms (8- and 16-bit planes with one bin per value) never change.
    robust_quantiles
        ``(low, high)`` in percent, ``0 <= low < high <= 100``; default
        ``(0.01, 99.99)``. Used for the robust range and by the range check.
    min_central_bins
        The range check reports a channel whose central range (1st to 99.8th
        percentile) covers fewer bins than this; default 100.

    Raises
    ------
    ValueError
        For an unknown range, invalid quantiles or a bin limit below 1.
    """

    histogram_range: str = "minmax"
    robust_quantiles: tuple[float, float] = DEFAULT_ROBUST_QUANTILES
    min_central_bins: int = DEFAULT_MIN_CENTRAL_BINS

    def __post_init__(self) -> None:
        if self.histogram_range not in HISTOGRAM_RANGES:
            raise ValueError(
                f"Histogram range {self.histogram_range!r} does not exist. Choose 'minmax' "
                "(bins from each plane's minimum to its maximum) or 'robust' (bins between "
                "extreme quantiles, the pixels outside counted separately)."
            )
        try:
            if isinstance(self.robust_quantiles, str):
                raise TypeError
            low, high = (float(q) for q in self.robust_quantiles)
        except (TypeError, ValueError):
            raise ValueError(
                f"robust_quantiles must be two numbers, [low, high]; got {self.robust_quantiles!r}."
            ) from None
        if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high <= 100):
            raise ValueError(
                f"robust_quantiles must be percentages with 0 <= low < high <= 100; got "
                f"low={low}, high={high}."
            )
        bins = self.min_central_bins
        if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 1:
            raise ValueError(
                f"min_central_bins must be a whole number of at least 1; got {bins!r}."
            )
        object.__setattr__(self, "robust_quantiles", (low, high))
        object.__setattr__(self, "min_central_bins", int(bins))

    @property
    def robust(self) -> bool:
        """Whether histograms that are not exact use the robust range."""
        return self.histogram_range == "robust"

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible settings, for the run configuration."""
        return {
            "library_version": __version__,
            "histogram_range": self.histogram_range,
            "robust_quantiles": list(self.robust_quantiles),
            "min_central_bins": self.min_central_bins,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HistogramSettings:
        """Read settings written by :meth:`to_dict`; missing keys take their defaults.

        ``library_version`` is the version that wrote the settings and is not checked.

        Raises
        ------
        ValueError
            For an unknown key, or a value the settings do not accept.
        """
        unknown = sorted(set(data) - set(_KEYS))
        if unknown:
            raise ValueError(
                f"Unknown histogram setting(s): {', '.join(unknown)}. Known settings: "
                f"{', '.join(_KEYS)}. Check the spelling in the run configuration."
            )
        values = {key: data[key] for key in _KEYS[1:] if key in data}
        if "robust_quantiles" in values and not isinstance(values["robust_quantiles"], str):
            values["robust_quantiles"] = tuple(values["robust_quantiles"])
        return cls(**values)


@dataclass(frozen=True)
class RangeCheck:
    """How well one channel plane's histogram resolves its central range.

    Computed with the histogram by
    :func:`~morphobricks_normalization.histogram.histogram_with_range_check`,
    for histograms that are not exact. Percentiles and quantiles are exact
    inverse-CDF order statistics of the plane's finite pixels.

    Attributes
    ----------
    histogram_range
        The range the histogram was binned with, ``"minmax"`` or ``"robust"``.
    central_percentiles
        ``(low, high)`` in percent: the central range, default ``(1.0, 99.8)``.
    central_bins
        Bins of the histogram that the central range covers: the bin of the
        high percentile minus the bin of the low percentile, plus one.
    robust_quantiles
        ``(low, high)`` in percent of the robust limits.
    robust_limits
        The plane's values at ``robust_quantiles``.
    outside_robust_count
        Pixels below or above the robust limits.
    robust_central_bins
        Bins the central range would cover with the robust range (equal to
        ``central_bins`` when the histogram used it).
    min_central_bins
        The limit below which the range is reported.
    """

    histogram_range: str
    central_percentiles: tuple[float, float]
    central_bins: int
    robust_quantiles: tuple[float, float]
    robust_limits: tuple[int | float, int | float]
    outside_robust_count: int
    robust_central_bins: int
    min_central_bins: int

    @property
    def narrow(self) -> bool:
        """True when the central range covers fewer than ``min_central_bins`` bins."""
        return self.central_bins < self.min_central_bins


def range_check_issues(checks: Iterable[tuple[str, int, RangeCheck | None]]) -> list[Issue]:
    """Warnings for the narrow central ranges among ``checks``, one per channel (N60).

    Parameters
    ----------
    checks
        ``(plane_id, channel, check)`` for every channel plane; a None check
        (an exact histogram) is skipped.

    Returns
    -------
    list of Issue
        One ``narrow_histogram_range`` warning per channel with at least one
        narrow plane, in channel order; ``items`` lists the planes and
        ``channel`` holds the channel. Empty when no range is narrow.
    """
    by_channel: dict[int, list[tuple[str, RangeCheck]]] = {}
    for plane_id, channel, check in checks:
        if check is not None and check.narrow:
            by_channel.setdefault(int(channel), []).append((plane_id, check))
    issues = []
    for channel, found in sorted(by_channel.items()):
        planes = list(dict.fromkeys(plane_id for plane_id, _ in found))
        first = found[0][1]
        low, high = (_percent(q) for q in first.central_percentiles)
        q_low, q_high = (_percent(q) for q in first.robust_quantiles)
        fewest = min(check.central_bins for _, check in found)
        outside = sum(check.outside_robust_count for _, check in found)
        robust_bins = min(check.robust_central_bins for _, check in found)
        shown = ", ".join(planes[:_MAX_LISTED])
        more = f" and {len(planes) - _MAX_LISTED} more" if len(planes) > _MAX_LISTED else ""
        if first.histogram_range == "robust":
            cause = (
                "Even between the robust limits, most pixels lie in a few bins: the channel may "
                "be nearly constant, or hold a few very different intensity levels."
            )
            action = (
                "Check these planes in the source images. Percentile parameters from them are "
                "coarse; a larger normalization group or another method may suit them better."
            )
        else:
            cause = (
                f"Usually a few extreme pixels (for example hot or saturated pixels) stretch the "
                f"histogram's range: {outside:,} pixel(s) of these planes lie outside their "
                f"{q_low}-{q_high} % quantiles, and binned between those quantiles the central "
                f"range would cover at least {robust_bins:,} bins."
            )
            action = (
                "Compute the statistics again with the robust histogram range, which bins each "
                f"plane between its {q_low} % and {q_high} % quantiles and counts the pixels "
                "outside separately; or check the images for hot or saturated pixels."
            )
        issues.append(
            Issue(
                WARNING,
                NARROW_HISTOGRAM_RANGE,
                f"In {len(planes)} plane(s), the central intensity range of channel C{channel} "
                f"(its {low}-{high} percentile range) covers fewer than "
                f"{first.min_central_bins:,} of the histogram's bins (as few as {fewest:,}): "
                f"{shown}{more}. Percentile parameters from these planes are coarse, or cannot "
                f"be computed. {cause}",
                items=tuple(planes),
                action=action,
                channel=channel,
            )
        )
    return issues


def _percent(value: float) -> str:
    return f"{value:g}"
