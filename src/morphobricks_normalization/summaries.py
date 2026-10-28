"""Per-image intensity summaries from ``plane_histograms.csv`` rows (contract §7.1, N38).

One :class:`ImageSummary` per image pools all its Z and T planes, channel by
channel, with the group statistics of contract §1.2.3: the median and the low
and high percentiles are inverse-CDF percentiles of the pooled histogram
(:class:`~morphobricks_normalization.pooling.PooledHistogram`), exact for 8- and
16-bit histograms. Images are never read.

The saturated fraction counts the pixels equal to the pixel type's maximum. It
is known exactly only for integer planes, and only when the maximum value has a
bin of its own (not the overflow of a robust range, contract §1.4); a plane
whose maximum stays below the type's maximum has no saturated pixel whatever its
binning. Otherwise the fraction is blank (None).

:meth:`ImageSummary.to_row` gives the columns of contract §7.1, channels in
CellProfiler-style columns (N41, N42), for the calling brick to show or write;
the library writes no file (N50).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .csv_io import channel_column
from .dtypes import is_integer
from .pooling import NoValidPixelsError, PooledHistogram
from .records import PlaneRecord, different_channel_counts
from .settings import DEFAULT_PERCENTILE

SHARED_COLUMNS = (
    "source_relative_path",
    "condition",
    "biological_repeat",
    "technical_repeat",
    "image_file",
)
# Per-channel columns as <Category>_<Feature>; the channel token is appended by
# channel_column.
SUMMARY_FEATURES = (
    "Intensity_MedianIntensity",
    "Intensity_LowPercentile",
    "Intensity_HighPercentile",
    "Intensity_SaturatedFraction",
    "Histogram_Exact",
)


@dataclass(frozen=True)
class ImageSummary:
    """Intensity summary of one image, all Z and T planes pooled per channel.

    Attributes
    ----------
    source_relative_path, condition, biological_repeat, technical_repeat
        The image's shared columns, as in ``plane_histograms.csv``.
    dtype
        Pixel type of the image.
    percentile
        ``(low, high)`` in percent: the percentiles held by ``p_low`` and ``p_high``.
    median, p_low, p_high
        One value per channel, channel 0 first; None for a channel without
        finite pixels.
    saturated_fraction
        Per channel, the pixels equal to the pixel type's maximum over the valid
        pixels; None for float images, for a maximum that shares its bin with
        other values, and for a channel without finite pixels.
    exact
        Per channel, whether every pooled histogram was exact
        (``max_representative_error == 0``).
    """

    source_relative_path: str
    condition: str
    biological_repeat: str
    technical_repeat: str
    dtype: np.dtype
    percentile: tuple[float, float]
    median: tuple[float | None, ...]
    p_low: tuple[float | None, ...]
    p_high: tuple[float | None, ...]
    saturated_fraction: tuple[float | None, ...]
    exact: tuple[bool, ...]

    @property
    def image_file(self) -> str:
        """File name of the image."""
        return self.source_relative_path.rsplit("/", 1)[-1]

    @property
    def channel_count(self) -> int:
        """Number of channels of the image."""
        return len(self.median)

    def to_row(self) -> dict[str, Any]:
        """Return the columns of contract §7.1, in order.

        The shared columns as text, then for each channel the median, low and
        high percentiles and saturated fraction as floats (None for a blank) and
        ``Histogram_Exact_C<i>`` as a bool. A list of rows can be passed to
        ``pandas.DataFrame`` or ``csv.DictWriter`` (with :func:`summary_columns`).
        """
        row: dict[str, Any] = {
            "source_relative_path": self.source_relative_path,
            "condition": self.condition,
            "biological_repeat": self.biological_repeat,
            "technical_repeat": self.technical_repeat,
            "image_file": self.image_file,
        }
        for channel in range(self.channel_count):
            values = (
                self.median[channel],
                self.p_low[channel],
                self.p_high[channel],
                self.saturated_fraction[channel],
                self.exact[channel],
            )
            for feature, value in zip(SUMMARY_FEATURES, values, strict=True):
                row[channel_column(feature, channel)] = value
        return row


def summary_columns(channel_count: int) -> list[str]:
    """Column names of :meth:`ImageSummary.to_row` for images with ``channel_count`` channels."""
    return [
        *SHARED_COLUMNS,
        *(
            channel_column(feature, channel)
            for channel in range(channel_count)
            for feature in SUMMARY_FEATURES
        ),
    ]


def image_summaries(
    records: Iterable[PlaneRecord], percentile: tuple[float, float] = DEFAULT_PERCENTILE
) -> list[ImageSummary]:
    """Summarize every image of ``plane_histograms.csv``, all Z and T planes pooled.

    Parameters
    ----------
    records
        Rows of the statistics file (one XYC plane each).
    percentile
        ``(low, high)`` in percent for ``p_low`` and ``p_high``,
        ``0 <= low < high <= 100``; default ``(1.0, 99.8)``.

    Returns
    -------
    list of ImageSummary
        One per image, sorted by condition, biological repeat, technical repeat
        and path.

    Raises
    ------
    ValueError
        For invalid percentiles, or images with different numbers of channels (N43).
    """
    low, high = (float(q) for q in percentile)
    if not 0 <= low < high <= 100:
        raise ValueError(f"Expected 0 <= low < high <= 100; got low={low}, high={high}.")
    records = list(records)
    if len({r.channel_count for r in records}) > 1:
        raise ValueError(
            different_channel_counts(
                [(r.source_relative_path, r.channel_count) for r in records], "These images"
            )
        )
    images: dict[tuple[str, ...], list[PlaneRecord]] = {}
    for record in records:
        images.setdefault(record.group_key("image"), []).append(record)
    return [_summary(planes, (low, high)) for _, planes in sorted(images.items())]


def _summary(planes: list[PlaneRecord], percentile: tuple[float, float]) -> ImageSummary:
    first = planes[0]
    median, p_low, p_high, saturated, exact = [], [], [], [], []
    for channel in range(first.channel_count):
        histograms = [plane.channel_histogram(channel) for plane in planes]
        try:
            pooled = PooledHistogram(histograms)
        except NoValidPixelsError:
            # A channel without finite pixels has no statistics; its values stay blank.
            median.append(None)
            p_low.append(None)
            p_high.append(None)
            saturated.append(None)
            exact.append(True)
            continue
        median.append(pooled.median)
        p_low.append(pooled.percentile(percentile[0]))
        p_high.append(pooled.percentile(percentile[1]))
        saturated.append(_saturated_fraction(histograms, pooled.pixel_count))
        exact.append(pooled.max_representative_error == 0)
    return ImageSummary(
        source_relative_path=first.source_relative_path,
        condition=first.condition,
        biological_repeat=first.biological_repeat,
        technical_repeat=first.technical_repeat,
        dtype=first.histograms[0].dtype,
        percentile=percentile,
        median=tuple(median),
        p_low=tuple(p_low),
        p_high=tuple(p_high),
        saturated_fraction=tuple(saturated),
        exact=tuple(exact),
    )


def _saturated_fraction(histograms, valid_pixels: int) -> float | None:
    dtype = histograms[0].dtype
    if not is_integer(dtype):
        return None
    top = int(np.iinfo(dtype).max)
    saturated = 0
    for hist in histograms:
        if hist.is_empty or hist.pixel_max < top:
            continue
        if not hist.overflow.is_empty:
            # The maximum lies above a robust range, among other overflow values.
            return None
        # The maximum lies in the last occupied bin; its count is the number of
        # saturated pixels only if the bin holds no other value.
        index = int(np.flatnonzero(hist.counts)[-1])
        first, last = hist.integer_bin_limits()[index]
        if first != last:
            return None
        saturated += int(hist.counts[index])
    return saturated / valid_pixels
