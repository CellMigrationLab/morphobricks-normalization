"""Fit normalization parameters for one group and store them as columns (contract §2, §3, §8).

:func:`fit` is the one entry point for every method and scope. It takes either
the ``plane_histograms.csv`` rows of one group (from
:func:`~morphobricks_normalization.scopes.scope_groups`) or the pixels of one
image or patch, and returns :class:`Parameters`: an offset and a scale per
channel. Pixels are turned into histograms with
:func:`~morphobricks_normalization.histogram.compute_plane_histogram`, so both
sources go through the same pooling and the same method code, and give the same
numbers whenever the histograms are exact.

The fluorescence methods are those of
:mod:`~morphobricks_normalization.normalization`, applied to each channel
separately. The RGB method is StarDist's percentile normalization of
:mod:`~morphobricks_normalization.rgb`: the three channels are pooled together
and share one offset and scale.

Channels are addressed by canonical channel ID, never by their index in the image
(ADR 0003 HV2): rows of one group may have different channels, and a channel is
pooled over the rows that have it. Parameters are stored as one
``Normalization_Offset_<ID>`` and ``Normalization_Scale_<ID>`` pair per channel
(contract §8.1, N41, N42 as amended); for RGB the shared value is written for R, G
and B.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .channels import check_channel_ids
from .histogram import PlaneHistogram, compute_plane_histogram
from .issues import (
    APPROXIMATE_PARAMETERS,
    CHANNEL_PARTIALLY_PRESENT,
    PLANES_WITHOUT_FINITE_PIXELS,
    WARNING,
    Issue,
)
from .normalization import (
    NormalizationParameters,
    min_max,
    percentile_range,
    robust_z_score,
    z_score,
)
from .parameters import COLUMNS, Parameters, channel_axis_index, offset_column, scale_column
from .pooling import NoValidPixelsError, PooledHistogram, WeightedHistogram
from .records import PlaneRecord, duplicate_planes
from .rgb import (
    RGB_CHANNELS,
    check_rgb_channels,
    check_rgb_pixels,
    check_rgb_records,
    check_rgb_selection,
    rgb_percentile,
)
from .settings import NormalizationSettings, check_settings

# Re-exported: the parameter columns are part of this module's interface (contract §1.3).
__all__ = ["COLUMNS", "Parameters", "fit", "offset_column", "scale_column"]
_MAX_LISTED = 5


def fit(
    statistics: Iterable[PlaneRecord] | np.ndarray,
    settings: NormalizationSettings,
    *,
    group: str | None = None,
    channel_axis: int | None = None,
    channels: Iterable[str] | None = None,
) -> Parameters:
    """Fit the settings' method to one group, channel by channel ID.

    Parameters
    ----------
    statistics
        The ``plane_histograms.csv`` rows of one group (each row an XYC plane), or a
        pixel array of one image or patch (image and patch scopes only). Rows may
        have different channels (ADR 0003): a channel is pooled over the rows that
        have it, by its canonical channel ID, never by its position in the image;
        for RGB the R, G and B values of every row are pooled together.
    settings
        The run's settings.
    group
        The ``normalization_group`` text written with the parameters (contract
        §4.1); also names the group in messages.
    channel_axis
        For a pixel array, the axis that holds the channels; None for one
        channel.
    channels
        For rows: the channel IDs to fit; default every channel some row of the
        group has. For a pixel array (required): the channel ID of each entry along
        ``channel_axis``, in order, or the one ID of an array without a channel
        axis.

    Returns
    -------
    parameters : Parameters
        The group's parameters, channels in sorted ID order, with ``issues`` listing
        ``approximate_parameters``, ``planes_without_finite_pixels`` and
        ``channel_partially_present`` warnings.

    Raises
    ------
    MethodNotAllowedError
        As :func:`~morphobricks_normalization.settings.check_settings`; also
        for the biological-repeat scope with pixels instead of rows.
    NoValidPixelsError
        If a channel of the group has no finite pixel.
    ValueError
        If a fluorescence channel's scale is zero or not finite, the rows repeat a
        ``plane_id``, a requested channel is in no row of the group, the channel IDs of
        a pixel array are missing or do not match it, or RGB images do not have the
        channels R, G and B of an integer type (contract §5).
    """
    from_pixels = isinstance(statistics, np.ndarray)
    check_settings(settings, statistics_file=not from_pixels)
    rgb = settings.image_type == "rgb"
    if from_pixels:
        items, wanted, issues = _pixel_items(statistics, channel_axis, channels, rgb)
    else:
        if channel_axis is not None:
            raise ValueError("channel_axis applies to pixel arrays only, not to histogram rows.")
        items, wanted, issues = _record_items(statistics, channels, group, rgb)
    # Contract §3.3: R, G and B pooled together give one shared offset and scale.
    channel_sets = [RGB_CHANNELS] if rgb else [(channel,) for channel in wanted]
    offset, scale, exact = [], [], True
    for pooled_ids in channel_sets:
        where = _where(group, pooled_ids, rgb)
        try:
            pooled = _pool(items, settings, pooled_ids)
        except NoValidPixelsError as exc:
            raise NoValidPixelsError(f"{where}: {exc}") from exc
        if rgb:
            shared = rgb_percentile(pooled, *settings.percentile)
            offset += [shared[0]] * len(wanted)
            scale += [shared[1]] * len(wanted)
        else:
            try:
                parameters = _channel_parameters(pooled, settings)
            except ValueError as exc:
                raise ValueError(f"{where}: {exc}") from exc
            offset.append(parameters.offset)
            scale.append(parameters.scale)
        exact = exact and pooled.max_representative_error == 0
    return Parameters(
        method=settings.method_name,
        scope=settings.scope,
        group=group,
        exact=exact,
        offset=tuple(offset),
        scale=tuple(scale),
        channels=wanted,
        issues=(*issues[0], *_fit_issues(group, exact, issues[1])),
    )


def _pixel_items(
    pixels: np.ndarray, channel_axis: int | None, channels: Iterable[str] | None, rgb: bool
) -> tuple[dict[str, PlaneHistogram], tuple[str, ...], tuple[list[Issue], list[str]]]:
    """Return a pixel array's histograms by channel ID, the IDs to fit and no issues."""
    if rgb:  # refuse before any histogram is built (#52)
        check_rgb_pixels(pixels, channel_axis)
        # An RGB unit's samples 0, 1, 2 are R, G and B by the description (§4.4b).
        channels = RGB_CHANNELS if channels is None else channels
    ids = _pixel_channel_ids(pixels, channel_axis, channels)
    if rgb:
        check_rgb_channels(ids, "The pixel array")
    # One histogram per channel of the array, pooled like the rows' channels.
    by_id = dict(zip(ids, _pixel_histograms(pixels, channel_axis), strict=True))
    return by_id, tuple(sorted(by_id)), ([], [])


def _record_items(
    statistics: Iterable[PlaneRecord], channels: Iterable[str] | None, group: str | None, rgb: bool
) -> tuple[list[PlaneRecord], tuple[str, ...], tuple[list[Issue], list[str]]]:
    """Return the rows, the IDs to fit, partial-presence warnings and planes left out."""
    items = _records(statistics, group)
    if rgb:  # refuse before the rows are pooled
        check_rgb_records(items, group)
    wanted = _wanted_channels(items, channels, group)
    if rgb:
        check_rgb_selection(list(wanted), tuple(sorted(RGB_CHANNELS)))
    excluded = [
        f"{record.plane_id} (channel {channel})"
        for record in items
        for channel, histogram in zip(record.channels, record.histograms, strict=True)
        if channel in wanted and histogram.is_empty
    ]
    partial = [] if rgb else _partial_presence(items, wanted, group)
    return items, wanted, (partial, excluded)


def _where(group: str | None, channels: tuple[str, ...], rgb: bool) -> str:
    what = "channels R, G and B" if rgb else f"channel {channels[0]}"
    return f"Group '{group}', {what}" if group else what[0].upper() + what[1:]


def _records(statistics: Iterable[PlaneRecord], group: str | None) -> list[PlaneRecord]:
    records = list(statistics)
    if not records:
        raise ValueError("No plane_histograms.csv rows were given for this group.")
    if not all(isinstance(record, PlaneRecord) for record in records):
        raise TypeError(
            "fit expects the plane_histograms.csv rows of one group (PlaneRecord objects) "
            "or a pixel array."
        )
    duplicate_planes(records, f"The rows of group '{group}'" if group else "The rows of this group")
    return records


def _wanted_channels(
    records: list[PlaneRecord], channels: Iterable[str] | None, group: str | None
) -> tuple[str, ...]:
    """Return the channel IDs to fit, sorted: the requested ones, or every ID of the rows."""
    present = {channel for record in records for channel in record.channels}
    if channels is None:
        return tuple(sorted(present))
    wanted = check_channel_ids(channels, "fit")
    if not wanted:
        raise ValueError("Give at least one channel ID to fit, or leave channels out.")
    absent = [channel for channel in wanted if channel not in present]
    if absent:
        name = f"group '{group}'" if group else "this group"
        raise ValueError(
            f"No plane of {name} has channel(s) {', '.join(absent)} (its planes have "
            f"{', '.join(sorted(present))}), so there is nothing to fit for them. An absent "
            "channel is never fitted from zeros; fit it in a group whose images have it."
        )
    return tuple(sorted(wanted))


def _partial_presence(
    records: list[PlaneRecord], wanted: tuple[str, ...], group: str | None
) -> list[Issue]:
    """One warning per channel that only some rows of the group have (never silent)."""
    issues = []
    name = f"group '{group}'" if group else "this group"
    for channel in wanted:
        lacking = [record.plane_id for record in records if not record.has_channel(channel)]
        if not lacking:
            continue
        shown = ", ".join(lacking[:_MAX_LISTED])
        more = f" and {len(lacking) - _MAX_LISTED} more" if len(lacking) > _MAX_LISTED else ""
        issues.append(
            Issue(
                WARNING,
                CHANNEL_PARTIALLY_PRESENT,
                f"Channel {channel} is absent from {len(lacking)} of the {len(records)} planes "
                f"of {name} ({shown}{more}); its parameters come from the "
                f"{len(records) - len(lacking)} planes that have it. Nothing is filled in for "
                "the others.",
                items=tuple(lacking),
                action="Normally no action is needed when these images were acquired without "
                "this channel. Their channel has no parameters; normalize it only where it was "
                "acquired.",
                channel=channel,
            )
        )
    return issues


def _pixel_channel_ids(
    pixels: np.ndarray, channel_axis: int | None, channels: Iterable[str] | None
) -> tuple[str, ...]:
    """Return the channel IDs of a pixel array, one per entry along the channel axis."""
    if channels is None:
        raise ValueError(
            "Give channels, the canonical channel ID of each entry along channel_axis (or the "
            "one ID of an array without a channel axis): parameters are named by channel ID, "
            "never by position."
        )
    ids = check_channel_ids(channels, "fit")
    count = (
        1 if channel_axis is None else pixels.shape[channel_axis_index(channel_axis, pixels.ndim)]
    )
    if len(ids) != count:
        where = "without a channel axis" if channel_axis is None else f"along axis {channel_axis}"
        raise ValueError(
            f"The pixel array has {count} channel(s) {where}, but {len(ids)} channel ID(s) were "
            f"given ({', '.join(ids)})."
        )
    return ids


def _pixel_histograms(pixels: np.ndarray, channel_axis: int | None) -> list[PlaneHistogram]:
    if channel_axis is None:
        return [compute_plane_histogram(pixels)]
    axis = channel_axis_index(channel_axis, pixels.ndim)
    return [
        compute_plane_histogram(np.take(pixels, c, axis=axis)) for c in range(pixels.shape[axis])
    ]


def _channel_histograms(
    items: list[PlaneRecord] | dict[str, PlaneHistogram], channels: tuple[str, ...]
) -> list[PlaneHistogram]:
    # The histograms of ``channels`` in every row that has them (only the same channel ID is
    # pooled; the rows without it are reported by _partial_presence), or of a pixel array.
    if isinstance(items, dict):
        return [items[c] for c in channels]
    return [
        record.channel_histogram(c) for record in items for c in channels if record.has_channel(c)
    ]


def _pool(
    items: list[PlaneRecord] | dict[str, PlaneHistogram],
    settings: NormalizationSettings,
    channels: tuple[str, ...],
) -> PooledHistogram | WeightedHistogram:
    """Pool ``channels`` of the rows that have them, weighted per condition if set (§4.3)."""
    if settings.pooling != "conditions":
        return PooledHistogram(_channel_histograms(items, channels))
    # Contract §4.3: each condition's counts are scaled to the same total. The
    # largest condition is the reference, so conditions with equal pixel
    # counts get weight exactly 1 and the numbers equal those of "pixels".
    by_condition: dict[str, list[PlaneRecord]] = {}
    for record in items:
        if all(record.has_channel(c) for c in channels):
            by_condition.setdefault(record.condition, []).append(record)
    parts = []
    for _, members in sorted(by_condition.items()):
        histograms = _channel_histograms(members, channels)
        if any(not histogram.is_empty for histogram in histograms):
            parts.append(PooledHistogram(histograms))
    if not parts:
        return PooledHistogram(_channel_histograms(items, channels))  # raises NoValidPixelsError
    reference = max(part.pixel_count for part in parts)
    return WeightedHistogram(parts, [reference / part.pixel_count for part in parts])


def _channel_parameters(
    pooled: PooledHistogram | WeightedHistogram, settings: NormalizationSettings
) -> NormalizationParameters:
    if settings.method == "minmax":
        return min_max(pooled)
    if settings.method == "percentile":
        low, high = settings.percentile
        return percentile_range(pooled, low, high)
    if settings.method == "zscore":
        return z_score(pooled)
    return robust_z_score(pooled, settings.consistency)


def _fit_issues(group: str | None, exact: bool, excluded: list[str]) -> tuple[Issue, ...]:
    name = f"group '{group}'" if group else "this group"
    issues = []
    if excluded:
        issues.append(
            Issue(
                WARNING,
                PLANES_WITHOUT_FINITE_PIXELS,
                f"{len(excluded)} channel plane(s) without finite pixels were left out of "
                f"{name}; its parameters come from its other planes.",
                items=tuple(excluded),
                action="Check these planes in the source images: they are empty or hold only "
                "NaN or infinite values.",
            )
        )
    if not exact:
        issues.append(
            Issue(
                WARNING,
                APPROXIMATE_PARAMETERS,
                f"The parameters of {name} come from histograms with more than one value per "
                "bin, so they are approximate (normalization_exact = false): every percentile "
                "is within half the widest bin of the exact value.",
                items=(group,) if group else (),
                action="Normally no action is needed. Integer images with at most 65,536 "
                "distinct values per plane give exact parameters.",
            )
        )
    return tuple(issues)
