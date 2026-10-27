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
and share one offset and scale. Parameters are stored as one
``Normalization_Offset_C<i>`` and ``Normalization_Scale_C<i>`` pair per channel
(contract §8.1, N41, N42); for RGB the shared value is written for C0, C1 and C2.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .histogram import PlaneHistogram, compute_plane_histogram
from .issues import APPROXIMATE_PARAMETERS, PLANES_WITHOUT_FINITE_PIXELS, WARNING, Issue
from .normalization import (
    NormalizationParameters,
    min_max,
    percentile_range,
    robust_z_score,
    z_score,
)
from .pooling import NoValidPixelsError, PooledHistogram, WeightedHistogram
from .records import PlaneRecord, different_channel_counts
from .rgb import RGB_CHANNEL_COUNT, apply_rgb, check_rgb, rgb_percentile
from .settings import NormalizationSettings, check_settings

# Columns written once per row; the per-channel pairs follow (offset_column,
# scale_column).
COLUMNS = (
    "normalization_method",
    "normalization_scope",
    "normalization_group",
    "normalization_exact",
)
OFFSET_COLUMN_PREFIX = "Normalization_Offset"
SCALE_COLUMN_PREFIX = "Normalization_Scale"
_CHANNEL_COLUMN = re.compile(r"Normalization_(?:Offset|Scale)_C(0|[1-9][0-9]*)")


def offset_column(channel: int) -> str:
    """Column of channel ``channel``'s offset, e.g. ``Normalization_Offset_C0``."""
    return f"{OFFSET_COLUMN_PREFIX}_C{channel}"


def scale_column(channel: int) -> str:
    """Column of channel ``channel``'s scale, e.g. ``Normalization_Scale_C0``."""
    return f"{SCALE_COLUMN_PREFIX}_C{channel}"


@dataclass(frozen=True)
class Parameters:
    """Normalization of one group: ``(pixels - offset[k]) / scale[k]`` for its ``k``-th channel.

    Attributes
    ----------
    method
        Written method name, e.g. ``"percentile 1-99.8"`` (contract §3.1).
    scope
        ``"biological_repeat"``, ``"image"`` or ``"patch"``.
    group
        The ``normalization_group`` text (contract §4.1), or None when the
        calling brick has not named the group.
    exact
        Whether every pooled histogram was exact.
    offset, scale
        One value per channel, in the order of ``channels``.
    channels
        The image's channel indices the parameters belong to, in increasing
        order; by default ``0, 1, ...``. A single-channel patch of channel 2 of
        an image has ``channels == (2,)`` (contract §8.1).
    issues
        Warnings about these parameters (``approximate_parameters``,
        ``planes_without_finite_pixels``); not stored in the columns and not
        compared.
    """

    method: str
    scope: str
    group: str | None
    exact: bool
    offset: tuple[float, ...]
    scale: tuple[float, ...]
    channels: tuple[int, ...] | None = None
    issues: tuple[Issue, ...] = field(default=(), compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "offset", tuple(float(value) for value in self.offset))
        object.__setattr__(self, "scale", tuple(float(value) for value in self.scale))
        if not self.offset or len(self.offset) != len(self.scale):
            raise ValueError(
                f"Expected one offset and one scale per channel; got {len(self.offset)} "
                f"offset(s) and {len(self.scale)} scale(s)."
            )
        channels = range(len(self.offset)) if self.channels is None else self.channels
        channels = tuple(int(c) for c in channels)
        increasing = all(b > a for a, b in zip((-1, *channels), channels, strict=False))
        if len(channels) != len(self.offset) or not increasing:
            raise ValueError(
                "Expected one channel index per offset, from 0 upwards in increasing order; "
                f"got channels {channels} for {len(self.offset)} offset(s)."
            )
        object.__setattr__(self, "channels", channels)

    @property
    def channel_count(self) -> int:
        """Number of channels the parameters normalize."""
        return len(self.offset)

    @property
    def image_type(self) -> str:
        """``"rgb"`` for StarDist's RGB method (``rgb percentile ...``), else ``"fluorescence"``."""
        return "rgb" if self.method.startswith("rgb ") else "fluorescence"

    def select(self, channels: Iterable[int]) -> Parameters:
        """Return the parameters of some of the channels only.

        For example ``parameters.select([2])`` gives the parameters of a
        single-channel patch of channel 2, whose columns fill only
        ``Normalization_Offset_C2`` and ``Normalization_Scale_C2`` (contract §8.1).

        Raises
        ------
        ValueError
            If a channel is not among :attr:`channels`.
        """
        wanted = sorted({int(c) for c in channels})
        missing = [c for c in wanted if c not in self.channels]
        if not wanted:
            raise ValueError("Select at least one channel of these parameters.")
        if missing:
            raise ValueError(
                f"These parameters have channel(s) {list(self.channels)}; channel(s) "
                f"{missing} cannot be selected."
            )
        index = [self.channels.index(c) for c in wanted]
        return replace(
            self,
            offset=tuple(self.offset[i] for i in index),
            scale=tuple(self.scale[i] for i in index),
            channels=tuple(wanted),
        )

    def apply(self, pixels: np.ndarray, *, channel_axis: int | None = None) -> np.ndarray:
        """Return the normalized pixels as a new float32 array.

        Computed in float64 as ``(pixels - offset) / scale`` and returned as
        float32, without clipping; non-finite pixels stay non-finite and the
        input is never modified. RGB parameters repeat CSBDeep's float32
        arithmetic instead, so the result is bit-identical to StarDist's
        normalization (:func:`~morphobricks_normalization.rgb.apply_rgb`, N48).

        Parameters
        ----------
        pixels
            Pixel values of the group.
        channel_axis
            Axis of ``pixels`` that holds the channels, one entry per channel of
            :attr:`channels` in that order; may be left out when the parameters
            have one channel.

        Raises
        ------
        ValueError
            If the channel axis is missing or does not hold one entry per channel.
        """
        values = np.asarray(pixels)
        if self.image_type == "fluorescence":
            values = values.astype(np.float64, copy=False)
        if channel_axis is None:
            if self.channel_count != 1:
                raise ValueError(
                    f"These parameters have {self.channel_count} channels; give channel_axis, "
                    "the axis of the channels in the pixel array."
                )
            offset, scale = self.offset[0], self.scale[0]
        else:
            axis = _axis_index(channel_axis, values.ndim)
            if values.shape[axis] != self.channel_count:
                raise ValueError(
                    f"The pixel array has {values.shape[axis]} channel(s) along axis "
                    f"{channel_axis}, but these parameters have {self.channel_count}."
                )
            shape = [1] * values.ndim
            shape[axis] = self.channel_count
            offset = np.asarray(self.offset).reshape(shape)
            scale = np.asarray(self.scale).reshape(shape)
        if self.image_type == "rgb":
            return apply_rgb(values, offset, scale)
        return ((values - offset) / scale).astype(np.float32, copy=False)

    def to_columns(self) -> dict[str, str]:
        """Return the columns of contract §8.1 as text, numbers in full.

        The four ``normalization_*`` columns, then ``Normalization_Offset_C<i>``
        and ``Normalization_Scale_C<i>`` for each channel ``i`` of
        :attr:`channels`, grouped by channel. Columns of other channels are not
        returned; a table holding rows of other channels leaves them blank.
        """
        columns = {
            "normalization_method": self.method,
            "normalization_scope": self.scope,
            "normalization_group": self.group or "",
            "normalization_exact": "true" if self.exact else "false",
        }
        for channel, offset, scale in zip(self.channels, self.offset, self.scale, strict=True):
            columns[offset_column(channel)] = repr(offset)
            columns[scale_column(channel)] = repr(scale)
        return columns

    @classmethod
    def from_columns(
        cls, columns: Mapping[str, Any], settings: NormalizationSettings
    ) -> Parameters:
        """Read parameters back from their columns; ``from_columns(p.to_columns()) == p``.

        Parameters
        ----------
        columns
            A row with the four ``normalization_*`` columns and the
            ``Normalization_Offset_C<i>`` and ``Normalization_Scale_C<i>`` pair of
            at least one channel; other columns are ignored. Values may be text
            or numbers. A channel whose two columns are both blank or absent is
            not part of the row's parameters (a single-channel patch fills only
            its own channel).
        settings
            The settings of the run that wrote the row; its method must be the
            row's ``normalization_method``.

        Raises
        ------
        ValueError
            Naming the column that is missing or cannot be read, or whose
            method differs from the settings; also for a channel with only one
            of its two columns.
        """
        missing = [name for name in COLUMNS if name not in columns]
        if missing:
            raise ValueError(
                f"The normalization column(s) {', '.join(missing)} are missing. Read the "
                "parameters from a table written with normalization."
            )
        method = _text(columns["normalization_method"])
        if method != settings.method_name:
            raise ValueError(
                f"Column normalization_method is {method!r}, but the settings use "
                f"{settings.method_name!r}. Read the parameters with the settings of the run "
                "that wrote them."
            )
        indices = sorted(
            {int(match[1]) for name in columns if (match := _CHANNEL_COLUMN.fullmatch(str(name)))}
        )
        if not indices:
            raise ValueError(
                f"The columns {offset_column(0)} and {scale_column(0)} are missing: the row has "
                "no Normalization_Offset_C<i> and Normalization_Scale_C<i> columns. Read the "
                "parameters from a table written with normalization."
            )
        channels, offset, scale = [], [], []
        for channel in indices:
            names = (offset_column(channel), scale_column(channel))
            texts = [_text(columns.get(name)) for name in names]
            if texts == ["", ""]:
                continue  # a channel this row does not hold
            if "" in texts:
                blank, given = names if texts[0] == "" else names[::-1]
                raise ValueError(
                    f"Column {blank} is blank or missing, but {given} holds a value; every "
                    "channel of a row needs both its offset and its scale."
                )
            channels.append(channel)
            offset.append(_number(columns, names[0]))
            scale.append(_number(columns, names[1]))
            if scale[-1] <= 0:
                raise ValueError(
                    f"Column {names[1]} must hold a positive number; got {texts[1]!r}."
                )
        if not channels:
            raise ValueError(
                "Every Normalization_Offset_C<i> and Normalization_Scale_C<i> column of this row "
                "is blank, so it holds no parameters. Read the parameters from a row written "
                "with normalization."
            )
        if settings.image_type == "rgb":
            _check_rgb_columns(channels, offset, scale)
        return cls(
            method=method,
            scope=_text(columns["normalization_scope"]),
            group=_text(columns["normalization_group"]) or None,
            exact=_flag(columns, "normalization_exact"),
            offset=tuple(offset),
            scale=tuple(scale),
            channels=tuple(channels),
        )


def fit(
    statistics: Iterable[PlaneRecord] | np.ndarray,
    settings: NormalizationSettings,
    *,
    group: str | None = None,
    channel_axis: int | None = None,
) -> Parameters:
    """Fit the settings' method to one group.

    Parameters
    ----------
    statistics
        The ``plane_histograms.csv`` rows of one group (each row an XYC plane;
        each channel gets its own parameters, in channel order, or for RGB the
        R, G and B values of every row are pooled together), or a pixel array
        of one image or patch (image and patch scopes only).
    settings
        The run's settings.
    group
        The ``normalization_group`` text written with the parameters (contract
        §4.1); also names the group in messages.
    channel_axis
        For a pixel array, the axis that holds the channels; None for one
        channel.

    Returns
    -------
    parameters : Parameters
        The group's parameters, with ``issues`` listing ``approximate_parameters`` and
        ``planes_without_finite_pixels`` warnings.

    Raises
    ------
    MethodNotAllowedError
        As :func:`~morphobricks_normalization.settings.check_settings`; also
        for the biological-repeat scope with pixels instead of rows.
    NoValidPixelsError
        If a channel of the group has no finite pixel.
    ValueError
        If a fluorescence channel's scale is zero or not finite, the rows have
        different numbers of channels (N43), or RGB images do not have exactly
        three channels of an integer type (contract §5).
    """
    from_pixels = isinstance(statistics, np.ndarray)
    check_settings(settings, statistics_file=not from_pixels)
    if from_pixels:
        # One histogram per channel of the array, pooled like the rows' channels.
        items: list[PlaneRecord] | list[PlaneHistogram] = _pixel_histograms(
            statistics, channel_axis
        )
        channel_count = len(items)
        excluded: list[str] = []
    else:
        if channel_axis is not None:
            raise ValueError("channel_axis applies to pixel arrays only, not to histogram rows.")
        items = _records(statistics, group)
        channel_count = items[0].channel_count
        excluded = [
            f"{record.plane_id} (channel {channel})"
            for record in items
            for channel, histogram in enumerate(record.histograms)
            if histogram.is_empty
        ]
    rgb = settings.image_type == "rgb"
    if rgb:
        check_rgb(channel_count, _dtypes(items))
        # Contract §3.3: R, G and B pooled together give one shared offset and scale.
        channel_sets = [tuple(range(channel_count))]
    else:
        channel_sets = [(channel,) for channel in range(channel_count)]
    offset, scale, exact = [], [], True
    for channels in channel_sets:
        if rgb:
            where = f"Group '{group}', channels R, G and B" if group else "Channels R, G and B"
        else:
            where = f"Group '{group}', channel {channels[0]}" if group else f"Channel {channels[0]}"
        try:
            pooled = _pool(items, settings, channels)
        except NoValidPixelsError as exc:
            raise NoValidPixelsError(f"{where}: {exc}") from exc
        if rgb:
            shared = rgb_percentile(pooled, *settings.percentile)
            offset += [shared[0]] * len(channels)
            scale += [shared[1]] * len(channels)
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
        issues=_fit_issues(group, exact, excluded),
    )


def _records(statistics: Iterable[PlaneRecord], group: str | None) -> list[PlaneRecord]:
    records = list(statistics)
    if not records:
        raise ValueError("No plane_histograms.csv rows were given for this group.")
    if not all(isinstance(record, PlaneRecord) for record in records):
        raise TypeError(
            "fit expects the plane_histograms.csv rows of one group (PlaneRecord objects) "
            "or a pixel array."
        )
    if len({record.channel_count for record in records}) > 1:
        subject = f"The images of group '{group}'" if group else "The images of this group"
        raise ValueError(
            different_channel_counts(
                [(r.source_relative_path, r.channel_count) for r in records], subject
            )
        )
    return records


def _pixel_histograms(pixels: np.ndarray, channel_axis: int | None) -> list[PlaneHistogram]:
    if channel_axis is None:
        return [compute_plane_histogram(pixels)]
    axis = _axis_index(channel_axis, pixels.ndim)
    return [
        compute_plane_histogram(np.take(pixels, c, axis=axis)) for c in range(pixels.shape[axis])
    ]


def _dtypes(items: list[PlaneRecord] | list[PlaneHistogram]) -> set[np.dtype]:
    if isinstance(items[0], PlaneHistogram):
        return {histogram.dtype for histogram in items}
    return {histogram.dtype for record in items for histogram in record.histograms}


def _channel_histograms(
    items: list[PlaneRecord] | list[PlaneHistogram], channels: tuple[int, ...]
) -> list[PlaneHistogram]:
    # The histograms of ``channels`` in every row, or of a pixel array's channels.
    if isinstance(items[0], PlaneHistogram):
        return [items[c] for c in channels]
    return [record.channel_histogram(c) for record in items for c in channels]


def _pool(
    items: list[PlaneRecord] | list[PlaneHistogram],
    settings: NormalizationSettings,
    channels: tuple[int, ...],
) -> PooledHistogram | WeightedHistogram:
    if settings.pooling != "conditions":
        return PooledHistogram(_channel_histograms(items, channels))
    # Contract §4.3: each condition's counts are scaled to the same total. The
    # largest condition is the reference, so conditions with equal pixel
    # counts get weight exactly 1 and the numbers equal those of "pixels".
    by_condition: dict[str, list[PlaneRecord]] = {}
    for record in items:
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


def _check_rgb_columns(channels: list[int], offset: list[float], scale: list[float]) -> None:
    # Contract §8.1: an RGB row holds the shared value in C0, C1 and C2.
    expected = list(range(RGB_CHANNEL_COUNT))
    if channels != expected:
        missing = [c for c in expected if c not in channels]
        names = [name for c in missing for name in (offset_column(c), scale_column(c))]
        extra = [c for c in channels if c not in expected]
        problem = (
            f"the column(s) {', '.join(names)} are blank or missing"
            if missing
            else f"it also holds channel(s) {extra}"
        )
        raise ValueError(
            "An RGB row holds one offset and scale in each of channels C0, C1 and C2, but "
            f"{problem}. Read the parameters from a row written with RGB normalization."
        )
    if len(set(offset)) != 1 or len(set(scale)) != 1:
        raise ValueError(
            "An RGB row holds the same offset and scale in channels C0, C1 and C2, but "
            f"Normalization_Offset_C0..C2 are {offset} and Normalization_Scale_C0..C2 are "
            f"{scale}. Read the parameters from a row written with RGB normalization."
        )


def _axis_index(axis: int, ndim: int) -> int:
    if not -ndim <= axis < ndim:
        raise ValueError(f"channel_axis {axis} is outside the pixel array's {ndim} axes.")
    return axis % ndim


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _number(columns: Mapping[str, Any], name: str) -> float:
    text = _text(columns[name])
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"Column {name} cannot be read as a number; got {text!r}.") from None
    if not math.isfinite(value):
        raise ValueError(f"Column {name} must hold a finite number; got {text!r}.")
    return value


def _flag(columns: Mapping[str, Any], name: str) -> bool:
    value = columns[name]
    if isinstance(value, bool | np.bool_):
        return bool(value)
    text = _text(value).strip().lower()
    if text not in ("true", "false"):
        raise ValueError(f"Column {name} must be 'true' or 'false'; got {value!r}.")
    return text == "true"
