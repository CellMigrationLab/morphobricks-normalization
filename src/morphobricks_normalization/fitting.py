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
separately.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
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
from .records import PlaneRecord
from .settings import NormalizationSettings, check_settings

COLUMNS = (
    "normalization_method",
    "normalization_scope",
    "normalization_group",
    "normalization_exact",
    "normalization_offset",
    "normalization_scale",
)


@dataclass(frozen=True)
class Parameters:
    """Normalization of one group: ``(pixels - offset[c]) / scale[c]`` for channel ``c``.

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
        One value per channel, in channel order.
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
    issues: tuple[Issue, ...] = field(default=(), compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "offset", tuple(float(value) for value in self.offset))
        object.__setattr__(self, "scale", tuple(float(value) for value in self.scale))
        if not self.offset or len(self.offset) != len(self.scale):
            raise ValueError(
                f"Expected one offset and one scale per channel; got {len(self.offset)} "
                f"offset(s) and {len(self.scale)} scale(s)."
            )

    @property
    def channel_count(self) -> int:
        """Number of channels the parameters normalize."""
        return len(self.offset)

    def apply(self, pixels: np.ndarray, *, channel_axis: int | None = None) -> np.ndarray:
        """Return the normalized pixels as a new float32 array.

        Computed in float64 as ``(pixels - offset) / scale`` and returned as
        float32, without clipping; non-finite pixels stay non-finite and the
        input is never modified.

        Parameters
        ----------
        pixels
            Pixel values of the group.
        channel_axis
            Axis of ``pixels`` that holds the channels; may be left out when the
            parameters have one channel.

        Raises
        ------
        ValueError
            If the channel axis is missing or does not hold one entry per channel.
        """
        values = np.asarray(pixels, dtype=np.float64)
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
        return ((values - offset) / scale).astype(np.float32, copy=False)

    def to_columns(self) -> dict[str, str]:
        """Return the ``normalization_*`` columns of contract §8.1 as text, numbers in full."""
        return {
            "normalization_method": self.method,
            "normalization_scope": self.scope,
            "normalization_group": self.group or "",
            "normalization_exact": "true" if self.exact else "false",
            "normalization_offset": ";".join(repr(value) for value in self.offset),
            "normalization_scale": ";".join(repr(value) for value in self.scale),
        }

    @classmethod
    def from_columns(
        cls, columns: Mapping[str, Any], settings: NormalizationSettings
    ) -> Parameters:
        """Read parameters back from their columns; ``from_columns(p.to_columns()) == p``.

        Parameters
        ----------
        columns
            A row with at least the six ``normalization_*`` columns; other
            columns are ignored. Values may be text or numbers.
        settings
            The settings of the run that wrote the row; its method must be the
            row's ``normalization_method``.

        Raises
        ------
        ValueError
            Naming the column that is missing or cannot be read, or whose
            method differs from the settings.
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
        offset = _numbers(columns, "normalization_offset")
        scale = _numbers(columns, "normalization_scale")
        if len(offset) != len(scale):
            raise ValueError(
                f"Column normalization_scale has {len(scale)} value(s) but "
                f"normalization_offset has {len(offset)}; each needs one value per channel."
            )
        if not all(value > 0 for value in scale):
            raise ValueError(
                "Column normalization_scale must hold positive numbers; got "
                f"{_text(columns['normalization_scale'])!r}."
            )
        return cls(
            method=method,
            scope=_text(columns["normalization_scope"]),
            group=_text(columns["normalization_group"]) or None,
            exact=_flag(columns, "normalization_exact"),
            offset=offset,
            scale=scale,
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
        The ``plane_histograms.csv`` rows of one group (every channel; each
        channel gets its own parameters, in the order of ``c``), or a pixel
        array of one image or patch (image and patch scopes only).
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
        If a channel's scale is zero or not finite.
    """
    from_pixels = isinstance(statistics, np.ndarray)
    check_settings(settings, statistics_file=not from_pixels)
    if settings.image_type == "rgb":
        raise NotImplementedError(
            "RGB normalization is not available in this version of morphobricks-normalization "
            "yet. Use image type 'fluorescence', or a version with the RGB method."
        )
    if from_pixels:
        channels = _pixel_channels(statistics, channel_axis)
        excluded: list[str] = []
    else:
        if channel_axis is not None:
            raise ValueError("channel_axis applies to pixel arrays only, not to histogram rows.")
        records = _records(statistics)
        channels = {}
        for record in sorted(records, key=lambda r: r.c):
            channels.setdefault(record.c, []).append(record)
        excluded = [record.plane_id for record in records if record.histogram.is_empty]
    parameters, exact = [], True
    for channel, items in channels.items():
        where = f"Group '{group}', channel {channel}" if group else f"Channel {channel}"
        try:
            pooled = _pool(items, settings)
        except NoValidPixelsError as exc:
            raise NoValidPixelsError(f"{where}: {exc}") from exc
        try:
            parameters.append(_channel_parameters(pooled, settings))
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc
        exact = exact and pooled.max_representative_error == 0
    return Parameters(
        method=parameters[0].method,
        scope=settings.scope,
        group=group,
        exact=exact,
        offset=tuple(p.offset for p in parameters),
        scale=tuple(p.scale for p in parameters),
        issues=_fit_issues(group, exact, excluded),
    )


def _records(statistics: Iterable[PlaneRecord]) -> list[PlaneRecord]:
    records = list(statistics)
    if not records:
        raise ValueError("No plane_histograms.csv rows were given for this group.")
    if not all(isinstance(record, PlaneRecord) for record in records):
        raise TypeError(
            "fit expects the plane_histograms.csv rows of one group (PlaneRecord objects) "
            "or a pixel array."
        )
    return records


def _pixel_channels(pixels: np.ndarray, channel_axis: int | None) -> dict[int, list]:
    if channel_axis is None:
        return {0: [compute_plane_histogram(pixels)]}
    axis = _axis_index(channel_axis, pixels.ndim)
    return {
        c: [compute_plane_histogram(np.take(pixels, c, axis=axis))]
        for c in range(pixels.shape[axis])
    }


def _pool(
    items: list[PlaneRecord] | list[PlaneHistogram], settings: NormalizationSettings
) -> PooledHistogram | WeightedHistogram:
    if settings.pooling != "conditions":
        return PooledHistogram(items)
    # Contract §4.3: each condition's counts are scaled to the same total. The
    # largest condition is the reference, so conditions with equal pixel
    # counts get weight exactly 1 and the numbers equal those of "pixels".
    by_condition: dict[str, list[PlaneRecord]] = {}
    for record in items:
        by_condition.setdefault(record.condition, []).append(record)
    parts = [
        PooledHistogram(members)
        for _, members in sorted(by_condition.items())
        if any(not record.histogram.is_empty for record in members)
    ]
    if not parts:
        return PooledHistogram(items)  # raises NoValidPixelsError
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
                f"{len(excluded)} plane(s) without finite pixels were left out of {name}; its "
                "parameters come from its other planes.",
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


def _axis_index(axis: int, ndim: int) -> int:
    if not -ndim <= axis < ndim:
        raise ValueError(f"channel_axis {axis} is outside the pixel array's {ndim} axes.")
    return axis % ndim


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _numbers(columns: Mapping[str, Any], name: str) -> tuple[float, ...]:
    text = _text(columns[name])
    try:
        values = tuple(float(part) for part in text.split(";"))
    except ValueError:
        raise ValueError(
            f"Column {name} cannot be read as numbers separated by ';'; got {text!r}."
        ) from None
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Column {name} must hold finite numbers; got {text!r}.")
    return values


def _flag(columns: Mapping[str, Any], name: str) -> bool:
    value = columns[name]
    if isinstance(value, bool | np.bool_):
        return bool(value)
    text = _text(value).strip().lower()
    if text not in ("true", "false"):
        raise ValueError(f"Column {name} must be 'true' or 'false'; got {value!r}.")
    return text == "true"
