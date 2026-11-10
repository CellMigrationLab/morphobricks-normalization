"""Normalization parameters and their columns (contract §2, §8; ADR 0003 HV2).

:class:`Parameters` holds an offset and a scale per channel of one group, each channel named
by its canonical channel ID; :func:`~morphobricks_normalization.fitting.fit` makes them. They
are stored as one ``Normalization_Offset_<ID>`` and ``Normalization_Scale_<ID>`` pair per
channel (contract §8.1, N41, N46, N47 with the channel token amended by HV2); for RGB the
shared value is written for R, G and B. ``from_columns(to_columns(p)) == p`` exactly.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .channels import CHANNEL_ID_PATTERN, check_channel_ids
from .issues import Issue
from .parameter_checks import check_offset_scale, check_scope
from .rgb import RGB_CHANNELS, apply_rgb, check_rgb_channels, check_rgb_selection
from .settings import SCOPES, NormalizationSettings

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
# The channel token is a canonical channel ID (ADR 0003 HV2), which may contain "_".
_CHANNEL_COLUMN = re.compile(rf"Normalization_(?:Offset|Scale)_({CHANNEL_ID_PATTERN.pattern})")


def offset_column(channel: str) -> str:
    """Column of channel ``channel``'s offset, e.g. ``Normalization_Offset_DNA``."""
    return f"{OFFSET_COLUMN_PREFIX}_{channel}"


def scale_column(channel: str) -> str:
    """Column of channel ``channel``'s scale, e.g. ``Normalization_Scale_DNA``."""
    return f"{SCALE_COLUMN_PREFIX}_{channel}"


@dataclass(frozen=True)
class Parameters:
    """Normalization of one group: ``(pixels - offset[k]) / scale[k]`` for channel ``channels[k]``.

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
        One value per channel, in the order of ``channels``: finite offsets,
        finite scales above zero.
    channels
        The canonical channel IDs the parameters belong to, sorted and distinct
        (ADR 0003 HV2); never C indices. A single-channel patch of channel ``DNA``
        has ``channels == ("DNA",)`` (contract §8.1). Required: nothing is guessed.
    issues
        Warnings about these parameters (``approximate_parameters``,
        ``planes_without_finite_pixels``, ``channel_partially_present``); not stored
        in the columns and not compared.
    """

    method: str
    scope: str
    group: str | None
    exact: bool
    offset: tuple[float, ...]
    scale: tuple[float, ...]
    channels: tuple[str, ...]
    issues: tuple[Issue, ...] = field(default=(), compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "offset", tuple(float(value) for value in self.offset))
        object.__setattr__(self, "scale", tuple(float(value) for value in self.scale))
        if not self.offset or len(self.offset) != len(self.scale):
            raise ValueError(
                f"Expected one offset and one scale per channel; got {len(self.offset)} "
                f"offset(s) and {len(self.scale)} scale(s)."
            )
        channels = check_channel_ids(self.channels, "Parameters")
        if len(channels) != len(self.offset) or list(channels) != sorted(channels):
            raise ValueError(
                "Expected one channel ID per offset, in sorted order; got channels "
                f"{list(channels)} for {len(self.offset)} offset(s)."
            )
        object.__setattr__(self, "channels", channels)
        for channel, offset, scale in zip(channels, self.offset, self.scale, strict=True):
            check_offset_scale(offset, scale, f"Channel {channel}")

    @property
    def channel_count(self) -> int:
        """Number of channels the parameters normalize."""
        return len(self.offset)

    @property
    def image_type(self) -> str:
        """``"rgb"`` for StarDist's RGB method (``rgb percentile ...``), else ``"fluorescence"``."""
        return "rgb" if self.method.startswith("rgb ") else "fluorescence"

    def select(self, channels: Iterable[str]) -> Parameters:
        """Return the parameters of some of the channels only, by channel ID.

        For example ``parameters.select(["DNA"])`` gives the parameters of a
        single-channel patch of channel ``DNA``, whose columns fill only
        ``Normalization_Offset_DNA`` and ``Normalization_Scale_DNA`` (contract §8.1).

        Raises
        ------
        ValueError
            If a channel is not among :attr:`channels`, or is not a channel ID (an
            index is refused); for RGB parameters, if not all of R, G and B are
            selected (issue #50).
        """
        wanted = sorted(set(check_channel_ids(channels, "Select")))
        missing = [c for c in wanted if c not in self.channels]
        if not wanted:
            raise ValueError("Select at least one channel of these parameters.")
        if missing:
            raise ValueError(
                f"These parameters have channel(s) {', '.join(self.channels)}; channel(s) "
                f"{', '.join(missing)} cannot be selected."
            )
        if self.image_type == "rgb":
            check_rgb_selection(wanted, self.channels)
        index = [self.channels.index(c) for c in wanted]
        return replace(
            self,
            offset=tuple(self.offset[i] for i in index),
            scale=tuple(self.scale[i] for i in index),
            channels=tuple(wanted),
        )

    def apply(
        self,
        pixels: np.ndarray,
        *,
        channel_axis: int | None = None,
        channels: Iterable[str] | None = None,
    ) -> np.ndarray:
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
            Axis of ``pixels`` that holds the channels; may be left out when the
            array holds one channel.
        channels
            The canonical channel ID of each entry along ``channel_axis``, in that
            order (or the one ID of an array without a channel axis): each entry is
            normalized with the parameters of its ID, never by its position. Required
            with ``channel_axis`` (for RGB parameters it defaults to ``R``, ``G``, ``B``,
            the samples of an RGB unit); may be left out without it when the parameters
            have one channel. Every ID must be among :attr:`channels`; the array may hold
            fewer channels than the parameters.

        Raises
        ------
        ValueError
            If the channel axis or the channel IDs are missing, do not match the array,
            or name a channel these parameters do not have.
        """
        values = np.asarray(pixels)
        if self.image_type == "fluorescence":
            values = values.astype(np.float64, copy=False)
        order = self._apply_order(values, channel_axis, channels)
        if channel_axis is None:
            offset, scale = self.offset[order[0]], self.scale[order[0]]
        else:
            axis = channel_axis_index(channel_axis, values.ndim)
            shape = [1] * values.ndim
            shape[axis] = len(order)
            offset = np.asarray([self.offset[k] for k in order]).reshape(shape)
            scale = np.asarray([self.scale[k] for k in order]).reshape(shape)
        if self.image_type == "rgb":
            return apply_rgb(values, offset, scale)
        return ((values - offset) / scale).astype(np.float32, copy=False)

    def _apply_order(
        self, values: np.ndarray, channel_axis: int | None, channels: Iterable[str] | None
    ) -> list[int]:
        """Positions in :attr:`channels` of the array's channels, checked against the array."""
        if channels is None and self.image_type == "rgb" and channel_axis is not None:
            channels = RGB_CHANNELS  # an RGB unit's samples (experiment-v1 §4.4b)
        if channels is None:
            if channel_axis is not None or self.channel_count != 1:
                raise ValueError(
                    f"These parameters have channel(s) {', '.join(self.channels)}; give "
                    "channels, the channel ID of each entry along channel_axis in order, so "
                    "that every channel is normalized with its own parameters."
                )
            return [0]
        ids = check_channel_ids(channels, "apply")
        count = (
            1
            if channel_axis is None
            else values.shape[channel_axis_index(channel_axis, values.ndim)]
        )
        if len(ids) != count:
            where = (
                "without a channel axis" if channel_axis is None else f"along axis {channel_axis}"
            )
            raise ValueError(
                f"The pixel array has {count} channel(s) {where}, but {len(ids)} channel ID(s) "
                f"were given ({', '.join(ids)})."
            )
        missing = [channel for channel in ids if channel not in self.channels]
        if missing:
            raise ValueError(
                f"These parameters have channel(s) {', '.join(self.channels)}, not "
                f"{', '.join(missing)}; normalize those channels with parameters fitted for them."
            )
        if self.image_type == "rgb":
            check_rgb_channels(ids, "The pixel array")
        return [self.channels.index(channel) for channel in ids]

    def to_columns(self) -> dict[str, str]:
        """Return the columns of contract §8.1 as text, numbers in full.

        The four ``normalization_*`` columns, then ``Normalization_Offset_<ID>``
        and ``Normalization_Scale_<ID>`` for each channel ID of :attr:`channels`,
        grouped by channel in sorted order. Columns of other channels are not
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
            ``Normalization_Offset_<ID>`` and ``Normalization_Scale_<ID>`` pair of at
            least one channel ID; other columns are ignored. Values may be text or
            numbers. A channel whose two columns are both blank or absent is not part
            of the row's parameters (a single-channel patch fills only its own
            channel, and a channel absent from the row's image is blank).
        settings
            The settings of the run that wrote the row; its method must be the
            row's ``normalization_method`` and its scope the row's
            ``normalization_scope`` (one of the scopes of contract §4.1), so that
            parameters fitted for one sampling population are never read as
            another's (issue #37).

        Raises
        ------
        ValueError
            Naming the column that is missing or cannot be read, whose method
            or scope differs from the settings, or whose scope is unknown; also
            for a channel with only one of its two columns.
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
        ids = sorted(
            {match[1] for name in columns if (match := _CHANNEL_COLUMN.fullmatch(str(name)))}
        )
        if not ids:
            raise ValueError(
                "The row has no Normalization_Offset_<ID> and Normalization_Scale_<ID> columns "
                "(one pair per canonical channel ID). Read the parameters from a table written "
                "with normalization."
            )
        channels, offset, scale = [], [], []
        for channel in ids:
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
                "Every Normalization_Offset_<ID> and Normalization_Scale_<ID> column of this "
                "row is blank, so it holds no parameters. Read the parameters from a row "
                "written with normalization."
            )
        if settings.image_type == "rgb":
            _check_rgb_columns(channels, offset, scale)
        return cls(
            method=method,
            scope=check_scope(_text(columns["normalization_scope"]), settings.scope, SCOPES),
            group=_text(columns["normalization_group"]) or None,
            exact=_flag(columns, "normalization_exact"),
            offset=tuple(offset),
            scale=tuple(scale),
            channels=tuple(channels),
        )


def _check_rgb_columns(channels: list[str], offset: list[float], scale: list[float]) -> None:
    # Contract §8.1: an RGB row holds the shared value in R, G and B.
    expected = sorted(RGB_CHANNELS)
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
            "An RGB row holds one offset and scale in each of channels R, G and B, but "
            f"{problem}. Read the parameters from a row written with RGB normalization."
        )
    if len(set(offset)) != 1 or len(set(scale)) != 1:
        raise ValueError(
            "An RGB row holds the same offset and scale in channels R, G and B, but "
            f"Normalization_Offset_B, _G, _R are {offset} and Normalization_Scale_B, _G, _R are "
            f"{scale}. Read the parameters from a row written with RGB normalization."
        )


def channel_axis_index(axis: int, ndim: int) -> int:
    """Return ``axis`` as a non-negative axis of an array of ``ndim`` axes, or refuse it."""
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
