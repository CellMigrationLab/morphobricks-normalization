"""Normalization settings and the checks that need nothing but the settings.

A :class:`NormalizationSettings` holds every choice of a run: image type,
method and its options, scope, pooling weight and per-time-point groups
(contract §8.3). The calling brick stores ``settings.to_dict()`` once per run
and reads it back with :meth:`NormalizationSettings.from_dict`.

Options that have no effect for the chosen method or scope (for example the
percentiles of a z-score, or the pooling weight of the image scope) are reset to
their defaults and left out of ``to_dict()``, so settings that normalize the
same way compare equal and round-trip exactly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._version import __version__
from .issues import BIOLOGICAL_REPEAT_SCOPE, WARNING, Issue
from .normalization import NORMAL_CONSISTENCY

IMAGE_TYPES = ("fluorescence", "rgb")
FLUORESCENCE_METHODS = ("minmax", "percentile", "zscore", "robust_zscore")
RGB_METHODS = ("percentile",)
SCOPES = ("biological_repeat", "image", "patch")
POOLING_WEIGHTS = ("pixels", "conditions")
DEFAULT_PERCENTILE = (1.0, 99.8)

# Written names of the methods without options (contract §3.1); the same names
# the functions of ``normalization.py`` give their parameters.
_METHOD_NAMES = {"minmax": "min-max", "zscore": "z-score", "robust_zscore": "robust z-score"}

_KEYS = (
    "library_version",
    "image_type",
    "method",
    "percentile",
    "consistency",
    "scope",
    "pooling",
    "per_time_point",
)


_RGB_METHOD_MESSAGE = (
    "RGB images are normalized with StarDist's percentile method only, which scales the three "
    "colour channels together. Use method 'percentile'."
)


class MethodNotAllowedError(ValueError):
    """Raised for a method the image type does not allow, or a scope the statistics cannot give."""


@dataclass(frozen=True)
class NormalizationSettings:
    """Every normalization choice of one run (contract §8.3).

    Attributes
    ----------
    image_type
        ``"fluorescence"`` (default): each channel separately; ``"rgb"``: R, G and B
        together, with StarDist's percentile normalization (contract §3.3).
    method
        ``"minmax"``, ``"percentile"`` (default), ``"zscore"`` or ``"robust_zscore"``;
        ``"percentile"`` only for RGB images.
    percentile
        ``(low, high)`` in percent for the percentile method, ``0 <= low < high <= 100``;
        default ``(1.0, 99.8)``.
    consistency
        Factor applied to the MAD by the robust z-score; default ``NORMAL_CONSISTENCY``.
    scope
        ``"biological_repeat"`` (one repeat group across all conditions), ``"image"``
        (default) or ``"patch"``.
    pooling
        At the biological-repeat scope, how much each condition weighs: ``"pixels"``
        (default, each pixel once) or ``"conditions"`` (each condition equally).
    per_time_point
        Whether each time point is a group of its own (default False: time points pooled).

    Raises
    ------
    MethodNotAllowedError
        For an RGB image type with a method other than ``"percentile"``.
    ValueError
        For an unknown image type, method, scope or pooling weight, or invalid
        percentiles or consistency factor.
    """

    image_type: str = "fluorescence"
    method: str = "percentile"
    percentile: tuple[float, float] = DEFAULT_PERCENTILE
    consistency: float = NORMAL_CONSISTENCY
    scope: str = "image"
    pooling: str = "pixels"
    per_time_point: bool = False

    def __post_init__(self) -> None:
        if self.image_type not in IMAGE_TYPES:
            raise ValueError(
                f"Image type {self.image_type!r} does not exist. Choose 'fluorescence' "
                "(each channel normalized separately) or 'rgb' (colour images)."
            )
        if self.method not in FLUORESCENCE_METHODS:
            raise ValueError(
                f"Method {self.method!r} does not exist. Choose 'minmax', 'percentile', "
                "'zscore' or 'robust_zscore' for fluorescence images, or 'percentile' for "
                "RGB images."
            )
        if self.image_type == "rgb" and self.method not in RGB_METHODS:
            raise MethodNotAllowedError(_RGB_METHOD_MESSAGE)
        if self.scope not in SCOPES:
            raise ValueError(
                f"Scope '{self.scope}' does not exist. Choose 'biological_repeat' (one "
                "biological repeat across all conditions), 'image' or 'patch'."
            )
        if self.scope == "biological_repeat" and self.pooling not in POOLING_WEIGHTS:
            raise ValueError(
                f"Pooling {self.pooling!r} does not exist. Choose 'pixels' (each pixel counts "
                "once) or 'conditions' (each condition weighs the same)."
            )
        if not isinstance(self.per_time_point, bool):
            raise ValueError(f"per_time_point must be true or false; got {self.per_time_point!r}.")
        percentile = DEFAULT_PERCENTILE
        if self.method == "percentile":
            percentile = _percentile_pair(self.percentile)
        consistency = NORMAL_CONSISTENCY
        if self.method == "robust_zscore":
            consistency = _positive_number(self.consistency, "consistency")
        object.__setattr__(self, "percentile", percentile)
        object.__setattr__(self, "consistency", consistency)
        if self.scope != "biological_repeat":
            object.__setattr__(self, "pooling", "pixels")

    @property
    def method_name(self) -> str:
        """Written name of the method, as in ``normalization_method`` (contract §3.1)."""
        if self.method == "percentile":
            low, high = self.percentile
            name = f"percentile {low:g}-{high:g}"
            return f"rgb {name}" if self.image_type == "rgb" else name
        return _METHOD_NAMES[self.method]

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible settings for the run configuration, without options that do not apply."""
        data: dict[str, Any] = {
            "library_version": __version__,
            "image_type": self.image_type,
            "method": self.method,
        }
        if self.method == "percentile":
            data["percentile"] = list(self.percentile)
        if self.method == "robust_zscore":
            data["consistency"] = self.consistency
        data["scope"] = self.scope
        if self.scope == "biological_repeat":
            data["pooling"] = self.pooling
        data["per_time_point"] = self.per_time_point
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NormalizationSettings:
        """Read settings written by :meth:`to_dict`; missing keys take their defaults.

        ``library_version`` is the version that wrote the settings and is not
        checked.

        Raises
        ------
        ValueError
            For an unknown key, or a value the settings do not accept.
        """
        unknown = sorted(set(data) - set(_KEYS))
        if unknown:
            raise ValueError(
                f"Unknown normalization setting(s): {', '.join(unknown)}. Known settings: "
                f"{', '.join(_KEYS)}. Check the spelling in the run configuration."
            )
        values = {key: data[key] for key in _KEYS[1:] if key in data}
        if "percentile" in values:
            values["percentile"] = tuple(values["percentile"])
        return cls(**values)


def check_settings(settings: NormalizationSettings, *, statistics_file: bool) -> list[Issue]:
    """Check the settings before any image is read (contract §2.2).

    Parameters
    ----------
    settings
        The run's settings.
    statistics_file
        Whether the preprocessing brick's ``plane_histograms.csv`` is available.

    Returns
    -------
    list of Issue
        Warnings about the choice, for the calling brick to show
        (``biological_repeat_scope``).

    Raises
    ------
    MethodNotAllowedError
        If the image type does not allow the method, or the scope needs the
        statistics file and it is not available.
    """
    if settings.image_type == "rgb" and settings.method not in RGB_METHODS:
        raise MethodNotAllowedError(_RGB_METHOD_MESSAGE)
    if settings.scope == "biological_repeat" and not statistics_file:
        raise MethodNotAllowedError(
            "The biological-repeat scope needs the preprocessing brick's plane_histograms.csv, "
            "which was not given. Run the preprocessing brick on this experiment, or use the "
            "image or patch scope."
        )
    issues = []
    if settings.scope == "biological_repeat":
        issues.append(
            Issue(
                WARNING,
                BIOLOGICAL_REPEAT_SCOPE,
                "Each biological repeat is normalized with the pixels of all its conditions "
                "together: intensity differences between biological repeats are removed, "
                "differences between conditions within a repeat are kept.",
                action="Use this scope only when the biological repeats were prepared and "
                "acquired under the same circumstances, as confirmed in the preprocessing brick.",
            )
        )
    return issues


def _percentile_pair(value: Any) -> tuple[float, float]:
    try:
        if isinstance(value, str):
            raise TypeError
        low, high = (float(number) for number in value)
    except (TypeError, ValueError):
        raise ValueError(f"percentile must be two numbers, [low, high]; got {value!r}.") from None
    if not 0 <= low < high <= 100:
        raise ValueError(f"Expected 0 <= low < high <= 100; got low={low}, high={high}.")
    return low, high


def _positive_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = math.nan
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive number; got {value!r}.")
    return number
