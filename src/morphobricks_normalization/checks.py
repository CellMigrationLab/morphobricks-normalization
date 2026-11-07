"""Intensity checks on per-image summaries (contract §7.2, N38).

Four warnings, computed from :class:`~morphobricks_normalization.summaries.ImageSummary`
objects alone:

``image_intensity_outlier``
    An image's median or high percentile in a channel is far from the other
    images of its condition and biological repeat.
``repeat_intensity_shift``
    The median of a biological repeat's image medians is far from the image
    medians of the other repeats of its condition.
``saturated_image``
    Too many pixels at the pixel type's maximum.
``no_intensity_spread``
    The low and high percentiles are equal.

Images are compared only within one condition, so a treatment effect is never
reported. "Far" is measured in robust deviations, ``MAD × 1.482602218505602``
of the values compared with, around their median; a zero robust deviation
(identical values) skips the comparison instead of firing. A comparison with
fewer than ``min_images`` values is skipped (N53). Each finding is one
:class:`~morphobricks_normalization.issues.Issue` naming the image or repeat,
the channel (also in ``Issue.channel``), the value and the group value (N52).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._version import __version__
from .issues import (
    CHANNEL_WITHOUT_FINITE_PIXELS,
    IMAGE_INTENSITY_OUTLIER,
    NO_INTENSITY_SPREAD,
    REPEAT_INTENSITY_SHIFT,
    SATURATED_IMAGE,
    WARNING,
    Issue,
)
from .normalization import NORMAL_CONSISTENCY
from .records import different_channel_counts
from .summaries import ImageSummary, IncompatibleSummariesError

_KEYS = ("library_version", "outlier_limit", "saturation_limit", "min_images")

_OUTLIER_ACTION = (
    "Look at the image: an image far from the others of its biological repeat may be out of "
    "focus, contain debris or bubbles, or have been acquired with other settings. If it is "
    "faulty, remove it from the experiment and run the preprocessing brick again; if the "
    "difference is real, no action is needed."
)
_SHIFT_ACTION = (
    "Check whether this biological repeat was prepared or acquired differently from the others "
    "(staining, exposure, laser power, microscope settings). Normalizing at the biological-"
    "repeat scope removes intensity differences between repeats; if the repeat cannot be "
    "compared with the others, exclude it. If the difference is expected, no action is needed."
)
_SATURATED_ACTION = (
    "Saturated pixels have lost their true intensity, so intensities in those regions are "
    "underestimated. Check the exposure, gain or laser power, and consider excluding the image "
    "or its saturated regions from intensity measurements."
)
_UNASSESSED_ACTION = (
    "Check the image: this channel is empty or holds only NaN or infinite values, so it was not "
    "checked and cannot be normalized from its own pixels. Exclude the image, or fix the channel "
    "in the source data and run the preprocessing brick again."
)
_SPREAD_ACTION = (
    "Check whether the channel is empty or the image is blank. Exclude the image, or normalize "
    "it at the biological-repeat scope, which pools it with the other images of its repeat."
)


@dataclass(frozen=True)
class IntensityCheckSettings:
    """Limits of the intensity checks (contract §7.2, N51).

    Attributes
    ----------
    outlier_limit
        Robust deviations beyond which an image or biological repeat is
        reported; default 3.0.
    saturation_limit
        Fraction of saturated pixels above which an image is reported, from 0
        (any saturated pixel) to below 1; default 0.01.
    min_images
        Fewest values a comparison needs: the images of a condition and
        biological repeat for the outlier check, the images of the other
        repeats for the repeat check; default 5 (**Provisional**, plan §13).

    Raises
    ------
    ValueError
        For a limit outside its range.
    """

    outlier_limit: float = 3.0
    saturation_limit: float = 0.01
    min_images: int = 5

    def __post_init__(self) -> None:
        outlier_limit = _number(self.outlier_limit, "outlier_limit")
        if not outlier_limit > 0:
            raise ValueError(
                f"outlier_limit must be a positive number of robust deviations; got "
                f"{self.outlier_limit!r}."
            )
        saturation_limit = _number(self.saturation_limit, "saturation_limit")
        if not 0 <= saturation_limit < 1:
            raise ValueError(
                "saturation_limit must be a fraction of the pixels, at least 0 and below 1 "
                f"(0.01 is 1 %); got {self.saturation_limit!r}."
            )
        min_images = self.min_images
        if isinstance(min_images, bool) or not isinstance(min_images, (int, np.integer)):
            raise ValueError(f"min_images must be a whole number; got {min_images!r}.")
        if min_images < 2:
            raise ValueError(
                f"min_images must be at least 2, since an image is compared with the others; "
                f"got {min_images}."
            )
        object.__setattr__(self, "outlier_limit", outlier_limit)
        object.__setattr__(self, "saturation_limit", saturation_limit)
        object.__setattr__(self, "min_images", int(min_images))

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible limits, for the calling brick's run configuration."""
        return {
            "library_version": __version__,
            "outlier_limit": self.outlier_limit,
            "saturation_limit": self.saturation_limit,
            "min_images": self.min_images,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> IntensityCheckSettings:
        """Read limits written by :meth:`to_dict`; missing keys take their defaults.

        ``library_version`` is the version that wrote the limits and is not checked.

        Raises
        ------
        ValueError
            For an unknown key, or a value outside its range.
        """
        unknown = sorted(set(data) - set(_KEYS))
        if unknown:
            raise ValueError(
                f"Unknown intensity check setting(s): {', '.join(unknown)}. Known settings: "
                f"{', '.join(_KEYS)}. Check the spelling in the run configuration."
            )
        return cls(**{key: data[key] for key in _KEYS[1:] if key in data})


def intensity_checks(
    summaries: Iterable[ImageSummary], settings: IntensityCheckSettings | None = None
) -> list[Issue]:
    """Return the warnings of contract §7.2 for the images of ``summaries``.

    Parameters
    ----------
    summaries
        One summary per image, from
        :func:`~morphobricks_normalization.summaries.image_summaries`.
    settings
        The limits; None uses the defaults.

    Returns
    -------
    list of Issue
        One per finding, in this order: image outliers, repeat shifts, saturated
        images, images without spread; each sorted by condition, biological
        repeat, image and channel. Empty when nothing is found.
    """
    settings = IntensityCheckSettings() if settings is None else settings
    summaries = sorted(summaries, key=_order)
    _check_comparable(summaries)
    _check_same_percentiles(summaries)
    return [
        *_image_outliers(summaries, settings),
        *_repeat_shifts(summaries, settings),
        *_saturated_images(summaries, settings),
        *_no_spread(summaries),
    ]


def unchecked_channels(summaries: Iterable[ImageSummary]) -> list[Issue]:
    """Return a ``channel_without_finite_pixels`` warning per image channel that was not checked.

    A channel without finite pixels (all NaN or infinite) has blank summaries,
    so :func:`intensity_checks` cannot check it and stays silent about it; this
    names it instead, so that silence is not read as a healthy channel (issue
    #57). It is separate from :func:`intensity_checks`, whose four codes the
    calling bricks list.

    Parameters
    ----------
    summaries
        One summary per image, from
        :func:`~morphobricks_normalization.summaries.image_summaries`.

    Returns
    -------
    list of Issue
        One per image and channel without finite pixels, sorted by condition,
        biological repeat, image and channel; ``Issue.channel`` is set.
    """
    return [
        Issue(
            WARNING,
            CHANNEL_WITHOUT_FINITE_PIXELS,
            f"Image {summary.source_relative_path}, channel C{channel}: it has no finite pixels "
            "(every value is NaN or infinite), so its intensities were not checked.",
            items=(summary.source_relative_path,),
            action=_UNASSESSED_ACTION,
            channel=channel,
        )
        for summary in sorted(summaries, key=_order)
        for channel, median in enumerate(summary.median)
        if median is None
    ]


def _image_outliers(summaries: list[ImageSummary], settings: IntensityCheckSettings) -> list[Issue]:
    groups: dict[tuple[str, str], list[ImageSummary]] = {}
    for summary in summaries:
        groups.setdefault((summary.condition, summary.biological_repeat), []).append(summary)
    found = []
    for (condition, repeat), members in groups.items():
        high = f"{_ordinal(members[0].percentile[1])} percentile"
        for channel in range(members[0].channel_count):
            for order, (statistic, name) in enumerate(
                (("median", "median intensity"), ("p_high", high))
            ):
                present = [
                    (s, getattr(s, statistic)[channel])
                    for s in members
                    if getattr(s, statistic)[channel] is not None
                ]
                if len(present) < settings.min_images:
                    continue
                for index, (summary, value) in enumerate(present):
                    others = [v for j, (_, v) in enumerate(present) if j != index]
                    comparison = _compare(value, others, settings.outlier_limit)
                    if comparison is None:
                        continue
                    reference, deviation, distance = comparison
                    message = (
                        f"Image {summary.source_relative_path}, channel C{channel}: its {name} "
                        f"{_text(value)} is {distance:.3g} robust deviations from "
                        f"{_text(reference)}, the median of the other {len(others)} images of "
                        f"condition '{condition}', biological repeat '{repeat}' (robust "
                        f"deviation {_text(deviation)}; limit {settings.outlier_limit:g})."
                    )
                    issue = Issue(
                        WARNING,
                        IMAGE_INTENSITY_OUTLIER,
                        message,
                        items=(summary.source_relative_path,),
                        action=_OUTLIER_ACTION,
                        channel=channel,
                    )
                    found.append(((summary.source_relative_path, channel, order), issue))
    # Within a group, every finding of one image together, channel by channel.
    keys = {s.source_relative_path: i for i, s in enumerate(summaries)}
    found.sort(key=lambda item: (keys[item[0][0]], *item[0][1:]))
    return [issue for _, issue in found]


def _repeat_shifts(summaries: list[ImageSummary], settings: IntensityCheckSettings) -> list[Issue]:
    conditions: dict[str, dict[str, list[ImageSummary]]] = {}
    for summary in summaries:
        repeats = conditions.setdefault(summary.condition, {})
        repeats.setdefault(summary.biological_repeat, []).append(summary)
    issues = []
    for condition, repeats in conditions.items():
        if len(repeats) < 2:
            continue
        for repeat in repeats:
            for channel in range(summaries[0].channel_count):
                medians = {
                    label: [s.median[channel] for s in members if s.median[channel] is not None]
                    for label, members in repeats.items()
                }
                own = medians[repeat]
                others = [v for label, values in medians.items() if label != repeat for v in values]
                if not own or len(others) < settings.min_images:
                    continue
                value = float(np.median(own))
                comparison = _compare(value, others, settings.outlier_limit)
                if comparison is None:
                    continue
                reference, deviation, distance = comparison
                message = (
                    f"Biological repeat '{repeat}' of condition '{condition}', channel "
                    f"C{channel}: the median of its {len(own)} image medians, {_text(value)}, is "
                    f"{distance:.3g} robust deviations from {_text(reference)}, the median of "
                    f"the {len(others)} image medians of the other biological repeats of "
                    f"'{condition}' (robust deviation {_text(deviation)}; limit "
                    f"{settings.outlier_limit:g})."
                )
                issues.append(
                    Issue(
                        WARNING,
                        REPEAT_INTENSITY_SHIFT,
                        message,
                        items=(f"{condition}/{repeat}",),
                        action=_SHIFT_ACTION,
                        channel=channel,
                    )
                )
    return issues


def _saturated_images(
    summaries: list[ImageSummary], settings: IntensityCheckSettings
) -> list[Issue]:
    issues = []
    for summary in summaries:
        for channel, fraction in enumerate(summary.saturated_fraction):
            if fraction is None or not fraction > settings.saturation_limit:
                continue
            top = np.iinfo(summary.dtype).max
            message = (
                f"Image {summary.source_relative_path}, channel C{channel}: {100 * fraction:.3g} % "
                f"of its pixels are at the maximum of its pixel type, {top} ({summary.dtype}), "
                f"above the limit of {100 * settings.saturation_limit:g} %."
            )
            issues.append(
                Issue(
                    WARNING,
                    SATURATED_IMAGE,
                    message,
                    items=(summary.source_relative_path,),
                    action=_SATURATED_ACTION,
                    channel=channel,
                )
            )
    return issues


def _no_spread(summaries: list[ImageSummary]) -> list[Issue]:
    issues = []
    for summary in summaries:
        low, high = (_ordinal(q) for q in summary.percentile)
        for channel, (p_low, p_high) in enumerate(zip(summary.p_low, summary.p_high, strict=True)):
            if p_low is None or p_low != p_high:
                continue
            message = (
                f"Image {summary.source_relative_path}, channel C{channel}: its {low} and {high} "
                f"percentiles are both {_text(p_low)}, so this channel has no intensity spread "
                "and cannot be normalized at the image scope with a percentile method."
            )
            issues.append(
                Issue(
                    WARNING,
                    NO_INTENSITY_SPREAD,
                    message,
                    items=(summary.source_relative_path,),
                    action=_SPREAD_ACTION,
                    channel=channel,
                )
            )
    return issues


def _compare(value: float, others: list[float], limit: float) -> tuple[float, float, float] | None:
    """Return ``(reference, robust deviation, distance)`` if ``value`` is beyond ``limit``.

    None when it is within the limit, or the robust deviation is zero.

    The reference is the median of ``others`` and the robust deviation their
    MAD around it times ``NORMAL_CONSISTENCY``.
    """
    values = np.asarray(others, dtype=np.float64)
    reference = float(np.median(values))
    deviation = NORMAL_CONSISTENCY * float(np.median(np.abs(values - reference)))
    if not deviation > 0:
        return None
    distance = abs(value - reference) / deviation
    return (reference, deviation, distance) if distance > limit else None


def _check_comparable(summaries: list[ImageSummary]) -> None:
    """Refuse summaries that do not describe the same channels (issue #56)."""
    if len({s.channel_count for s in summaries}) > 1:
        raise IncompatibleSummariesError(
            different_channel_counts(
                [(s.source_relative_path, s.channel_count) for s in summaries], "These images"
            )
        )


def _check_same_percentiles(summaries: list[ImageSummary]) -> None:
    """Refuse summaries computed with different percentile definitions (issue #55)."""
    by_definition: dict[tuple[float, float], list[str]] = {}
    for s in summaries:
        by_definition.setdefault(tuple(s.percentile), []).append(s.source_relative_path)
    if len(by_definition) > 1:
        parts = []
        for (low, high), names in sorted(by_definition.items()):
            shown = ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")
            parts.append(f"percentiles ({low:g}, {high:g}): {shown}")
        raise IncompatibleSummariesError(
            f"These images have summaries with different percentile definitions "
            f"({'; '.join(parts)}). Values at different percentiles are different measurements and "
            "cannot be compared. Summarize every "
            "image with one percentile pair, with image_summaries(records, percentile=...)."
        )


def _order(summary: ImageSummary) -> tuple[str, ...]:
    # The order of image_summaries: labels from the top down, then the path.
    return (
        summary.condition,
        summary.biological_repeat,
        summary.technical_repeat,
        summary.source_relative_path,
    )


def _ordinal(q: float) -> str:
    # "1st", "2nd", "99.8th": a suffix after the last digit of whole numbers only.
    text = f"{q:g}"
    if not text.isdigit():
        return f"{text}th"
    number = int(text)
    if number % 100 in (11, 12, 13):
        return f"{text}th"
    return text + {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")


def _text(value: float) -> str:
    return f"{value:.6g}"


def _number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = math.nan
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a number; got {value!r}.")
    return number
