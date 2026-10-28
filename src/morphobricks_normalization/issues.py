"""Warnings returned to the calling brick (contract §9.2).

The library never prints. Errors are raised as exceptions; warnings are
returned as :class:`Issue` objects, and the calling brick shows them to the
user and writes them to its log.
"""

from __future__ import annotations

from dataclasses import dataclass

WARNING = "warning"

# Warning codes of contract §9.2 produced in Milestone 2.
BIOLOGICAL_REPEAT_SCOPE = "biological_repeat_scope"
REPEAT_MISSING_CONDITIONS = "repeat_missing_conditions"
APPROXIMATE_PARAMETERS = "approximate_parameters"
PLANES_WITHOUT_FINITE_PIXELS = "planes_without_finite_pixels"

# Intensity checks of contract §7.2, Milestone 4.
IMAGE_INTENSITY_OUTLIER = "image_intensity_outlier"
REPEAT_INTENSITY_SHIFT = "repeat_intensity_shift"
SATURATED_IMAGE = "saturated_image"
NO_INTENSITY_SPREAD = "no_intensity_spread"

# Histogram range check of contract §1.4, Milestone 5; reported by the preprocessing brick.
NARROW_HISTOGRAM_RANGE = "narrow_histogram_range"


@dataclass(frozen=True)
class Issue:
    """One warning for the calling brick to report.

    Attributes
    ----------
    severity
        ``"warning"``: processing continues.
    code
        Stable identifier of the situation (contract §9.2), for filtering and tests.
    message
        Plain-language description of what happened.
    items
        The affected images, planes or groups.
    action
        What the user can do about it.
    channel
        The zero-based channel the warning is about, for the intensity checks
        (N52), so a calling brick can mark the image in that channel's plots;
        None for warnings about whole images or groups.
    """

    severity: str
    code: str
    message: str
    items: tuple[str, ...] = ()
    action: str = ""
    channel: int | None = None
