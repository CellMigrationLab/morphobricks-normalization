"""The one region checker and the region estimate (MorphoBricks-Spec ADR 0005, BM3, BM7, BM8).

A mask used as a **region** is ``mask != 0``, whatever its representation, and is never
labelled. :func:`check_region` is the checker every MorphoBricks brick calls when it takes a
region, in memory or from a run; it lives here because normalization is the one package every
brick can import (it depends on NumPy only). It returns the boolean foreground or raises
``MASK_REGION_INVALID`` (:class:`RegionMaskError`) with one of five reasons. It never labels,
casts, thresholds, squeezes or resizes, and it returns an empty foreground instead of refusing
it: what is undefined over an empty region is decided by the measurement.

An estimate taken over a region's pixels (a white, a stain vector) is a :class:`RegionEstimate`:
its value, or ``None`` with the reason ``empty_region`` when the region holds no pixel. NaN never
stands for an undefined estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "EMPTY_REGION",
    "INVALID_REASONS",
    "MASK_REGION_INVALID",
    "REGION_REASONS",
    "RegionEstimate",
    "RegionMaskError",
    "check_region",
]

MASK_REGION_INVALID = "MASK_REGION_INVALID"
"""Runtime code of a region :func:`check_region` refuses."""
INVALID_REASONS = ("not_an_array", "masked_array", "not_2d", "shape", "dtype")
"""The reasons of ``MASK_REGION_INVALID``, in the order :func:`check_region` tests them."""
EMPTY_REGION = "empty_region"
"""Undefined reason of a value taken over a region that holds no pixel (ADR 0005, BM8)."""
REGION_REASONS = (EMPTY_REGION,)
"""Every reason a :class:`RegionEstimate` may give."""


class RegionMaskError(ValueError):
    """A region refused by :func:`check_region` (code ``MASK_REGION_INVALID``).

    Attributes
    ----------
    code
        Always ``"MASK_REGION_INVALID"``.
    reason
        One of :data:`INVALID_REASONS`.
    """

    code = MASK_REGION_INVALID

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


def _refusal(name: str, reason: str, detail: str) -> RegionMaskError:
    """Return the ``MASK_REGION_INVALID`` error: the only place its message is written."""
    return RegionMaskError(
        f"{MASK_REGION_INVALID} ({reason}): the region {name!r} {detail}. A region is a 2-D "
        "integer or boolean array of the plane's shape (Y, X); its non-zero pixels are inside.",
        reason=reason,
    )


def _plane_shape(shape: object) -> tuple[int, int]:
    """Return the plane's ``(Y, X)`` as Python ints, or refuse a malformed argument.

    A malformed ``shape`` is the caller's programming error, not a refused region, so it is a
    plain ``ValueError`` without a code.
    """
    if not isinstance(shape, tuple | list) or len(shape) != 2:
        raise ValueError(f"shape must be the plane's (Y, X), two integers; got {shape!r}.")
    plane = []
    for size in shape:
        if isinstance(size, bool) or not isinstance(size, int | np.integer) or size < 0:
            raise ValueError(f"shape must be the plane's (Y, X), two integers; got {shape!r}.")
        plane.append(int(size))
    return plane[0], plane[1]


def check_region(mask: object, shape: tuple[int, int], *, name: str) -> np.ndarray:
    """Return the foreground ``mask != 0`` of a region after checking it.

    Parameters
    ----------
    mask
        The region: a NumPy array, 2-D, of the plane's shape, of a boolean or integer
        (signed or unsigned) dtype. Any representation is accepted (instances, binary,
        parent labels); negative values are inside the region. Never modified.
    shape
        The plane's ``(Y, X)``.
    name
        The region's name, used in the message.

    Returns
    -------
    numpy.ndarray
        A new boolean ``(Y, X)`` array, ``True`` where ``mask != 0``. It may hold no ``True``
        (an empty region is not refused).

    Raises
    ------
    RegionMaskError
        ``MASK_REGION_INVALID`` with the reason ``masked_array`` (a ``numpy.ma.MaskedArray``,
        whose mask would be ambiguous), ``not_an_array`` (not a NumPy array), ``not_2d`` (a
        ``(1, Y, X)`` stack included: never squeezed), ``shape`` (never resized or cropped) or
        ``dtype`` (float, complex, object or any other non-integer, non-boolean dtype: a
        probability map is thresholded by the user first), tested in that order.
    ValueError
        ``shape`` is not two non-negative integers (a programming error, no code).
    """
    plane = _plane_shape(shape)
    if isinstance(mask, np.ma.MaskedArray):
        raise _refusal(name, "masked_array", "is a masked array")
    if not isinstance(mask, np.ndarray):
        raise _refusal(name, "not_an_array", f"is not a NumPy array (got {type(mask).__name__})")
    if mask.ndim != 2:
        raise _refusal(name, "not_2d", f"has {mask.ndim} dimensions")
    if mask.shape != plane:
        raise _refusal(name, "shape", f"has shape {mask.shape}, not the plane's {plane}")
    if mask.dtype != np.bool_ and mask.dtype.kind not in "iu":
        raise _refusal(name, "dtype", f"has pixel type {mask.dtype}")
    # np.asarray: an ndarray subclass (a memory map) gives a plain boolean array.
    return np.asarray(mask != 0, dtype=np.bool_)


@dataclass(frozen=True)
class RegionEstimate:
    """An estimate over a region's pixels: its value, or why it is undefined.

    Exactly one of ``value`` and ``reason`` is set. NaN never appears in an estimate.

    Attributes
    ----------
    value
        The estimate, a non-empty tuple of finite Python floats (three for a white or a stain
        vector); ``None`` when undefined.
    reason
        ``None`` when defined; else one of :data:`REGION_REASONS` (``"empty_region"``).

    Raises
    ------
    ValueError
        Both or neither of ``value`` and ``reason`` set, a value that is not a non-empty tuple
        of finite Python floats, or an unknown reason.
    """

    value: tuple[float, ...] | None
    reason: str | None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.reason is None):
            raise ValueError(
                "A RegionEstimate has exactly one of value and reason; got "
                f"value={self.value!r}, reason={self.reason!r}."
            )
        if self.reason is not None and self.reason not in REGION_REASONS:
            raise ValueError(
                f"Unknown RegionEstimate reason {self.reason!r}; known: {list(REGION_REASONS)}."
            )
        if self.value is not None and not (
            type(self.value) is tuple
            and self.value
            and all(type(v) is float and math.isfinite(v) for v in self.value)
        ):
            raise ValueError(
                "A RegionEstimate value is a non-empty tuple of finite Python floats; got "
                f"{self.value!r}."
            )
