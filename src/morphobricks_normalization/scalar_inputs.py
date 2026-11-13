"""Input contracts of the image-to-scalar presets (contract §3.4, plan N67, N68).

:func:`rgb_yxc` checks an 8-bit sRGB image with declared axes and returns a ``(Y, X, 3)``
copy; :func:`plane_float32` checks a single ``"YX"`` plane for the ``passthrough`` presets and
returns a float32 copy. Nothing is reordered, rescaled or repaired silently, and the caller's
array is never returned or modified.
"""

from __future__ import annotations

from typing import Any

import numpy as np

AXES = ("YXC", "CYX")
PLANE_AXES = "YX"
SAMPLE_COUNT = 3


def rgb_yxc(pixels: np.ndarray, axes: str) -> np.ndarray:
    """Return a new C-contiguous ``(Y, X, 3)`` array after checking the RGB input contract.

    Always a copy, never a view of ``pixels``, so no later step can modify the caller's array.
    """
    if not isinstance(pixels, np.ndarray):
        raise TypeError(f"pixels must be a NumPy array; got {type(pixels).__name__}.")
    if not isinstance(axes, str) or axes not in AXES:
        raise ValueError(f"axes must be exactly 'YXC' or 'CYX', never guessed; got {axes!r}.")
    if pixels.ndim != 3:
        raise ValueError(f"Expected a 3-D array with axes {axes}; got {pixels.ndim}-D.")
    colour_axis = axes.index("C")
    if pixels.shape[colour_axis] != SAMPLE_COUNT:
        extra = (
            " An alpha channel is refused, not dropped." if pixels.shape[colour_axis] == 4 else ""
        )
        raise ValueError(
            f"Expected exactly 3 samples (R, G, B) on the colour axis of {axes}; "
            f"got {pixels.shape[colour_axis]}.{extra}"
        )
    if pixels.dtype != np.uint8:
        raise TypeError(
            f"Only 8-bit sRGB (uint8) is supported; got {pixels.dtype}. "
            "Nothing is rescaled or converted silently."
        )
    return np.array(np.moveaxis(pixels, colour_axis, 2), order="C", copy=True)


def plane_float32(pixels: np.ndarray, axes: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a float32 copy of a finite 2-D ``"YX"`` plane and whether the cast was exact.

    Raises
    ------
    ValueError
        axes other than ``"YX"``, not 2-D, NaN or infinity, or values beyond float32.
    TypeError
        Not a NumPy array, or a dtype that is not integer or float (bool, complex).
    """
    if not isinstance(pixels, np.ndarray):
        raise TypeError(f"pixels must be a NumPy array; got {type(pixels).__name__}.")
    if axes != PLANE_AXES:
        raise ValueError(f"axes must be exactly 'YX' for a single plane; got {axes!r}.")
    if pixels.ndim != 2:
        raise ValueError(f"Expected a 2-D array with axes YX; got {pixels.ndim}-D.")
    if pixels.dtype.kind not in "iuf":
        raise TypeError(
            f"Expected a real numeric dtype (integer or float); got {pixels.dtype}. "
            "bool and complex planes are refused."
        )
    if pixels.dtype.kind == "f" and not np.isfinite(pixels).all():
        raise ValueError("The plane has NaN or infinite values; they are refused, not repaired.")
    with np.errstate(over="ignore"):  # an overflow becomes infinity, refused just below
        image = pixels.astype(np.float32)  # always a new array
    if not np.isfinite(image).all():
        raise ValueError(
            "The plane has values beyond the float32 range (about 3.4e38); they are refused."
        )
    return image, {"float32_exact": float32_exact(image, pixels)}


def float32_exact(image: np.ndarray, pixels: np.ndarray) -> bool:
    """Return whether the float32 ``image`` holds exactly the values of ``pixels``.

    Every float32 is exact in float64, so float64 compares floats and integers up to 32 bits
    exactly. 64-bit integers above 2**53 are not exact in float64; they are compared as
    integers after checking that the float32 values fit the integer type.
    """
    back = image.astype(np.float64)
    if pixels.dtype.kind == "f" or pixels.dtype.itemsize < 8:
        return bool(np.array_equal(back, pixels.astype(np.float64)))
    info = np.iinfo(pixels.dtype)
    # float(info.max) rounds up to 2**63 or 2**64, the first value that does not fit.
    if (back < info.min).any() or (back >= float(info.max)).any():
        return False
    return bool(np.array_equal(back.astype(pixels.dtype), pixels))
