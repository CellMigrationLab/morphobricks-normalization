"""StarDist's percentile normalization of RGB images (contract §3.3, N33, N39, N40, N48).

StarDist normalizes its inputs with ``csbdeep.utils.normalize(x, pmin, pmax,
axis=(0, 1, 2))``: the percentiles are taken over the R, G and B values
together, and every channel gets the same offset and scale::

    offset = P(low)
    scale  = P(high) - P(low) + 1e-20
    y      = (x - offset) / scale

``P`` is NumPy's default linear-interpolation percentile
(:meth:`~morphobricks_normalization.pooling.PooledHistogram.linear_percentile`),
so the parameters equal CSBDeep's for exact histograms. Because of the
``1e-20`` term a group without spread gives zeros instead of an error.

:func:`apply_rgb` repeats CSBDeep's float32 arithmetic step by step (N48), so
the normalized pixels are bit-identical to ``normalize``'s. Implemented here in
NumPy; CSBDeep is used only in the tests.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .pooling import PooledHistogram, WeightedHistogram

# Added to the scale by csbdeep.utils.normalize_mi_ma (its ``eps``).
EPSILON = 1e-20
RGB_CHANNEL_COUNT = 3


def rgb_percentile(
    pooled: PooledHistogram | WeightedHistogram, low: float = 1.0, high: float = 99.8
) -> tuple[float, float]:
    """Return ``(offset, scale)`` of StarDist's percentile normalization.

    Parameters
    ----------
    pooled
        The R, G and B histograms of the group pooled together.
    low, high
        Percentiles in percent, ``0 <= low < high <= 100``.

    Returns
    -------
    offset, scale : float
        ``P(low)`` and ``P(high) - P(low) + 1e-20``, the scale never zero.
    """
    if not 0 <= low < high <= 100:
        raise ValueError(f"Expected 0 <= low < high <= 100; got low={low}, high={high}.")
    lower = pooled.linear_percentile(low)
    upper = pooled.linear_percentile(high)
    return lower, upper - lower + EPSILON


def apply_rgb(
    pixels: np.ndarray, offset: np.ndarray | float, scale: np.ndarray | float
) -> np.ndarray:
    """Normalize with CSBDeep's float32 arithmetic and return a new float32 array.

    ``csbdeep.utils.normalize_mi_ma`` casts the pixels, ``mi``, ``ma`` and
    ``eps`` to float32 and computes ``(x - mi) / (ma - mi + eps)``. ``mi`` is
    the offset and ``ma`` is rebuilt as ``offset + scale - 1e-20``; both round
    to the same float32 values as CSBDeep's, so the result is bit-identical.
    ``offset`` and ``scale`` may be arrays that broadcast against ``pixels``.
    """
    offset = np.asarray(offset, dtype=np.float64)
    upper = offset + np.asarray(scale, dtype=np.float64) - EPSILON
    mi = offset.astype(np.float32)
    ma = upper.astype(np.float32)
    x = np.asarray(pixels).astype(np.float32)  # always a copy for integer pixels
    return ((x - mi) / (ma - mi + np.float32(EPSILON))).astype(np.float32, copy=False)


def check_rgb(channel_count: int, dtypes: Iterable[np.dtype]) -> None:
    """Refuse images that are not three channels of an integer type (contract §5, §9.1).

    Parameters
    ----------
    channel_count
        Number of channels of the images.
    dtypes
        Their pixel types.

    Raises
    ------
    ValueError
        With the contract's message, naming the channel count and pixel type.
    """
    dtypes = sorted({np.dtype(dtype).name for dtype in dtypes})
    integer = all(np.issubdtype(np.dtype(name), np.integer) for name in dtypes)
    if channel_count != RGB_CHANNEL_COUNT or not integer:
        raise ValueError(
            "RGB normalization needs images with exactly three 8- or 16-bit colour channels, "
            f"but these images have {channel_count} channels of type {' and '.join(dtypes)}. "
            "Check the axis mapping (the samples axis must be the channel axis), or choose "
            "image type 'fluorescence'."
        )
