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
from .records import PlaneRecord
from .summaries import IncompatibleSummariesError

# Added to the scale by csbdeep.utils.normalize_mi_ma (its ``eps``).
EPSILON = 1e-20
RGB_CHANNEL_COUNT = 3
# The pixel types of RGB images: 8- and 16-bit integers, every value exact in float32 (#23).
RGB_DTYPES = frozenset(np.dtype(name) for name in ("uint8", "uint16", "int8", "int16"))
# The fixed channel IDs of an RGB imaging unit, in source order (experiment-v1 §4.4b).
RGB_CHANNELS = ("R", "G", "B")


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
    x = np.asarray(pixels).astype(np.float32)  # always a new array, never the caller's
    # The same float32 operations in the same order, written into that one array (issue #53).
    np.subtract(x, mi, out=x)
    np.divide(x, ma - mi + np.float32(EPSILON), out=x)
    return x


def check_rgb(channel_count: int, dtypes: Iterable[np.dtype]) -> None:
    """Refuse images that are not three channels of an 8- or 16-bit integer type (contract §5).

    32- and 64-bit integers are refused (issue #23): CSBDeep's float32 arithmetic, which
    :func:`apply_rgb` repeats, cannot hold every such value (above ``2**24``), and the
    contract's message asks for 8- or 16-bit colour channels.

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
    supported = all(np.dtype(name) in RGB_DTYPES for name in dtypes)
    if channel_count != RGB_CHANNEL_COUNT or not supported:
        raise ValueError(
            "RGB normalization needs images with exactly three 8- or 16-bit colour channels, "
            f"but these images have {channel_count} channels of type {' and '.join(dtypes)}. "
            "Check the axis mapping (the samples axis must be the channel axis), or choose "
            "image type 'fluorescence'."
        )


def check_rgb_channels(channels: Iterable[str], where: str) -> None:
    """Refuse RGB channels that are not ``R``, ``G`` and ``B`` in this order (§4.4b).

    Parameters
    ----------
    channels
        The channel IDs, in source order.
    where
        Starts the message, e.g. ``"Plane a.tif::z=0,t=0"``.

    Raises
    ------
    ValueError
        Naming the IDs found.
    """
    channels = tuple(channels)
    if channels != RGB_CHANNELS:
        raise ValueError(
            f"{where}: RGB normalization needs the channels R, G and B of an RGB imaging unit, "
            f"in this order (experiment-v1 §4.4b), but it has {', '.join(map(str, channels))}. "
            "Declare the unit as image type 'rgb' in the experiment description, or choose "
            "image type 'fluorescence'."
        )


def check_rgb_pixels(pixels: np.ndarray, channel_axis: int | None) -> None:
    """Refuse a pixel array that is not three 8- or 16-bit channels, before any work (#52, #23).

    Parameters
    ----------
    pixels
        The image or patch.
    channel_axis
        Axis of the channels (negative counts from the end); None for one channel.

    Raises
    ------
    ValueError
        If the axis is outside the array, or as :func:`check_rgb`.
    """
    if channel_axis is None:
        count = 1
    elif -pixels.ndim <= channel_axis < pixels.ndim:
        count = pixels.shape[channel_axis]
    else:
        raise ValueError(
            f"channel_axis {channel_axis} is outside the pixel array's {pixels.ndim} axes."
        )
    check_rgb(count, [pixels.dtype])


def check_rgb_records(records: list[PlaneRecord], group: str | None) -> None:
    """Refuse rows that are not three channels of one 8- or 16-bit type (contract §5, #51, #23).

    RGB pools all images of a group into one distribution, so bit depths are never mixed
    or converted; the message names the images of each pixel type.
    """
    check_rgb(records[0].channel_count, [h.dtype for r in records for h in r.histograms])
    for record in records:
        check_rgb_channels(record.channels, f"Plane {record.plane_id}")
    by_dtype: dict[str, list[str]] = {}
    for record in records:
        files = by_dtype.setdefault(record.histograms[0].dtype.name, [])
        if record.source_relative_path not in files:
            files.append(record.source_relative_path)
    if len(by_dtype) > 1:
        subject = f"The images of group '{group}'" if group else "The images of this group"
        parts = "; ".join(f"{name}: {', '.join(files)}" for name, files in sorted(by_dtype.items()))
        raise IncompatibleSummariesError(
            f"{subject} have different pixel types ({parts}). RGB normalization pools the "
            "colour channels of all images into one shared offset and scale, and pixel types "
            "are not converted, so it needs one pixel type. Save the images with one pixel "
            "type, or put them in separate groups, then fit again."
        )


def check_rgb_selection(wanted: list[str], channels: tuple[str, ...]) -> None:
    """Refuse a strict subset of R, G and B: RGB parameters keep the channels together (#50).

    A subset would serialize as a single channel that ``from_columns`` reads back as
    something else, and a lone colour channel is not a valid RGB patch (contract §5).
    """
    if wanted != list(channels):
        raise ValueError(
            f"RGB parameters belong to the three colour channels together, so channels {wanted} "
            f"alone cannot be selected. Select all of channels {list(channels)}, or choose "
            "image type 'fluorescence' to normalize channels separately."
        )
