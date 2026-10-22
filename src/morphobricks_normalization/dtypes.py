"""Supported pixel types and their bin-count rules."""

from __future__ import annotations

import numpy as np

MAX_BIN_COUNT = 65_536
DEFAULT_FLOAT_BIN_COUNT = 10_000
MIN_FLOAT_BIN_COUNT = 2

INTEGER_DTYPES = frozenset(
    np.dtype(name)
    for name in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")
)
FLOAT_DTYPES = frozenset(np.dtype(name) for name in ("float32", "float64"))


class UnsupportedDtypeError(ValueError):
    """Raised for pixel types this brick does not process (bool, complex, float16, ...)."""


def check_supported(dtype: np.dtype | type | str) -> np.dtype:
    """Return ``dtype`` as a NumPy dtype, or raise if it is not supported.

    Parameters
    ----------
    dtype
        Any value accepted by :func:`numpy.dtype`.

    Returns
    -------
    numpy.dtype
        The normalised dtype, in native byte order.

    Raises
    ------
    UnsupportedDtypeError
        If the dtype is not a scalar integer type or float32/float64.
    """
    dt = np.dtype(dtype).newbyteorder("=")
    if dt in INTEGER_DTYPES or dt in FLOAT_DTYPES:
        return dt
    raise UnsupportedDtypeError(
        f"Pixel type {dt.name!r} is not supported. Supported types are signed and unsigned "
        "8-, 16-, 32- and 64-bit integers, float32 and float64."
    )


def is_integer(dtype: np.dtype) -> bool:
    """Return True for the supported integer dtypes."""
    return np.dtype(dtype) in INTEGER_DTYPES


def bit_depth(dtype: np.dtype) -> int:
    """Return the storage bit depth of a supported dtype (8, 16, 32 or 64)."""
    return check_supported(dtype).itemsize * 8


def default_bin_count(dtype: np.dtype) -> int:
    """Return the default number of bins for a dtype.

    Integer types get one bin per possible value, capped at 65,536, so 8- and
    16-bit histograms are exact by default. Float types get 10,000 bins.
    """
    dt = check_supported(dtype)
    if is_integer(dt):
        return min(MAX_BIN_COUNT, 2 ** bit_depth(dt))
    return DEFAULT_FLOAT_BIN_COUNT


def validate_bin_count(dtype: np.dtype, bin_count: int) -> int:
    """Check that ``bin_count`` is allowed for ``dtype`` and return it.

    Integer types allow 1 to min(65,536, number of representable values) bins.
    Float types allow 2 to 65,536 bins.

    Raises
    ------
    ValueError
        If the bin count is outside the allowed range.
    """
    dt = check_supported(dtype)
    if isinstance(bin_count, bool) or not isinstance(bin_count, (int, np.integer)):
        raise ValueError(f"The bin count must be a whole number, got {bin_count!r}.")
    if is_integer(dt):
        low, high = 1, min(MAX_BIN_COUNT, 2 ** bit_depth(dt))
    else:
        low, high = MIN_FLOAT_BIN_COUNT, MAX_BIN_COUNT
    if not low <= bin_count <= high:
        raise ValueError(
            f"The bin count for {dt.name} images must be between {low:,} and {high:,}; "
            f"got {bin_count:,}."
        )
    return int(bin_count)
