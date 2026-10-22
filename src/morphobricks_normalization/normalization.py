"""Normalization methods for later bricks (for example patch extraction).

Nothing here is applied by the preprocessing brick itself. Each method turns
pooled group statistics into an offset and a scale; applying a normalization
is always ``(pixels - offset) / scale``.

==================  ====================  ===============================
Method              offset                scale
==================  ====================  ===============================
min-max             exact minimum         maximum - minimum
percentile range    low percentile        high percentile - low percentile
z-score             mean                  standard deviation
robust z-score      median                MAD x consistency constant
==================  ====================  ===============================

Parameters come from a :class:`~morphobricks_normalization.pooling.PooledHistogram`,
so a scope (an image, a technical repeat, a condition, ...) is chosen simply
by choosing which planes to pool.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pooling import PooledHistogram

# Makes the MAD a consistent estimator of the standard deviation for normal data.
NORMAL_CONSISTENCY = 1.482602218505602


@dataclass(frozen=True)
class NormalizationParameters:
    """How to normalize pixel values: ``(pixels - offset) / scale``."""

    method: str
    offset: float
    scale: float

    def apply(
        self,
        pixels: np.ndarray,
        *,
        clip: tuple[float, float] | None = None,
        dtype: np.dtype | type = np.float32,
    ) -> np.ndarray:
        """Return normalized pixel values as a new array.

        Parameters
        ----------
        pixels
            Values to normalize; the input is never modified.
        clip
            Optional ``(low, high)`` limits applied after normalizing, for
            example ``(0, 1)`` for min-max scaling.
        dtype
            Floating-point type of the result.
        """
        result = (np.asarray(pixels, dtype=np.float64) - self.offset) / self.scale
        if clip is not None:
            np.clip(result, clip[0], clip[1], out=result)
        return result.astype(dtype, copy=False)


def min_max(pooled: PooledHistogram) -> NormalizationParameters:
    """Scale the exact minimum to 0 and the exact maximum to 1."""
    return _parameters("min-max", pooled.minimum, pooled.maximum - pooled.minimum)


def percentile_range(
    pooled: PooledHistogram, low: float = 1.0, high: float = 99.8
) -> NormalizationParameters:
    """Scale the ``low`` percentile to 0 and the ``high`` percentile to 1."""
    if not 0 <= low < high <= 100:
        raise ValueError(f"Expected 0 <= low < high <= 100; got low={low}, high={high}.")
    lower = pooled.percentile(low)
    return _parameters(f"percentile {low:g}-{high:g}", lower, pooled.percentile(high) - lower)


def z_score(pooled: PooledHistogram) -> NormalizationParameters:
    """Subtract the mean and divide by the standard deviation."""
    return _parameters("z-score", pooled.mean, pooled.std)


def robust_z_score(
    pooled: PooledHistogram, consistency: float = NORMAL_CONSISTENCY
) -> NormalizationParameters:
    """Subtract the median and divide by the scaled median absolute deviation."""
    return _parameters("robust z-score", pooled.median, pooled.mad * consistency)


def _parameters(method: str, offset: float, scale: float) -> NormalizationParameters:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"The {method} normalization cannot be used for this group: its intensity spread "
            "is zero (all pixels have the same value). Choose a larger group or another method."
        )
    return NormalizationParameters(method, float(offset), float(scale))
