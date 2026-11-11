"""Exact centred moments of the pixels of a histogram tail (contract §1.4; schema version 5).

The pixels outside a robust range are stored by their count, mean and standard deviation,
not by their sum and sum of squares as schema version 4 did. The raw sum of squares lost
the spread of a tail of large values with a small spread by cancellation
(``sum_of_squares - sum * mean``), and it overflowed float64 for values beyond about
``1e154`` although the spread was finite (#91, #33).

Here every pixel value is turned into an exact integer multiple of a power of two (its
float64 significand and exponent, or the integer itself), so the sum and the sum of squares
are kept as exact rationals over any number of blocks, in any order. The mean is then the
float64 nearest the exact mean, and the standard deviation the float64 nearest the exact
population standard deviation about the exact mean. Both are finite for finite pixels (the
standard deviation is at most half the tail's range), and a plane gives the same bits read
whole or in blocks.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

_SIGNIFICAND_BITS = 53
# Extra bits of the integer square root before the one rounding to float64.
_SQRT_BITS = 2 * _SIGNIFICAND_BITS + 16


class TailMoments:
    """Running exact count, sum and sum of squares of one tail's pixels, block by block."""

    def __init__(self) -> None:
        self.count = 0
        self.total = Fraction(0)
        self.squares = Fraction(0)

    def add(self, values: np.ndarray) -> None:
        """Add the pixels ``values`` (finite; integers or floats) to the tail."""
        if values.size == 0:
            return
        total, squares = exact_sums(values)
        self.count += int(values.size)
        self.total += total
        self.squares += squares

    def moments(self) -> tuple[int, float, float]:
        """Return ``(count, mean, std)`` of the pixels added so far (zeros when none)."""
        return centred_moments(self.count, self.total, self.squares)


def exact_sums(values: np.ndarray) -> tuple[Fraction, Fraction]:
    """Return the exact sum and sum of squares of finite integer or float values."""
    if values.dtype.kind in "iu":
        whole = values.tolist()  # Python integers: exact for every 64-bit value
        return Fraction(sum(whole)), Fraction(sum(v * v for v in whole))
    significands, exponents = np.frexp(values.astype(np.float64, copy=False))
    # Each value is whole * 2**shift exactly: frexp's significand has at most 53 bits.
    whole = (significands * 2.0**_SIGNIFICAND_BITS).astype(np.int64)
    shifts = exponents.astype(np.int64) - _SIGNIFICAND_BITS
    order = np.argsort(shifts, kind="stable")
    unique, starts = np.unique(shifts[order], return_index=True)
    total, squares = 0, 0
    lowest = int(unique[0])
    for shift, group in zip(unique.tolist(), np.split(whole[order], starts[1:]), strict=True):
        group = group.tolist()
        offset = shift - lowest  # every group on the lowest power of two, as integers
        total += sum(group) << offset
        squares += sum(m * m for m in group) << (2 * offset)
    return _times_power_of_two(total, lowest), _times_power_of_two(squares, 2 * lowest)


def counted_sums(counts: np.ndarray, first_value: int) -> tuple[Fraction, Fraction]:
    """Return the exact sum and sum of squares of per-value ``counts`` from ``first_value``."""
    index = np.flatnonzero(counts)
    pairs = zip(counts[index].tolist(), (index + first_value).tolist(), strict=True)
    total, squares = 0, 0
    for count, value in pairs:
        total += count * value
        squares += count * value * value
    return Fraction(total), Fraction(squares)


def centred_moments(count: int, total: Fraction, squares: Fraction) -> tuple[int, float, float]:
    """Return ``(count, mean, std)`` of values with exact ``total`` and ``squares``.

    The mean and the population standard deviation are each rounded once, to the nearest
    float64, from their exact values; zeros for no values.
    """
    if count == 0:
        return 0, 0.0, 0.0
    mean = total / count
    variance = squares / count - mean * mean  # exact, so never negative
    return count, float(mean), sqrt_nearest(variance)


def sqrt_nearest(value: Fraction) -> float:
    """Return the float64 nearest the square root of a non-negative rational."""
    if value < 0:
        raise ValueError(f"A variance cannot be negative; got {value}.")
    if value == 0:
        return 0.0
    numerator, denominator = value.numerator, value.denominator
    # 2**scale * sqrt(value) has about _SQRT_BITS bits, far more than float64 keeps.
    scale = _SQRT_BITS - (numerator.bit_length() - denominator.bit_length()) // 2
    if scale >= 0:
        scaled, remainder = divmod(numerator << (2 * scale), denominator)
    else:
        scaled, remainder = divmod(numerator, denominator << (-2 * scale))
    root = math.isqrt(scaled)
    # A sticky last bit for the part below the integer root, so that rounding to float64
    # once more cannot land on a false tie.
    inexact = remainder != 0 or root * root != scaled
    return float(_times_power_of_two(2 * root + int(inexact), -(scale + 1)))


def _times_power_of_two(whole: int, exponent: int) -> Fraction:
    """Return ``whole * 2**exponent`` as an exact rational."""
    if exponent >= 0:
        return Fraction(whole << exponent)
    return Fraction(whole, 1 << -exponent)
