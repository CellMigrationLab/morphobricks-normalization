"""The one place a float is turned into written text: signed zero is written as ``0.0`` (#105).

A float plane holding both ``-0.0`` and ``+0.0`` gets a ``pixel_min`` or ``pixel_max`` of
either zero, depending on NumPy's ``min``/``max``, the element order and the platform. So that
the same pixels write the same bytes everywhere, every float the package writes (the cells of
``plane_histograms.csv``, the parameter columns of contract §8.1, the float cells of the image
summary rows of contract §7.1 and the percentiles of a method name) goes through
:func:`canonical_zero`: ``-0.0`` becomes ``0.0``. Every other value, NaN and the infinities
included, is unchanged, and no computed value changes: only what is written does.
"""

from __future__ import annotations


def canonical_zero(value: float) -> float:
    """Return ``value`` as a Python float, with ``-0.0`` replaced by ``0.0``.

    ``x + 0.0`` is ``x`` for every float but ``-0.0``, whose sum with ``0.0`` is ``0.0``
    under round-to-nearest (IEEE 754), so no other value can change.
    """
    return float(value) + 0.0


def float_text(value: float) -> str:
    """Return the written text of a float: its shortest round-trip ``repr``, ``0.0`` for ``-0.0``.

    The text reads back as the same float (``float(float_text(x)) == x``), the sign of zero
    aside.
    """
    return repr(canonical_zero(value))
