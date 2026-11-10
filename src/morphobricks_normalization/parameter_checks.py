"""Checks of normalization parameters: offsets, scales, scope (contract §8).

Channel IDs are checked by :mod:`~morphobricks_normalization.channels`.

:class:`~morphobricks_normalization.normalization.NormalizationParameters` and
:class:`~morphobricks_normalization.fitting.Parameters` check their values when
they are made, so parameters fitted here, read from columns or built by a
calling brick follow the same rules. A value that breaks one is refused with a
``ValueError``; nothing is rounded, truncated or replaced.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


def check_offset_scale(offset: float, scale: float, where: str) -> None:
    """Refuse an offset that is not finite, or a scale that is not finite and above zero.

    Parameters
    ----------
    offset, scale
        The values of one channel.
    where
        Names the channel or method in the message, e.g. ``"Channel DNA"``.

    Raises
    ------
    ValueError
        Naming ``where``, the value and what to do.
    """
    if not math.isfinite(offset):
        raise ValueError(
            f"{where}: the normalization offset must be a finite number; got {offset!r}. Fit the "
            "parameters again, or correct the stored offset."
        )
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError(
            f"{where}: the normalization scale must be a finite number above 0; got {scale!r}. "
            "Fit the parameters again, or correct the stored scale."
        )


def check_scope(scope: str, expected: str, scopes: Iterable[str]) -> str:
    """Return a parameter row's scope if it is a scope and the one the settings use.

    Parameters fitted at one scope describe another sampling population than
    those of another scope (contract §4.1), so a row is never read with the
    settings of another scope (issue #37).

    Parameters
    ----------
    scope
        The row's ``normalization_scope``.
    expected
        The scope of the settings the row is read with.
    scopes
        The scopes that exist.

    Returns
    -------
    str
        ``scope``.

    Raises
    ------
    ValueError
        For a scope that does not exist, or that differs from ``expected``.
    """
    scopes = tuple(scopes)
    if scope not in scopes:
        raise ValueError(
            f"Column normalization_scope is {scope!r}, which is not a scope; the scopes are "
            f"{', '.join(scopes)}. Read the parameters from a row written with normalization."
        )
    if scope != expected:
        raise ValueError(
            f"Column normalization_scope is {scope!r}, but the settings use scope {expected!r}: "
            "parameters fitted at one scope describe another population of pixels. Read the "
            "parameters with the settings of the run that wrote them."
        )
    return scope
