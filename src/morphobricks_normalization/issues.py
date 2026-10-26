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
    """

    severity: str
    code: str
    message: str
    items: tuple[str, ...] = ()
    action: str = ""
