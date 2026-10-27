"""Groups of ``plane_histograms.csv`` rows at each scope (contract §4).

``biological_repeat``
    One repeat group: one biological repeat in every condition, all technical
    repeats. Key ``(repeat group label,)``.
``image``
    All Z and T planes of one image. Key ``(source_relative_path,)``.
``patch``
    The patch's own pixels, not rows of the file; the calling brick names the
    group ``<source_relative_path>::<patch id>``.

With ``per_time_point`` the time point is added to the key, so each time point
is a group of its own. The repeat group of a row is its ``biological_repeat``
label, so repeats with the same name match across conditions, unless the
calling brick passes ``repeat_groups``, a mapping from ``(condition,
biological_repeat)`` to the label of the repeat group the user paired it with.

These keys are separate from :meth:`PlaneRecord.group_key` and
:func:`group_records`, whose levels keep their meaning (contract §1).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .issues import REPEAT_MISSING_CONDITIONS, WARNING, Issue
from .records import PlaneRecord
from .settings import NormalizationSettings

RepeatGroups = Mapping[tuple[str, str], str]


def group_key(
    record: PlaneRecord,
    settings: NormalizationSettings,
    repeat_groups: RepeatGroups | None = None,
) -> tuple:
    """Key of the group ``record`` belongs to at the settings' scope.

    Parameters
    ----------
    record
        One ``plane_histograms.csv`` row.
    settings
        The run's settings: scope and ``per_time_point``.
    repeat_groups
        At the biological-repeat scope, the repeat group label of every
        ``(condition, biological_repeat)``; None matches repeats by name.

    Returns
    -------
    tuple
        ``(label,)`` or, with ``per_time_point``, ``(label, t)``; ``label`` is
        the repeat group label or the image's ``source_relative_path``.

    Raises
    ------
    ValueError
        At the patch scope, whose groups are the calling brick's patches, or
        when ``repeat_groups`` has no label for the record's condition and
        biological repeat.
    """
    if settings.scope == "patch":
        raise ValueError(
            "At the patch scope every patch is its own group, named "
            "'<source_relative_path>::<patch id>' by the calling brick, so the rows of "
            "plane_histograms.csv have no patch group. Fit patch parameters from the "
            "patch's own pixels."
        )
    if settings.scope == "biological_repeat":
        label = _repeat_group(record, repeat_groups)
    else:
        label = record.source_relative_path
    return (label, record.t) if settings.per_time_point else (label,)


def group_name(key: tuple) -> str:
    """Return the ``normalization_group`` text of a group key (contract §4.1, §4.4)."""
    return key[0] if len(key) == 1 else f"{key[0]}::t={key[1]}"


def scope_groups(
    records: Iterable[PlaneRecord],
    settings: NormalizationSettings,
    repeat_groups: RepeatGroups | None = None,
) -> tuple[dict[tuple, list[PlaneRecord]], list[Issue]]:
    """Group the rows of ``plane_histograms.csv`` at the settings' scope.

    Every row of a group is returned, including planes the calling brick does
    not normalize (contract §4.1).

    Parameters
    ----------
    records
        The rows of the statistics file.
    settings
        The run's settings.
    repeat_groups
        As for :func:`group_key`.

    Returns
    -------
    groups : dict
        The rows of each group, by group key in sorted order.
    issues : list of Issue
        ``repeat_missing_conditions`` when a repeat group has no rows from
        some condition of the statistics; the group is still used.

    Raises
    ------
    ValueError
        As :func:`group_key`; all missing ``(condition, biological_repeat)``
        pairs are named in one message.
    """
    records = list(records)
    if settings.scope == "biological_repeat" and repeat_groups is not None:
        missing = {(r.condition, r.biological_repeat) for r in records} - set(repeat_groups)
        if missing:
            raise _missing_pairs_error(missing)
    groups: dict[tuple, list[PlaneRecord]] = {}
    for record in records:
        groups.setdefault(group_key(record, settings, repeat_groups), []).append(record)
    issues = []
    if settings.scope == "biological_repeat":
        issues.extend(_incomplete_repeat_groups(records, repeat_groups))
    return dict(sorted(groups.items())), issues


def _repeat_group(record: PlaneRecord, repeat_groups: RepeatGroups | None) -> str:
    if repeat_groups is None:
        return record.biological_repeat
    pair = (record.condition, record.biological_repeat)
    if pair not in repeat_groups:
        raise _missing_pairs_error({pair})
    return repeat_groups[pair]


def _missing_pairs_error(pairs: set[tuple[str, str]]) -> ValueError:
    names = ", ".join(f"{condition}/{repeat}" for condition, repeat in sorted(pairs))
    return ValueError(
        f"The experiment design does not assign these biological repeats to a repeat group: "
        f"{names}. Update the experiment design in the preprocessing brick so that every "
        "biological repeat of every condition belongs to a repeat group, then run again."
    )


def _incomplete_repeat_groups(
    records: list[PlaneRecord], repeat_groups: RepeatGroups | None
) -> list[Issue]:
    conditions = {record.condition for record in records}
    present: dict[str, set[str]] = {}
    for record in records:
        present.setdefault(_repeat_group(record, repeat_groups), set()).add(record.condition)
    missing = {
        label: sorted(conditions - found)
        for label, found in sorted(present.items())
        if conditions - found
    }
    if not missing:
        return []
    lines = "\n".join(
        f"  - {label}: no images from {', '.join(absent)}" for label, absent in missing.items()
    )
    return [
        Issue(
            WARNING,
            REPEAT_MISSING_CONDITIONS,
            f"{len(missing)} repeat group(s) do not have images from every condition; each is "
            f"normalized with the conditions it has:\n{lines}",
            items=tuple(missing),
            action="If a biological repeat was acquired on different days for different "
            "conditions, pair the repeats in the preprocessing brick's experiment design; "
            "otherwise no action is needed.",
        )
    ]
