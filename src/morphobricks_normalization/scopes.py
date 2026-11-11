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
from .records import PlaneRecord, duplicate_planes
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
        some condition of the statistics; the group is still used. With
        ``per_time_point`` each time point's group is checked (``<label>::t=<t>``).

    Raises
    ------
    ValueError
        As :func:`group_key`; all missing ``(condition, biological_repeat)``
        pairs are named in one message. Also when two rows share a
        ``plane_id``, or ``repeat_groups`` gives a present pair a label that is
        not text or is empty, or puts two biological repeats of one condition in
        the same repeat group (contract §4.1: one repeat per condition and group).
    """
    records = list(records)
    duplicate_planes(records, "The rows of plane_histograms.csv")
    if settings.scope == "biological_repeat" and repeat_groups is not None:
        pairs = {(r.condition, r.biological_repeat) for r in records}
        missing = pairs - set(repeat_groups)
        if missing:
            raise _missing_pairs_error(missing)
        _check_pairing(pairs, repeat_groups)
    groups: dict[tuple, list[PlaneRecord]] = {}
    for record in records:
        groups.setdefault(group_key(record, settings, repeat_groups), []).append(record)
    issues = []
    if settings.scope == "biological_repeat":
        issues.extend(_incomplete_repeat_groups(groups))
    return dict(sorted(groups.items())), issues


def _repeat_group(record: PlaneRecord, repeat_groups: RepeatGroups | None) -> str:
    if repeat_groups is None:
        return record.biological_repeat
    pair = (record.condition, record.biological_repeat)
    if pair not in repeat_groups:
        raise _missing_pairs_error({pair})
    return repeat_groups[pair]


def _check_pairing(pairs: set[tuple[str, str]], repeat_groups: RepeatGroups) -> None:
    """Refuse labels that are not text, and two repeats of one condition in one repeat group."""
    unlabelled = sorted(
        p for p in pairs if not isinstance(repeat_groups[p], str) or not repeat_groups[p]
    )
    if unlabelled:
        names = ", ".join(f"{c}/{r} ({repeat_groups[(c, r)]!r})" for c, r in unlabelled)
        raise ValueError(
            f"The experiment design gives these biological repeats no repeat group name: {names}. "
            "Name the repeat group of every biological repeat in the preprocessing brick's "
            "experiment design, then run again."
        )
    members: dict[tuple[str, str], set[str]] = {}
    for condition, repeat in pairs:
        members.setdefault((repeat_groups[(condition, repeat)], condition), set()).add(repeat)
    conflicts = [
        f"{label}: {condition}/{' and '.join(sorted(repeats))}"
        for (label, condition), repeats in sorted(members.items())
        if len(repeats) > 1
    ]
    if conflicts:
        raise ValueError(
            "A repeat group holds one biological repeat of each condition, but the experiment "
            "design puts several repeats of one condition in the same group "
            f"({'; '.join(conflicts)}). Pair each biological repeat with one repeat of every "
            "other condition in the preprocessing brick's experiment design, then run again."
        )


def _missing_pairs_error(pairs: set[tuple[str, str]]) -> ValueError:
    names = ", ".join(f"{condition}/{repeat}" for condition, repeat in sorted(pairs))
    return ValueError(
        f"The experiment design does not assign these biological repeats to a repeat group: "
        f"{names}. Update the experiment design in the preprocessing brick so that every "
        "biological repeat of every condition belongs to a repeat group, then run again."
    )


def _incomplete_repeat_groups(groups: dict[tuple, list[PlaneRecord]]) -> list[Issue]:
    """Warn about every fitted group without rows from some condition of the statistics.

    The groups are the ones that are fitted, so with ``per_time_point`` each time point of
    a repeat group is checked on its own and named ``<label>::t=<t>`` (#41).
    """
    conditions = {record.condition for members in groups.values() for record in members}
    missing = {
        group_name(key): sorted(conditions - {record.condition for record in members})
        for key, members in sorted(groups.items())
        if conditions - {record.condition for record in members}
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
