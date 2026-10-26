import numpy as np
import pytest

from morphobricks_normalization import (
    SINGLE,
    NormalizationSettings,
    PlaneRecord,
    compute_plane_histogram,
    fit,
    group_key,
    group_name,
    group_records,
    scope_groups,
)
from simulation import cached_design, make_design_records, simulate_design

REPEAT = NormalizationSettings(scope="biological_repeat")
IMAGE = NormalizationSettings(scope="image")


def record(condition, repeat, technical=SINGLE, image="img.tif", c=0, t=0):
    path = f"{condition}/{repeat}/{technical}/{image}"
    histogram = compute_plane_histogram(np.arange(4, dtype=np.uint8))
    return PlaneRecord(path, condition, repeat, technical, histogram, c=c, t=t)


def test_repeats_with_the_same_name_form_one_group_across_conditions():
    control, treated = record("control", "2024_05_01"), record("treated", "2024_05_01")
    assert group_key(control, REPEAT) == group_key(treated, REPEAT) == ("2024_05_01",)
    assert group_key(record("control", "2024_05_02"), REPEAT) == ("2024_05_02",)


def test_biological_repeat_pools_its_planes_across_conditions_and_technical_repeats():
    _, records = cached_design(seed=1)
    groups, issues = scope_groups(records, REPEAT)
    assert issues == []
    assert list(groups) == [("2024_05_01",), ("2024_05_02",)]
    for (label,), members in groups.items():
        expected = {r.plane_id for r in records if r.biological_repeat == label}
        assert {r.plane_id for r in members} == expected
        assert {r.condition for r in members} == {"control", "treated"}
        assert {r.technical_repeat for r in members} == {"tech_rep1", "tech_rep2"}
        assert {r.c for r in members} == {0, 1}
    assert sum(len(members) for members in groups.values()) == len(records)


def test_pairing_differently_named_repeats_changes_only_the_grouping():
    # The same pixels, once with matching names and once acquired on other days
    # in the treated condition and paired in the experiment design.
    _, matching = cached_design(seed=2)
    renamed_labels = {"2024_05_01": "2024_06_11", "2024_05_02": "2024_06_12"}
    renamed = [
        PlaneRecord(
            r.source_relative_path,
            r.condition,
            renamed_labels[r.biological_repeat]
            if r.condition == "treated"
            else r.biological_repeat,
            r.technical_repeat,
            r.histogram,
            c=r.c,
            t=r.t,
        )
        for r in matching
    ]
    pairing = {("control", "2024_05_01"): "A", ("control", "2024_05_02"): "B"}
    pairing |= {("treated", "2024_06_11"): "A", ("treated", "2024_06_12"): "B"}

    unpaired, issues = scope_groups(renamed, REPEAT)
    assert len(unpaired) == 4 and [issue.code for issue in issues] == ["repeat_missing_conditions"]

    paired, issues = scope_groups(renamed, REPEAT, pairing)
    reference, _ = scope_groups(matching, REPEAT)
    assert issues == [] and list(paired) == [("A",), ("B",)]
    for (label, members), (_, expected) in zip(paired.items(), reference.items(), strict=True):
        assert [r.plane_id for r in members] == [r.plane_id for r in expected]
        assert fit(members, REPEAT, group=label).offset == fit(expected, REPEAT).offset


def test_every_pair_must_be_in_the_pairing():
    records = [record("control", "r1"), record("treated", "r2"), record("treated", "r3")]
    with pytest.raises(ValueError) as error:
        scope_groups(records, REPEAT, {("control", "r1"): "A"})
    message = str(error.value)
    assert "treated/r2, treated/r3" in message and "experiment design" in message
    with pytest.raises(ValueError, match="treated/r2"):
        group_key(records[1], REPEAT, {("control", "r1"): "A"})


def test_a_repeat_missing_from_some_conditions_is_used_with_a_warning():
    design = {"control": ("r1", "r2"), "treated": ("r1",), "drug": ("r1",)}
    records = make_design_records(simulate_design(seed=3, design=design, n_channels=1))
    groups, issues = scope_groups(records, REPEAT)
    assert list(groups) == [("r1",), ("r2",)]
    assert {r.condition for r in groups[("r2",)]} == {"control"}
    (issue,) = issues
    assert (issue.severity, issue.code, issue.items) == (
        "warning",
        "repeat_missing_conditions",
        ("r2",),
    )
    assert "r2: no images from drug, treated" in issue.message
    assert "experiment design" in issue.action


def test_single_repeats_group_together_or_with_the_repeat_they_are_paired_with():
    records = [record("control", SINGLE), record("treated", SINGLE)]
    groups, issues = scope_groups(records, REPEAT)
    assert list(groups) == [(SINGLE,)] and issues == []
    mixed = [record("control", SINGLE), record("treated", "2024_05_01")]
    pairing = {("control", SINGLE): "A", ("treated", "2024_05_01"): "A"}
    groups, issues = scope_groups(mixed, REPEAT, pairing)
    assert list(groups) == [("A",)] and issues == []


def test_image_scope_and_time_points():
    plane = record("control", "r1", image="a.tif", t=3)
    assert group_key(plane, IMAGE) == ("control/r1/single/a.tif",)
    per_t = NormalizationSettings(scope="image", per_time_point=True)
    assert group_key(plane, per_t) == ("control/r1/single/a.tif", 3)
    assert group_name(group_key(plane, per_t)) == "control/r1/single/a.tif::t=3"
    assert group_name(group_key(plane, IMAGE)) == "control/r1/single/a.tif"
    repeat_per_t = NormalizationSettings(scope="biological_repeat", per_time_point=True)
    assert group_name(group_key(plane, repeat_per_t, {("control", "r1"): "A"})) == "A::t=3"
    assert group_name(group_key(plane, repeat_per_t)) == "r1::t=3"


def test_per_time_point_groups_split_every_scope_by_time():
    _, records = cached_design(seed=4, n_channels=1, n_times=3)
    for scope in ("biological_repeat", "image"):
        settings = NormalizationSettings(scope=scope, per_time_point=True)
        groups, _ = scope_groups(records, settings)
        pooled, _ = scope_groups(records, NormalizationSettings(scope=scope))
        assert len(groups) == 3 * len(pooled)
        for key, members in groups.items():
            assert {r.t for r in members} == {key[-1]}


def test_image_scope_pools_all_planes_of_one_image():
    _, records = cached_design(seed=5, n_times=2)
    groups, issues = scope_groups(records, IMAGE)
    assert issues == []
    for (path,), members in groups.items():
        assert {r.source_relative_path for r in members} == {path}
        assert len(members) == 2 * 2  # channels x time points


def test_patch_scope_has_no_row_groups():
    with pytest.raises(ValueError, match="patch's own pixels"):
        group_key(record("control", "r1"), NormalizationSettings(scope="patch"))
    with pytest.raises(ValueError, match="patch's own pixels"):
        scope_groups([record("control", "r1")], NormalizationSettings(scope="patch"))


def test_existing_label_groups_keep_their_meaning():
    plane = record("control", "r1", technical="w1")
    assert plane.group_key("biological_repeat") == ("control", "r1")
    assert plane.group_key("image") == ("control", "r1", "w1", "control/r1/w1/img.tif")
    records = [plane, record("treated", "r1")]
    assert list(group_records(records, "biological_repeat")) == [
        ("control", "r1"),
        ("treated", "r1"),
    ]
