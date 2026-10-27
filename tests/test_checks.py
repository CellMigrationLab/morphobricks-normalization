"""Intensity checks (contract §7.2, N38, N51-N53)."""

from dataclasses import replace

import numpy as np
import pytest

from morphobricks_normalization import (
    SINGLE,
    ImageSummary,
    IntensityCheckSettings,
    Issue,
    PlaneRecord,
    compute_plane_histogram,
    image_summaries,
    intensity_checks,
)
from morphobricks_normalization.normalization import NORMAL_CONSISTENCY

# Every simulated image has the same intensity ramp around its level, so its
# median and high percentile move with the level; the noise keeps the images
# from being identical.
SHAPE = (32, 32)
RAMP = np.linspace(-20, 20, SHAPE[0] * SHAPE[1]).reshape(SHAPE)
LEVELS = {"control": 60.0, "treated": 150.0}  # a large treatment effect
REPEATS = ("2024_05_01", "2024_05_02", "2024_05_03")


def image(level, rng):
    """A uint8 image around ``level``, kept below 255 so it is not saturated."""
    return np.clip(np.rint(level + RAMP + rng.normal(0, 1, SHAPE)), 0, 254).astype(np.uint8)


def balanced(n_images=6, repeats=REPEATS, levels=None):
    """``{(condition, repeat): [level, ...]}``: the same spread of levels in every repeat."""
    spread = np.linspace(0.9, 1.1, n_images)
    levels = LEVELS if levels is None else levels
    return {
        (condition, repeat): [base * f for f in np.roll(spread, k)]
        for condition, base in levels.items()
        for k, repeat in enumerate(repeats)
    }


def path(condition, repeat, index):
    return f"{condition}/{repeat}/img_{index:02d}.tif"


def summaries_of(design, seed=0):
    """Summaries of one plane per image.

    An entry of ``design`` is a level, a pixel array, or a tuple of those, one per channel.
    """
    rng = np.random.default_rng(seed)
    records = []
    for (condition, repeat), images in design.items():
        for index, entry in enumerate(images):
            channels = entry if isinstance(entry, tuple) else (entry,)
            pixels = [c if isinstance(c, np.ndarray) else image(c, rng) for c in channels]
            histograms = [compute_plane_histogram(p) for p in pixels]
            records.append(
                PlaneRecord(path(condition, repeat, index), condition, repeat, SINGLE, histograms)
            )
    return image_summaries(records)


def found(issues, code=None):
    return [(i.code, i.items, i.channel) for i in issues if code is None or i.code == code]


def test_balanced_data_with_a_treatment_effect_is_silent():
    summaries = summaries_of(balanced())
    medians = {s.condition: s.median[0] for s in summaries}
    assert medians["treated"] > 2 * medians["control"]
    assert intensity_checks(summaries) == []


def test_a_shifted_image_is_an_outlier_in_its_median_and_high_percentile():
    design = balanced()
    design["control", "2024_05_02"][3] *= 1.6
    issues = intensity_checks(summaries_of(design))
    shifted = path("control", "2024_05_02", 3)
    assert found(issues) == [("image_intensity_outlier", (shifted,), 0)] * 2
    median, high = issues
    assert median.message.startswith(f"Image {shifted}, channel C0: its median intensity ")
    assert "the median of the other 5 images of condition 'control', biological repeat " in (
        median.message
    )
    assert high.message.startswith(f"Image {shifted}, channel C0: its 99.8th percentile ")
    assert all(issue.severity == "warning" and issue.action for issue in issues)


def test_outlier_message_gives_the_value_the_group_value_and_the_distance():
    design = balanced()
    design["treated", "2024_05_01"][0] *= 0.5
    summaries = summaries_of(design)
    issue = intensity_checks(summaries)[0]
    target = next(s for s in summaries if s.source_relative_path in issue.items)
    others = [
        s.median[0]
        for s in summaries
        if (s.condition, s.biological_repeat) == ("treated", "2024_05_01") and s is not target
    ]
    reference = np.median(others)
    deviation = NORMAL_CONSISTENCY * np.median(np.abs(np.subtract(others, reference)))
    distance = abs(target.median[0] - reference) / deviation
    assert distance > 3
    assert (
        f"its median intensity {target.median[0]:.6g} is {distance:.3g} robust deviations from "
        f"{reference:.6g}, the median of the other 5 images" in issue.message
    )
    assert f"(robust deviation {deviation:.6g}; limit 3)" in issue.message


def test_a_shifted_repeat_is_reported_for_its_condition_only():
    design = balanced()
    design["control", "2024_05_03"] = [level * 1.5 for level in design["control", "2024_05_03"]]
    issues = intensity_checks(summaries_of(design))
    assert found(issues) == [("repeat_intensity_shift", ("control/2024_05_03",), 0)]
    assert issues[0].message.startswith(
        "Biological repeat '2024_05_03' of condition 'control', channel C0: the median of its "
        "6 image medians, "
    )
    assert "the median of the 12 image medians of the other biological repeats of 'control'" in (
        issues[0].message
    )


def test_a_condition_with_one_repeat_has_no_repeat_check():
    design = {
        key: levels
        for key, levels in balanced(n_images=6).items()
        if key[0] == "treated" or key[1] == "2024_05_01"
    }
    design["control", "2024_05_01"] = [level * 0.5 for level in design["control", "2024_05_01"]]
    assert intensity_checks(summaries_of(design)) == []


def test_groups_below_min_images_are_skipped():
    design = balanced(n_images=4)
    design["control", "2024_05_01"][1] *= 1.8
    summaries = summaries_of(design)
    assert found(intensity_checks(summaries), "image_intensity_outlier") == []
    issues = intensity_checks(summaries, IntensityCheckSettings(min_images=4))
    assert (
        found(issues, "image_intensity_outlier")
        == [("image_intensity_outlier", (path("control", "2024_05_01", 1),), 0)] * 2
    )


def test_repeat_check_needs_min_images_in_the_other_repeats():
    design = balanced(n_images=2, repeats=REPEATS[:2])
    design["control", "2024_05_02"] = [level * 2 for level in design["control", "2024_05_02"]]
    summaries = summaries_of(design)
    # Two images in the other repeat: below the default of 5.
    assert intensity_checks(summaries) == []
    issues = intensity_checks(summaries, IntensityCheckSettings(min_images=2))
    # With two repeats, each is compared with the other, so both are reported.
    assert found(issues) == [
        ("repeat_intensity_shift", ("control/2024_05_01",), 0),
        ("repeat_intensity_shift", ("control/2024_05_02",), 0),
    ]


def test_a_zero_robust_deviation_skips_the_comparison():
    rng = np.random.default_rng(1)
    same = image(80, rng)
    design = {("control", "rep1"): [same] * 5 + [image(160, rng)]}
    summaries = summaries_of(design)
    assert len({s.median[0] for s in summaries[:5]}) == 1
    assert intensity_checks(summaries) == []


def test_saturated_images():
    rng = np.random.default_rng(2)
    many = image(100, rng)
    many.flat[:52] = 255  # 5.1 %
    few = image(100, rng)
    few.flat[:5] = 255  # 0.5 %
    design = {("control", "rep1"): [many, few, image(100, rng)]}
    summaries = summaries_of(design)
    issues = intensity_checks(summaries)
    assert found(issues) == [("saturated_image", (path("control", "rep1", 0),), 0)]
    assert issues[0].message == (
        f"Image {path('control', 'rep1', 0)}, channel C0: 5.08 % of its pixels are at the maximum "
        "of its pixel type, 255 (uint8), above the limit of 1 %."
    )
    issues = intensity_checks(summaries, IntensityCheckSettings(saturation_limit=0.001))
    assert [i.items[0] for i in issues] == [path("control", "rep1", 0), path("control", "rep1", 1)]
    assert intensity_checks(summaries, IntensityCheckSettings(saturation_limit=0.06)) == []


def test_constant_images_have_no_spread():
    design = {("control", "rep1"): [np.full(SHAPE, 7, np.uint8), np.zeros(SHAPE, np.uint8)]}
    design["control", "rep1"].append(image(100, np.random.default_rng(3)))
    issues = intensity_checks(summaries_of(design))
    assert found(issues) == [
        ("no_intensity_spread", (path("control", "rep1", 0),), 0),
        ("no_intensity_spread", (path("control", "rep1", 1),), 0),
    ]
    assert issues[0].message == (
        f"Image {path('control', 'rep1', 0)}, channel C0: its 1st and 99.8th percentiles are "
        "both 7, so this channel has no intensity spread and cannot be normalized at the image "
        "scope with a percentile method."
    )


def test_each_channel_is_checked_separately():
    design = {key: [(level, level / 2) for level in levels] for key, levels in balanced().items()}
    first, second = design["treated", "2024_05_02"][4]
    design["treated", "2024_05_02"][4] = (first, second * 0.4)  # only channel 1 is dimmer
    issues = intensity_checks(summaries_of(design))
    shifted = path("treated", "2024_05_02", 4)
    assert found(issues) == [("image_intensity_outlier", (shifted,), 1)] * 2
    assert all(", channel C1: " in issue.message for issue in issues)


def test_warnings_come_in_the_contract_order():
    design = balanced()
    design["control", "2024_05_01"][0] *= 1.6
    design["treated", "2024_05_03"] = [level * 1.4 for level in design["treated", "2024_05_03"]]
    rng = np.random.default_rng(4)
    saturated = image(60, rng)
    saturated.flat[:100] = 255
    design["control", "2024_05_02"][2] = saturated
    design["control", "2024_05_03"][5] = np.full(SHAPE, 60, np.uint8)
    codes = [issue.code for issue in intensity_checks(summaries_of(design))]
    assert codes == sorted(
        codes,
        key=[
            "image_intensity_outlier",
            "repeat_intensity_shift",
            "saturated_image",
            "no_intensity_spread",
        ].index,
    )
    assert set(codes) == {
        "image_intensity_outlier",
        "repeat_intensity_shift",
        "saturated_image",
        "no_intensity_spread",
    }


def test_checks_match_a_direct_implementation():
    # Summaries with random values, heavy tails and ties, compared with contract
    # §7.2 written out with NumPy.
    rng = np.random.default_rng(5)
    settings = IntensityCheckSettings(outlier_limit=2.5, saturation_limit=0.02, min_images=4)
    for _ in range(20):
        summaries = []
        for condition in ("a", "b"):
            for repeat in ("r1", "r2", "r3")[: rng.integers(1, 4)]:
                for index in range(rng.integers(1, 8)):
                    median = np.round(rng.standard_t(2, size=2) + 3 * (repeat == "r2"), 1)
                    high = median + np.round(np.abs(rng.standard_t(2, size=2)), 1)
                    low = np.where(rng.random(2) < 0.1, high, median - 1)
                    summaries.append(
                        ImageSummary(
                            path(condition, repeat, index),
                            condition,
                            repeat,
                            SINGLE,
                            np.dtype(np.uint16),
                            (1.0, 99.8),
                            tuple(map(float, median)),
                            tuple(map(float, low)),
                            tuple(map(float, high)),
                            tuple(map(float, rng.choice([0.0, 0.01, 0.05], size=2))),
                            (True, True),
                        )
                    )
        expected = direct_checks(summaries, settings)
        assert found(intensity_checks(summaries, settings)) == expected


def direct_checks(summaries, settings):
    def beyond(value, others):
        reference = np.median(others)
        deviation = NORMAL_CONSISTENCY * np.median(np.abs(np.subtract(others, reference)))
        return deviation > 0 and abs(value - reference) / deviation > settings.outlier_limit

    outliers, shifts, saturated, spread = [], [], [], []
    for s in summaries:
        group = [
            o
            for o in summaries
            if (o.condition, o.biological_repeat) == (s.condition, s.biological_repeat)
        ]
        for c in range(2):
            for statistic, order in (("median", 0), ("p_high", 1)):
                others = [getattr(o, statistic)[c] for o in group if o is not s]
                if len(group) >= settings.min_images and beyond(getattr(s, statistic)[c], others):
                    outliers.append(((s.source_relative_path, c, order), s.source_relative_path, c))
            if s.saturated_fraction[c] > settings.saturation_limit:
                saturated.append(("saturated_image", (s.source_relative_path,), c))
            if s.p_low[c] == s.p_high[c]:
                spread.append(("no_intensity_spread", (s.source_relative_path,), c))
    outliers = [("image_intensity_outlier", (p,), c) for _, p, c in sorted(outliers)]
    for condition in ("a", "b"):
        repeats = sorted({s.biological_repeat for s in summaries if s.condition == condition})
        for repeat in repeats if len(repeats) > 1 else ():
            for c in range(2):
                own = [
                    s.median[c]
                    for s in summaries
                    if (s.condition, s.biological_repeat) == (condition, repeat)
                ]
                others = [
                    s.median[c]
                    for s in summaries
                    if s.condition == condition and s.biological_repeat != repeat
                ]
                if len(others) >= settings.min_images and beyond(np.median(own), others):
                    shifts.append(("repeat_intensity_shift", (f"{condition}/{repeat}",), c))
    return outliers + shifts + saturated + spread


def test_settings_defaults_and_round_trip():
    settings = IntensityCheckSettings()
    assert (settings.outlier_limit, settings.saturation_limit, settings.min_images) == (
        3.0,
        0.01,
        5,
    )
    data = settings.to_dict()
    assert set(data) == {"library_version", "outlier_limit", "saturation_limit", "min_images"}
    assert IntensityCheckSettings.from_dict(data) == settings
    changed = IntensityCheckSettings(outlier_limit=4, saturation_limit=0, min_images=np.int64(8))
    assert IntensityCheckSettings.from_dict(changed.to_dict()) == changed
    assert changed.min_images == 8 and type(changed.min_images) is int
    assert IntensityCheckSettings.from_dict({}) == settings


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"outlier_limit": 0}, "outlier_limit must be a positive number"),
        ({"outlier_limit": "three"}, "outlier_limit must be a number"),
        ({"outlier_limit": float("inf")}, "outlier_limit must be a number"),
        ({"saturation_limit": 1}, "saturation_limit must be a fraction"),
        ({"saturation_limit": -0.1}, "saturation_limit must be a fraction"),
        ({"min_images": 1}, "min_images must be at least 2"),
        ({"min_images": 4.5}, "min_images must be a whole number"),
        ({"min_images": True}, "min_images must be a whole number"),
    ],
)
def test_invalid_settings_are_refused(changes, message):
    with pytest.raises(ValueError, match=message):
        IntensityCheckSettings(**changes)


def test_unknown_settings_are_refused():
    with pytest.raises(ValueError, match="Unknown intensity check setting.*outlier_limt"):
        IntensityCheckSettings.from_dict({"outlier_limt": 3})


def test_other_warnings_have_no_channel():
    issue = Issue("warning", "repeat_missing_conditions", "message")
    assert issue.channel is None
    assert replace(issue, channel=1) != issue
