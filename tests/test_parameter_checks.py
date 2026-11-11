"""Regression tests for parameter validation and unchecked channels (issues #36, #37, #38, #57)."""

import math

import numpy as np
import pytest

from morphobricks_normalization import (
    SINGLE,
    NormalizationParameters,
    NormalizationSettings,
    Parameters,
    PlaneRecord,
    compute_plane_histogram,
    fit,
    image_summaries,
    intensity_checks,
    unchecked_channels,
)

BAD_NUMBERS = [math.nan, math.inf, -math.inf]


def parameters(offset=(0.0,), scale=(1.0,), channels=None, scope="image", group=None):
    if channels is None:
        channels = tuple(f"ch{c}" for c in range(len(offset)))
    return Parameters("z-score", scope, group, True, offset, scale, channels=channels)


# --- Offsets and scales (#36) ------------------------------------------------------------------


@pytest.mark.parametrize("value", BAD_NUMBERS)
def test_parameters_refuse_a_non_finite_offset(value):
    with pytest.raises(ValueError, match="Channel ch1: the normalization offset must be a finite"):
        parameters(offset=(0.0, value), scale=(1.0, 1.0))
    with pytest.raises(ValueError, match="The min-max method: the normalization offset"):
        NormalizationParameters("min-max", value, 1.0)


@pytest.mark.parametrize("value", [*BAD_NUMBERS, 0.0, -0.0, -2.5])
def test_parameters_refuse_a_scale_that_is_not_finite_and_positive(value):
    with pytest.raises(ValueError, match="Channel ch0: the normalization scale must be a finite"):
        parameters(scale=(value,))
    with pytest.raises(ValueError, match="normalization scale must be a finite number above 0"):
        NormalizationParameters("z-score", 0.0, value)


def test_small_scales_and_non_finite_pixels_are_kept():
    tiny = parameters(offset=(-3.0,), scale=(5e-324,))
    assert tiny.scale == (5e-324,)
    pixels = np.array([np.nan, np.inf, 1.0])
    result = parameters(offset=(1.0,), scale=(2.0,)).apply(pixels)
    assert np.isnan(result[0]) and np.isinf(result[1]) and result[2] == 0.0
    assert np.isnan(NormalizationParameters("z-score", 1.0, 2.0).apply(pixels)[0])


def test_accepted_parameters_round_trip_through_their_columns():
    original = parameters(offset=(1.5, -2.0), scale=(1e-300, 7.0), channels=("DNA", "PAX_488"))
    assert (
        Parameters.from_columns(original.to_columns(), NormalizationSettings(method="zscore"))
        == original
    )


def test_fitted_parameters_are_valid_and_a_zero_spread_keeps_its_message():
    plane = np.arange(64, dtype=np.uint16).reshape(8, 8)
    record = PlaneRecord(
        "c/r/a.tif",
        "c",
        "r",
        SINGLE,
        [compute_plane_histogram(plane)],
        position_id="p",
        channels=("DNA",),
    )
    fitted = fit([record], NormalizationSettings(method="zscore"))
    assert all(math.isfinite(o) for o in fitted.offset) and fitted.scale[0] > 0
    flat = PlaneRecord(
        "c/r/b.tif",
        "c",
        "r",
        SINGLE,
        [compute_plane_histogram(plane * 0)],
        position_id="p",
        channels=("DNA",),
    )
    with pytest.raises(ValueError, match="intensity spread is zero"):
        fit([flat], NormalizationSettings(method="zscore"))


# --- Scope of the columns (#37) ----------------------------------------------------------------


@pytest.mark.parametrize("scope", ["experiment", "", "Image", "condition"])
def test_from_columns_refuses_an_unknown_scope(scope):
    columns = parameters().to_columns() | {"normalization_scope": scope}
    with pytest.raises(ValueError, match=f"normalization_scope is {scope!r}, which is not a scope"):
        Parameters.from_columns(columns, NormalizationSettings(method="zscore"))


@pytest.mark.parametrize("scope", ["image", "patch", "biological_repeat"])
def test_from_columns_keeps_a_known_scope(scope):
    columns = parameters(scope=scope).to_columns()
    read = Parameters.from_columns(columns, NormalizationSettings(method="zscore", scope=scope))
    assert read.scope == scope


@pytest.mark.parametrize("row", ["image", "patch", "biological_repeat"])
@pytest.mark.parametrize("wanted", ["image", "patch", "biological_repeat"])
def test_from_columns_refuses_a_row_of_another_scope(row, wanted):
    columns = parameters(scope=row).to_columns()
    settings = NormalizationSettings(method="zscore", scope=wanted)
    if row == wanted:
        assert Parameters.from_columns(columns, settings).scope == row
        return
    with pytest.raises(
        ValueError, match=f"normalization_scope is {row!r}, but the settings use scope {wanted!r}"
    ):
        Parameters.from_columns(columns, settings)


def test_the_scope_check_keeps_group_and_time_point_as_written():
    # The group (an image, a repeat group, a time point) is the row's own; only the scope
    # is compared, so per-time-point groups of one scope read back unchanged.
    for group in ("A", "A::t=3", "control/r1/img.tif::t=0", None):
        written = parameters(scope="biological_repeat", group=group)
        settings = NormalizationSettings(
            method="zscore", scope="biological_repeat", per_time_point=True
        )
        assert Parameters.from_columns(written.to_columns(), settings) == written


# --- Channel IDs (#38; ADR 0003 HV2) -----------------------------------------------------------


@pytest.mark.parametrize("channel", [1.9, 1, np.int64(1), True, np.bool_(True), None])
def test_channel_indices_are_refused_not_read_as_ids(channel):
    with pytest.raises(ValueError, match="never by its position in the image"):
        parameters(offset=(0.0, 10.0), scale=(1.0, 1.0), channels=("A", channel))
    with pytest.raises(ValueError, match="never by its position in the image"):
        parameters(offset=(0.0, 10.0), scale=(1.0, 1.0)).select([channel])


@pytest.mark.parametrize(
    "channel", ["1", "", " DNA", "DNA ", "dna-1", "D__NA", "_DNA", "DNA_", "A;B"]
)
def test_text_that_is_not_a_token_is_refused(channel):
    with pytest.raises(ValueError, match="is not a canonical channel ID"):
        parameters(offset=(0.0,), scale=(1.0,), channels=(channel,))


def test_channel_ids_with_underscores_are_accepted_and_kept_as_written():
    p = parameters(offset=(0.0, 10.0), scale=(1.0, 2.0), channels=("PAX_488", "dna"))
    assert p.channels == ("PAX_488", "dna")
    selected = p.select(c for c in ["dna"])
    assert (selected.channels, selected.offset, selected.scale) == (("dna",), (10.0,), (2.0,))


# --- Channels without finite pixels (#57) ------------------------------------------------------


def summaries_with(*channels):
    histograms = [compute_plane_histogram(np.asarray(c, np.float32), 100) for c in channels]
    return image_summaries(
        [
            PlaneRecord(
                "c/r/img.tif",
                "c",
                "r",
                SINGLE,
                histograms,
                position_id="p",
                channels=("DNA", "PHASE", "ACTIN")[: len(histograms)],
            )
        ]
    )


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_a_channel_without_finite_pixels_is_not_exact_and_is_reported(missing):
    good = np.arange(16, dtype=np.float32).reshape(4, 4)
    (summary,) = summaries_with(good, np.full((4, 4), missing))
    assert summary.channels == ("DNA", "PHASE")
    assert summary.exact == (False, None)
    assert summary.to_row()["Histogram_Exact_PHASE"] is None
    (issue,) = unchecked_channels([summary])
    assert (issue.code, issue.channel, issue.items) == (
        "channel_without_finite_pixels",
        "PHASE",
        ("c/r/img.tif",),
    )
    assert "channel PHASE: it has no finite pixels" in issue.message
    assert intensity_checks([summary]) == []  # the four checks of §7.2 are unchanged


def test_partly_missing_planes_and_clean_channels_are_not_reported():
    plane = np.arange(16, dtype=np.float32).reshape(4, 4)
    partly = plane.copy()
    partly[0, 0] = np.nan
    flat = np.full((4, 4), 3.0)
    assert unchecked_channels(summaries_with(plane, partly, flat)) == []
