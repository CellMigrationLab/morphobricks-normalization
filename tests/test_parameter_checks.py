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


def parameters(offset=(0.0,), scale=(1.0,), channels=None, scope="image"):
    return Parameters("z-score", scope, None, True, offset, scale, channels=channels)


# --- Offsets and scales (#36) ------------------------------------------------------------------


@pytest.mark.parametrize("value", BAD_NUMBERS)
def test_parameters_refuse_a_non_finite_offset(value):
    with pytest.raises(ValueError, match="Channel 1: the normalization offset must be a finite"):
        parameters(offset=(0.0, value), scale=(1.0, 1.0))
    with pytest.raises(ValueError, match="The min-max method: the normalization offset"):
        NormalizationParameters("min-max", value, 1.0)


@pytest.mark.parametrize("value", [*BAD_NUMBERS, 0.0, -0.0, -2.5])
def test_parameters_refuse_a_scale_that_is_not_finite_and_positive(value):
    with pytest.raises(ValueError, match="Channel 0: the normalization scale must be a finite"):
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
    original = parameters(offset=(1.5, -2.0), scale=(1e-300, 7.0), channels=(0, 3))
    assert (
        Parameters.from_columns(original.to_columns(), NormalizationSettings(method="zscore"))
        == original
    )


def test_fitted_parameters_are_valid_and_a_zero_spread_keeps_its_message():
    plane = np.arange(64, dtype=np.uint16).reshape(8, 8)
    record = PlaneRecord("c/r/a.tif", "c", "r", SINGLE, [compute_plane_histogram(plane)])
    fitted = fit([record], NormalizationSettings(method="zscore"))
    assert all(math.isfinite(o) for o in fitted.offset) and fitted.scale[0] > 0
    flat = PlaneRecord("c/r/b.tif", "c", "r", SINGLE, [compute_plane_histogram(plane * 0)])
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
    read = Parameters.from_columns(columns, NormalizationSettings(method="zscore"))
    assert read.scope == scope


# --- Channel indices (#38) ---------------------------------------------------------------------


@pytest.mark.parametrize("channel", [1.9, 1.0, True, np.bool_(True), "1", -1, None])
def test_channel_indices_are_never_truncated(channel):
    with pytest.raises(ValueError, match="A channel index is a whole number counted from 0"):
        parameters(offset=(0.0, 10.0), scale=(1.0, 1.0), channels=(0, channel))
    with pytest.raises(ValueError, match="A channel index is a whole number counted from 0"):
        parameters(offset=(0.0, 10.0), scale=(1.0, 1.0)).select([channel])


def test_integer_channel_indices_are_accepted():
    p = parameters(offset=(0.0, 10.0), scale=(1.0, 2.0), channels=(np.int64(0), np.uint8(4)))
    assert p.channels == (0, 4) and all(type(c) is int for c in p.channels)
    selected = p.select(c for c in [np.int32(4)])
    assert (selected.channels, selected.offset, selected.scale) == ((4,), (10.0,), (2.0,))


# --- Channels without finite pixels (#57) ------------------------------------------------------


def summaries_with(*channels):
    histograms = [compute_plane_histogram(np.asarray(c, np.float32), 100) for c in channels]
    return image_summaries([PlaneRecord("c/r/img.tif", "c", "r", SINGLE, histograms)])


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_a_channel_without_finite_pixels_is_not_exact_and_is_reported(missing):
    good = np.arange(16, dtype=np.float32).reshape(4, 4)
    (summary,) = summaries_with(good, np.full((4, 4), missing))
    assert summary.exact == (False, None)
    assert summary.to_row()["Histogram_Exact_C1"] is None
    (issue,) = unchecked_channels([summary])
    assert (issue.code, issue.channel, issue.items) == (
        "channel_without_finite_pixels",
        1,
        ("c/r/img.tif",),
    )
    assert "channel C1: it has no finite pixels" in issue.message
    assert intensity_checks([summary]) == []  # the four checks of §7.2 are unchanged


def test_partly_missing_planes_and_clean_channels_are_not_reported():
    plane = np.arange(16, dtype=np.float32).reshape(4, 4)
    partly = plane.copy()
    partly[0, 0] = np.nan
    flat = np.full((4, 4), 3.0)
    assert unchecked_channels(summaries_with(plane, partly, flat)) == []
