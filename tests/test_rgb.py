"""RGB images: StarDist's percentile normalization (contract §3.3, §5, N33, N39, N40, N48, N49)."""

import numpy as np
import pytest

from morphobricks_normalization import (
    MethodNotAllowedError,
    NormalizationSettings,
    NoValidPixelsError,
    Parameters,
    PlaneRecord,
    PooledHistogram,
    WeightedHistogram,
    check_settings,
    compute_plane_histogram,
    fit,
    group_name,
    rgb_percentile,
    scope_groups,
)
from morphobricks_normalization.fitting import offset_column, scale_column
from morphobricks_normalization.rgb import EPSILON, apply_rgb
from simulation import cached_rgb_design, make_rgb_records, simulate_rgb_design

RGB_MESSAGE = (
    "RGB normalization needs images with exactly three 8- or 16-bit colour channels, but these "
    "images have {} channels of type {}. Check the axis mapping (the samples axis must be the "
    "channel axis), or choose image type 'fluorescence'."
)
DTYPES = {"uint8": {"dtype": np.uint8}, "uint16": {"dtype": np.uint16, "top": 4095}}


def rgb(**changes):
    return NormalizationSettings(image_type="rgb", **changes)


def direct(pixels, low=1.0, high=99.8):
    """Contract §3.3 on the pixels: np.percentile, then CSBDeep's float32 arithmetic."""
    x = np.asarray(pixels)
    mi, ma = np.percentile(x, low), np.percentile(x, high)
    x32, mi32, ma32 = x.astype(np.float32), np.float32(mi), np.float32(ma)
    return mi, ma - mi + 1e-20, (x32 - mi32) / (ma32 - mi32 + np.float32(1e-20))


def assert_rgb(parameters, offset, scale):
    assert parameters.channels == (0, 1, 2)
    assert parameters.offset == (offset,) * 3
    assert parameters.scale == (scale,) * 3


def group_pixels(images, members):
    lookup = {(path, t): pixels for _, path, t, pixels in images}
    return np.stack([lookup[r.source_relative_path, r.t] for r in members])


@pytest.mark.parametrize("kind", DTYPES)
def test_image_scope_matches_the_direct_formula_from_pixels_and_from_the_file(kind):
    images, records = cached_rgb_design(seed=30, n_times=2, **DTYPES[kind])
    settings = rgb()
    groups, _ = scope_groups(records, settings)
    assert len(groups) == 8
    for key, members in groups.items():
        pixels = group_pixels(images, members)  # (T, Y, X, 3)
        offset, scale, expected = direct(pixels)
        from_file = fit(members, settings, group=group_name(key))
        from_pixels = fit(pixels, settings, group=group_name(key), channel_axis=-1)
        assert from_file == from_pixels
        assert_rgb(from_file, offset, scale)
        assert from_file.exact and from_file.issues == ()
        assert from_file.method == "rgb percentile 1-99.8"
        normalized = from_file.apply(pixels, channel_axis=-1)
        assert normalized.dtype == np.float32
        np.testing.assert_array_equal(normalized, expected)


@pytest.mark.parametrize("kind", DTYPES)
def test_matches_csbdeep_normalize_bit_for_bit(kind):
    utils = pytest.importorskip("csbdeep.utils")
    images, _ = cached_rgb_design(seed=31, **DTYPES[kind])
    for _, _, _, pixels in images:
        expected = utils.normalize(pixels, 1, 99.8, axis=(0, 1, 2))
        parameters = fit(pixels, rgb(scope="patch"), channel_axis=-1)
        assert parameters.offset[0] == np.percentile(pixels, 1)
        normalized = parameters.apply(pixels, channel_axis=-1)
        assert normalized.dtype == expected.dtype == np.float32
        np.testing.assert_array_equal(normalized, expected)


def test_percentiles_other_than_the_default():
    images, records = cached_rgb_design(seed=32)
    settings = rgb(percentile=(0.5, 95))
    assert settings.method_name == "rgb percentile 0.5-95"
    pixels = images[0][3]
    offset, scale, expected = direct(pixels, 0.5, 95)
    parameters = fit(records[:1], settings)
    assert_rgb(parameters, offset, scale)
    np.testing.assert_array_equal(parameters.apply(pixels, channel_axis=2), expected)


@pytest.mark.parametrize("pooling", ["pixels", "conditions"])
def test_biological_repeat_pools_red_green_and_blue_of_the_repeat_group(pooling):
    images, records = cached_rgb_design(seed=33)
    settings = rgb(scope="biological_repeat", pooling=pooling)
    groups, _ = scope_groups(records, settings)
    assert sorted(groups) == [("2024_05_01",), ("2024_05_02",)]
    for key, members in groups.items():
        assert {r.condition for r in members} == {"control", "treated"}
        offset, scale, _ = direct(group_pixels(images, members))
        # Both conditions have the same pixel count, so every weight is 1 (N49).
        parameters = fit(members, settings, group=group_name(key))
        assert_rgb(parameters, offset, scale)
        assert parameters.scope == "biological_repeat"


def test_pooling_weights_agree_for_a_single_condition():
    images = simulate_rgb_design(seed=34, design={"control": ("r1", "r2")}, n_images=3)
    records = make_rgb_records(images)
    by_pixels = fit(records, rgb(scope="biological_repeat", pooling="pixels"))
    by_conditions = fit(records, rgb(scope="biological_repeat", pooling="conditions"))
    assert by_pixels == by_conditions
    offset, scale, _ = direct(np.stack([pixels for *_, pixels in images]))
    assert_rgb(by_pixels, offset, scale)


def test_condition_weights_count_as_repeated_pixels():
    # Contract §4.3 and N49: with whole-number weights, the weighted linear
    # percentile is np.percentile of the pixels repeated by their weight.
    images = simulate_rgb_design(seed=35, design={"a": ("r1",)}, n_images=3)
    # Condition b is darker, so weighting it more moves the percentiles.
    images += simulate_rgb_design(seed=36, design={"b": ("r1",)}, n_images=1, top=150)
    records = make_rgb_records(images)
    a = np.stack([p for labels, *_, p in images if labels[0] == "a"])
    b = np.stack([p for labels, *_, p in images if labels[0] == "b"])
    assert a.size == 3 * b.size
    parameters = fit(records, rgb(scope="biological_repeat", pooling="conditions"))
    offset, scale, _ = direct(np.concatenate([a.ravel(), np.tile(b.ravel(), 3)]))
    assert_rgb(parameters, offset, scale)
    by_pixels = fit(records, rgb(scope="biological_repeat"))
    assert_rgb(by_pixels, *direct(np.concatenate([a.ravel(), b.ravel()]))[:2])
    assert by_pixels.offset != parameters.offset or by_pixels.scale != parameters.scale


def test_per_time_point_groups():
    images, records = cached_rgb_design(seed=37, n_times=2)
    settings = rgb(per_time_point=True)
    groups, _ = scope_groups(records, settings)
    for key, members in groups.items():
        assert {r.t for r in members} == {key[1]}
        offset, scale, _ = direct(group_pixels(images, members))
        assert_rgb(fit(members, settings, group=group_name(key)), offset, scale)


def test_patch_scope_from_pixels_in_any_axis_order():
    images, _ = cached_rgb_design(seed=38)
    pixels = images[0][3][3:15, 5:20]
    offset, scale, expected = direct(pixels)
    channels_first = np.moveaxis(pixels, -1, 0)
    parameters = fit(channels_first, rgb(scope="patch"), group="img::p1", channel_axis=0)
    assert_rgb(parameters, offset, scale)
    assert (parameters.scope, parameters.group) == ("patch", "img::p1")
    normalized = parameters.apply(channels_first, channel_axis=0)
    np.testing.assert_array_equal(normalized, np.moveaxis(expected, -1, 0))


def test_a_constant_image_gives_zeros_not_an_error():
    pixels = np.full((5, 6, 3), 200, np.uint8)
    parameters = fit(pixels, rgb(scope="patch"), channel_axis=-1)
    assert_rgb(parameters, 200.0, EPSILON)
    normalized = parameters.apply(pixels, channel_axis=-1)
    np.testing.assert_array_equal(normalized, np.zeros(pixels.shape, np.float32))
    np.testing.assert_array_equal(normalized, direct(pixels)[2])
    # Read back from its columns, the 1e-20 scale gives the same zeros.
    again = Parameters.from_columns(parameters.to_columns(), rgb(scope="patch"))
    assert again == parameters
    np.testing.assert_array_equal(again.apply(pixels, channel_axis=-1), normalized)
    # A constant colour (different R, G and B) still spreads its three values.
    colour = np.empty((4, 4, 3), np.uint8)
    colour[...] = (10, 20, 30)
    offset, scale, expected = direct(colour)
    parameters = fit(colour, rgb(scope="patch"), channel_axis=-1)
    assert_rgb(parameters, offset, scale)
    np.testing.assert_array_equal(parameters.apply(colour, channel_axis=-1), expected)


def test_fluorescence_still_refuses_zero_spread():
    pixels = np.full((5, 6, 3), 200, np.uint8)
    with pytest.raises(ValueError, match="intensity spread is zero"):
        fit(pixels, NormalizationSettings(scope="patch"), channel_axis=-1)


def test_columns_hold_the_shared_value_and_round_trip_exactly():
    _, records = cached_rgb_design(seed=39, **DTYPES["uint16"])
    settings = rgb()
    parameters = fit(records[:2], settings, group="control/2024_05_01/img_01.tif")
    columns = parameters.to_columns()
    for c in (0, 1, 2):
        assert columns[offset_column(c)] == repr(parameters.offset[0])
        assert columns[scale_column(c)] == repr(parameters.scale[0])
    assert columns["normalization_method"] == "rgb percentile 1-99.8"
    assert offset_column(3) not in columns
    again = Parameters.from_columns(columns, settings)
    assert again == parameters
    assert (again.offset, again.scale, again.channels) == (
        parameters.offset,
        parameters.scale,
        parameters.channels,
    )


def test_from_columns_refuses_incomplete_or_different_rgb_values():
    parameters = Parameters("rgb percentile 1-99.8", "image", "g", True, (2.0,) * 3, (5.0,) * 3)
    columns = parameters.to_columns()
    with pytest.raises(ValueError, match="Normalization_Offset_C2, Normalization_Scale_C2"):
        Parameters.from_columns({**columns, offset_column(2): "", scale_column(2): ""}, rgb())
    with pytest.raises(ValueError, match="also holds channel"):
        Parameters.from_columns({**columns, offset_column(3): "1.0", scale_column(3): "2.0"}, rgb())
    with pytest.raises(ValueError, match="the same offset and scale"):
        Parameters.from_columns({**columns, offset_column(1): "2.5"}, rgb())
    with pytest.raises(ValueError, match="the settings use 'percentile 1-99.8'"):
        Parameters.from_columns(columns, NormalizationSettings())


@pytest.mark.parametrize("shape", [(6, 6, 2), (6, 6, 4), (6, 6, 1)])
def test_wrong_channel_counts_are_refused(shape):
    with pytest.raises(ValueError) as error:
        fit(np.zeros(shape, np.uint8), rgb(scope="patch"), channel_axis=-1)
    assert str(error.value) == RGB_MESSAGE.format(shape[-1], "uint8")
    with pytest.raises(ValueError, match="have 1 channels of type uint16"):
        fit(np.zeros((6, 6), np.uint16), rgb(scope="patch"))
    histogram = compute_plane_histogram(np.arange(10, dtype=np.uint8))
    row = PlaneRecord("c/r/a.tif", "c", "r", "single", (histogram,) * shape[-1])
    with pytest.raises(ValueError) as error:
        fit([row], rgb())
    assert str(error.value) == RGB_MESSAGE.format(shape[-1], "uint8")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_rgb_is_refused(dtype):
    pixels = np.random.default_rng(40).random((6, 6, 3)).astype(dtype)
    with pytest.raises(ValueError) as error:
        fit(pixels, rgb(scope="patch"), channel_axis=-1)
    assert str(error.value) == RGB_MESSAGE.format(3, np.dtype(dtype).name)
    rows = make_rgb_records([(("c", "r", "single"), "c/r/a.tif", 0, pixels)])
    with pytest.raises(ValueError, match=f"3 channels of type {np.dtype(dtype).name}"):
        fit(rows, rgb())


def test_wrong_choices_are_refused_before_any_pixel_is_read():
    with pytest.raises(MethodNotAllowedError, match="StarDist's percentile method only"):
        rgb(method="zscore")
    with pytest.raises(MethodNotAllowedError, match="needs the preprocessing brick"):
        check_settings(rgb(scope="biological_repeat"), statistics_file=False)
    assert check_settings(rgb(scope="patch"), statistics_file=False) == []
    with pytest.raises(ValueError, match="0 <= low < high <= 100"):
        rgb(percentile=(99.8, 1))


def test_approximate_histograms_stay_within_their_bound():
    images, records = cached_rgb_design(seed=41, bin_count=16)
    parameters = fit(records[:2], rgb(), group="g")
    assert not parameters.exact
    assert [issue.code for issue in parameters.issues] == ["approximate_parameters"]
    pixels = group_pixels(images, records[:2])
    bound = max(h.max_representative_error for r in records[:2] for h in r.histograms)
    offset, scale, _ = direct(pixels)
    assert abs(parameters.offset[0] - offset) <= bound
    assert abs(parameters.offset[0] + parameters.scale[0] - (offset + scale)) <= bound


def test_a_group_without_pixels_names_the_three_channels():
    empty = compute_plane_histogram(np.zeros((0, 0), np.uint8))
    row = PlaneRecord("c/r/a.tif", "c", "r", "single", (empty,) * 3)
    with pytest.raises(NoValidPixelsError, match="Group 'g', channels R, G and B"):
        fit([row], rgb(), group="g")


def test_apply_returns_a_new_array_and_never_changes_the_input():
    pixels = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    before = pixels.copy()
    parameters = fit(pixels, rgb(scope="patch"), channel_axis=-1)
    normalized = parameters.apply(pixels, channel_axis=-1)
    np.testing.assert_array_equal(pixels, before)
    assert not np.shares_memory(normalized, pixels)
    with pytest.raises(ValueError, match="give channel_axis"):
        parameters.apply(pixels)
    float_pixels = pixels.astype(np.float32)
    again = parameters.apply(float_pixels, channel_axis=-1)
    assert not np.shares_memory(again, float_pixels)
    np.testing.assert_array_equal(again, normalized)


def test_apply_rgb_rebuilds_csbdeeps_upper_percentile():
    # offset + scale - 1e-20 rounds to the same float32 as the high percentile.
    offset, scale, expected = direct(np.array([[[3, 7, 250]]], np.uint8), 1, 99.8)
    np.testing.assert_array_equal(
        apply_rgb(np.array([[[3, 7, 250]]], np.uint8), offset, scale), expected
    )


def test_rgb_percentile_uses_linear_interpolation():
    pooled = PooledHistogram([compute_plane_histogram(np.array([0, 10], np.uint8))])
    assert rgb_percentile(pooled, 25, 75) == (2.5, 7.5 - 2.5 + EPSILON)
    with pytest.raises(ValueError, match="0 <= low < high <= 100"):
        rgb_percentile(pooled, 50, 50)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16])
def test_linear_percentile_equals_numpy_percentile(dtype):
    rng = np.random.default_rng(42)
    info = np.iinfo(dtype)
    for size in (1, 2, 3, 7, 100, 999, 1500):
        x = rng.integers(max(info.min, -3000), min(info.max, 3000), size, endpoint=True)
        x = x.astype(dtype)
        parts = [part for part in np.array_split(x, 3) if part.size]
        pooled = PooledHistogram([compute_plane_histogram(part) for part in parts])
        unit = WeightedHistogram([pooled], [1.0])
        for q in (0, 0.1, 1, 25, 33.3, 50, 66.7, 99, 99.8, 100, *rng.uniform(0, 100, 10)):
            assert pooled.linear_percentile(q) == np.percentile(x, q), (size, q)
            assert unit.linear_percentile(q) == pooled.linear_percentile(q)
    with pytest.raises(ValueError, match="between 0 and 100"):
        pooled.linear_percentile(101)
    with pytest.raises(ValueError, match="between 0 and 100"):
        unit.linear_percentile(-1)
