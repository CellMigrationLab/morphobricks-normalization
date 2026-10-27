import math
import re

import numpy as np
import pytest

from morphobricks_normalization import (
    MethodNotAllowedError,
    NormalizationParameters,
    NormalizationSettings,
    NoValidPixelsError,
    Parameters,
    PlaneRecord,
    PooledHistogram,
    compute_plane_histogram,
    fit,
    group_name,
    min_max,
    percentile_range,
    robust_z_score,
    scope_groups,
    z_score,
)
from morphobricks_normalization.fitting import COLUMNS, offset_column, scale_column
from morphobricks_normalization.settings import FLUORESCENCE_METHODS
from simulation import cached_design

METHOD_SETTINGS = [
    NormalizationSettings(method="minmax"),
    NormalizationSettings(method="percentile"),
    NormalizationSettings(method="percentile", percentile=(0.5, 95)),
    NormalizationSettings(method="zscore"),
    NormalizationSettings(method="robust_zscore"),
    NormalizationSettings(method="robust_zscore", consistency=1.0),
]


def with_scope(settings, **changes):
    data = settings.to_dict() | changes
    return NormalizationSettings.from_dict(data)


def direct(pixels, settings):
    """The formulas of contract §3.2 computed on the pixels themselves."""
    x = np.asarray(pixels, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if settings.method == "minmax":
        return x.min(), x.max() - x.min()
    if settings.method == "percentile":
        low, high = (np.percentile(x, q, method="inverted_cdf") for q in settings.percentile)
        return low, high - low
    if settings.method == "zscore":
        return x.mean(), x.std()
    median = np.percentile(x, 50, method="inverted_cdf")
    mad = np.percentile(np.abs(x - median), 50, method="inverted_cdf")
    return median, mad * settings.consistency


def assert_matches_direct(parameters, channel_pixels, settings):
    assert parameters.channel_count == len(channel_pixels)
    for c, pixels in enumerate(channel_pixels):
        offset, scale = direct(pixels, settings)
        if settings.method == "zscore":
            assert parameters.offset[c] == pytest.approx(offset, rel=1e-12)
            assert parameters.scale[c] == pytest.approx(scale, rel=1e-12)
        else:
            assert (parameters.offset[c], parameters.scale[c]) == (offset, scale)


def group_pixels(planes, members):
    """The pixels of each channel of a group's XYC planes, channel 0 first."""
    lookup = {(path, c, t): pixels for _, path, c, t, pixels in planes}
    (n_channels,) = {r.channel_count for r in members}
    return [
        np.concatenate([lookup[r.source_relative_path, c, r.t].ravel() for r in members])
        for c in range(n_channels)
    ]


@pytest.mark.parametrize("settings", METHOD_SETTINGS, ids=lambda s: s.method_name)
@pytest.mark.parametrize("scope", ["biological_repeat", "image"])
def test_every_method_matches_its_formula_at_every_file_scope(settings, scope):
    settings = with_scope(settings, scope=scope)
    planes, records = cached_design(seed=10, n_times=2)
    groups, _ = scope_groups(records, settings)
    for key, members in groups.items():
        parameters = fit(members, settings, group=group_name(key))
        assert_matches_direct(parameters, group_pixels(planes, members), settings)
        assert parameters.exact and parameters.issues == ()
        assert (parameters.scope, parameters.group) == (scope, key[0])
        assert parameters.method == settings.method_name


@pytest.mark.parametrize("settings", METHOD_SETTINGS, ids=lambda s: s.method_name)
def test_numbers_are_those_of_the_existing_methods(settings):
    functions = {
        "minmax": min_max,
        "percentile": lambda pooled: percentile_range(pooled, *settings.percentile),
        "zscore": z_score,
        "robust_zscore": lambda pooled: robust_z_score(pooled, settings.consistency),
    }
    _, records = cached_design(seed=11, dtype=np.float32, bin_count=512)
    parameters = fit(records, settings)
    for c in (0, 1):
        expected = functions[settings.method](PooledHistogram(records, c))
        assert (parameters.offset[c], parameters.scale[c]) == (expected.offset, expected.scale)
        assert parameters.method == expected.method
    assert not parameters.exact


@pytest.mark.parametrize("method", FLUORESCENCE_METHODS)
def test_image_scope_from_pixels_equals_image_scope_from_the_file(method):
    settings = NormalizationSettings(method=method, scope="image")
    planes, records = cached_design(seed=12, n_channels=3, n_times=2)
    groups, _ = scope_groups(records, settings)
    for (path,), members in list(groups.items())[:2]:
        # The image as the calling brick holds it: (T, C, Y, X).
        stack = np.stack(
            [
                np.stack([p for _, q, c, t, p in planes if q == path and t == time])
                for time in range(2)
            ]
        )
        from_pixels = fit(stack, settings, group=path, channel_axis=1)
        from_file = fit(members, settings, group=path)
        assert from_pixels == from_file
        assert fit(stack, settings, group=path, channel_axis=-3) == from_file


@pytest.mark.parametrize("settings", METHOD_SETTINGS, ids=lambda s: s.method_name)
def test_patch_scope_from_pixels(settings):
    settings = with_scope(settings, scope="patch")
    rng = np.random.default_rng(13)
    patch = rng.integers(100, 4000, size=(16, 16, 2)).astype(np.uint16)
    parameters = fit(patch, settings, group="control/r1/img.tif::7", channel_axis=2)
    assert_matches_direct(parameters, [patch[..., 0], patch[..., 1]], settings)
    assert parameters.group == "control/r1/img.tif::7" and parameters.scope == "patch"
    # Pixels selected by a mask: one row of values per channel.
    mask = rng.random((16, 16)) > 0.5
    selected = np.stack([patch[..., c][mask] for c in range(2)])
    assert_matches_direct(fit(selected, settings, channel_axis=0), list(selected), settings)


def test_per_time_point_groups_fit_each_time_point():
    planes, records = cached_design(seed=14, n_channels=1, n_times=3)
    settings = NormalizationSettings(scope="biological_repeat", per_time_point=True)
    groups, _ = scope_groups(records, settings)
    for key, members in groups.items():
        parameters = fit(members, settings, group=group_name(key))
        assert parameters.group == f"{key[0]}::t={key[1]}"
        assert_matches_direct(parameters, group_pixels(planes, members), settings)


def test_apply_matches_the_formula_per_channel():
    parameters = Parameters("z-score", "image", "a.tif", True, (10.0, -3.5), (2.0, 0.25))
    rng = np.random.default_rng(15)
    pixels = rng.normal(0, 50, size=(2, 5, 6))
    pixels[1, 0, 0] = np.nan
    original = pixels.copy()
    result = parameters.apply(pixels, channel_axis=0)
    expected = np.stack([(pixels[0] - 10.0) / 2.0, (pixels[1] + 3.5) / 0.25]).astype(np.float32)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float32 and np.isnan(result[1, 0, 0])
    np.testing.assert_array_equal(pixels, original)  # input never modified
    moved = parameters.apply(np.moveaxis(pixels, 0, -1), channel_axis=-1)
    np.testing.assert_array_equal(moved, np.moveaxis(expected, 0, -1))


def test_apply_with_one_channel_equals_the_existing_apply():
    pooled = PooledHistogram([compute_plane_histogram(np.arange(1000, dtype=np.uint16))])
    existing = robust_z_score(pooled)
    parameters = Parameters(
        existing.method, "patch", None, True, (existing.offset,), (existing.scale,)
    )
    pixels = np.random.default_rng(16).integers(0, 1000, (8, 8)).astype(np.uint16)
    np.testing.assert_array_equal(parameters.apply(pixels), existing.apply(pixels))
    no_clip = parameters.apply(np.array([-1e6, 1e6]))
    assert no_clip.min() < -1 and no_clip.max() > 1


def test_apply_refuses_a_wrong_channel_axis():
    parameters = Parameters("min-max", "image", None, True, (0.0, 1.0), (1.0, 2.0))
    with pytest.raises(ValueError, match="give channel_axis"):
        parameters.apply(np.zeros((2, 4)))
    with pytest.raises(ValueError, match="has 3 channel"):
        parameters.apply(np.zeros((3, 4)), channel_axis=0)
    with pytest.raises(ValueError, match="outside"):
        parameters.apply(np.zeros((2, 4)), channel_axis=2)


VALUES = [0.1, 1 / 3, 1e-300, 123456789.123456789, -2.5e-7, 65535.0, math.pi]
CHANNEL_COLUMN = re.compile(r"Normalization_(Offset|Scale)_C(0|[1-9][0-9]*)")


@pytest.mark.parametrize("channels", [(0,), (0, 1), (0, 1, 2, 3), (2,), (1, 3)])
def test_columns_round_trip_exactly(channels):
    rng = np.random.default_rng(len(channels) + sum(channels))
    offset = tuple(rng.choice(VALUES, len(channels)))
    scale = tuple(abs(v) for v in rng.choice(VALUES, len(channels)))
    for group in (None, "control/r1/img.tif::5::t=2"):
        parameters = Parameters(
            "percentile 1-99.8", "patch", group, False, offset, scale, channels=channels
        )
        columns = parameters.to_columns()
        pairs = [name for c in channels for name in (offset_column(c), scale_column(c))]
        assert list(columns) == [*COLUMNS, *pairs]  # grouped by channel
        assert all(isinstance(value, str) for value in columns.values())
        assert Parameters.from_columns(columns, NormalizationSettings()) == parameters
    assert columns["normalization_exact"] == "false"
    assert columns[f"Normalization_Offset_C{channels[0]}"] == repr(float(offset[0]))
    assert columns[f"Normalization_Scale_C{channels[-1]}"] == repr(float(scale[-1]))


def test_columns_of_a_fit_round_trip_exactly():
    settings = NormalizationSettings(method="robust_zscore", scope="biological_repeat")
    _, records = cached_design(seed=17, n_channels=3, bin_count=300)
    groups, _ = scope_groups(records, settings)
    for key, members in groups.items():
        parameters = fit(members, settings, group=group_name(key))
        columns = parameters.to_columns()
        assert columns["normalization_method"] == "robust z-score"
        assert columns["normalization_scope"] == "biological_repeat"
        assert columns["normalization_group"] == key[0]
        assert parameters.channels == (0, 1, 2)
        per_channel = [name for name in columns if CHANNEL_COLUMN.fullmatch(name)]
        assert per_channel == [
            f"Normalization_{feature}_C{c}" for c in range(3) for feature in ("Offset", "Scale")
        ]
        assert Parameters.from_columns(columns, settings) == parameters


def test_a_single_channel_patch_fills_only_its_own_channel():
    parameters = Parameters("z-score", "image", "a.tif", True, (1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    second = parameters.select([1])
    assert (second.channels, second.offset, second.scale) == ((1,), (2.0,), (5.0,))
    np.testing.assert_array_equal(second.apply(np.array([2.0, 7.0])), [0.0, 1.0])
    assert parameters.select([2, 0]).channels == (0, 2)
    with pytest.raises(ValueError, match=r"channel\(s\) \[3\] cannot be selected"):
        parameters.select([3])
    # A table of single-channel patches of every channel: each row has the columns
    # of every channel, blank except for its own (as a CSV or pandas reader gives them).
    settings = NormalizationSettings(method="zscore")
    header = list(parameters.to_columns())
    for c in range(3):
        written = parameters.select([c]).to_columns()
        row = {name: written.get(name, "") for name in header}
        assert Parameters.from_columns(row, settings) == parameters.select([c])
        row = {name: written.get(name, float("nan")) for name in header}
        assert Parameters.from_columns(row, settings) == parameters.select([c])


def test_from_columns_reads_numbers_and_booleans_as_a_table_reader_gives_them():
    row = {
        "normalization_method": "z-score",
        "normalization_scope": "image",
        "normalization_group": float("nan"),  # blank cell read by pandas
        "normalization_exact": True,
        "Normalization_Offset_C0": 12.5,
        "Normalization_Scale_C0": np.float64(0.1),
        "Normalization_Offset_C1": float("nan"),
        "Normalization_Scale_C1": None,
        "image_file": "a.tif",
        "Normalization_Offset_C2_extra": "ignored",
    }
    parameters = Parameters.from_columns(row, NormalizationSettings(method="zscore"))
    assert parameters == Parameters("z-score", "image", None, True, (12.5,), (0.1,))


@pytest.mark.parametrize(
    "change, message",
    [
        ({"normalization_scope": None}, "normalization_scope"),
        ({"Normalization_Scale_C0": None}, "Column Normalization_Scale_C0 is blank or missing"),
        ({"Normalization_Offset_C1": ""}, "Column Normalization_Offset_C1 is blank or missing"),
        ({"Normalization_Offset_C0": "1.0;2.0"}, "Normalization_Offset_C0 cannot be read"),
        ({"Normalization_Scale_C1": "0.0"}, "Normalization_Scale_C1 must hold a positive"),
        ({"Normalization_Offset_C0": "nan"}, "Normalization_Offset_C0 must hold a finite"),
        ({"normalization_exact": "maybe"}, "normalization_exact must be 'true' or 'false'"),
        ({"normalization_method": "min-max"}, "normalization_method is 'min-max'"),
        (
            {name: None for name in ("Normalization_Offset_C0", "Normalization_Scale_C0")}
            | {name: None for name in ("Normalization_Offset_C1", "Normalization_Scale_C1")},
            "Normalization_Offset_C0 and Normalization_Scale_C0 are missing",
        ),
        (
            {name: "" for name in ("Normalization_Offset_C0", "Normalization_Scale_C0")}
            | {name: "" for name in ("Normalization_Offset_C1", "Normalization_Scale_C1")},
            "column of this row is blank",
        ),
    ],
)
def test_from_columns_names_the_column_it_cannot_read(change, message):
    parameters = Parameters("percentile 1-99.8", "image", "a", True, (1.0, 3.0), (2.0, 4.0))
    columns = parameters.to_columns()
    for name, value in change.items():
        if value is None:
            del columns[name]
        else:
            columns[name] = value
    with pytest.raises(ValueError, match=message):
        Parameters.from_columns(columns, NormalizationSettings())


def test_parameters_refuse_channels_that_do_not_match_the_values():
    with pytest.raises(ValueError, match="one channel index per offset"):
        Parameters("min-max", "image", None, True, (0.0, 1.0), (1.0, 1.0), channels=(0,))
    with pytest.raises(ValueError, match="increasing order"):
        Parameters("min-max", "image", None, True, (0.0, 1.0), (1.0, 1.0), channels=(1, 0))


def test_approximate_parameters_are_reported():
    _, records = cached_design(seed=18, n_channels=1, bin_count=64)
    parameters = fit(records[:4], NormalizationSettings(), group="g")
    assert not parameters.exact
    assert [issue.code for issue in parameters.issues] == ["approximate_parameters"]
    assert parameters.issues[0].items == ("g",)


def test_planes_without_finite_pixels_are_left_out_and_reported():
    good = PlaneRecord("c/r/a.tif", "c", "r", "single", compute_plane_histogram(np.arange(10.0)))
    empty = PlaneRecord(
        "c/r/b.tif", "c", "r", "single", compute_plane_histogram(np.full(4, np.nan))
    )
    parameters = fit([good, empty], NormalizationSettings(method="minmax"))
    assert (parameters.offset, parameters.scale) == ((0.0,), (9.0,))
    codes = [issue.code for issue in parameters.issues]
    assert "planes_without_finite_pixels" in codes
    issue = parameters.issues[codes.index("planes_without_finite_pixels")]
    assert issue.items == (f"{empty.plane_id} (channel 0)",)
    with pytest.raises(NoValidPixelsError, match="Group 'x', channel 0"):
        fit([empty], NormalizationSettings(), group="x")


def test_a_channel_without_finite_pixels_in_one_plane_is_left_out_of_that_channel_only():
    finite = compute_plane_histogram(np.arange(10.0))
    empty = compute_plane_histogram(np.full(10, np.nan))
    planes = [
        PlaneRecord("c/r/a.tif", "c", "r", "single", (finite, empty), t=0),
        PlaneRecord("c/r/a.tif", "c", "r", "single", (finite, finite), t=1),
    ]
    parameters = fit(planes, NormalizationSettings(method="minmax"), group="c/r/a.tif")
    assert (parameters.offset, parameters.scale) == ((0.0, 0.0), (9.0, 9.0))
    issue = {issue.code: issue for issue in parameters.issues}["planes_without_finite_pixels"]
    assert issue.items == ("c/r/a.tif::z=0,t=0 (channel 1)",)
    with pytest.raises(NoValidPixelsError, match="Group 'g', channel 1"):
        fit(planes[:1], NormalizationSettings(), group="g")


def test_fit_refuses_rows_with_different_channel_counts():
    one = compute_plane_histogram(np.arange(10, dtype=np.uint8))
    rows = [
        PlaneRecord("c/r/a.tif", "c", "r", "single", (one, one)),
        PlaneRecord("c/r/b.tif", "c", "r", "single", (one,)),
    ]
    with pytest.raises(ValueError) as error:
        fit(rows, NormalizationSettings(), group="r")
    message = str(error.value)
    assert "group 'r' have different numbers of channels" in message
    assert "1 channel(s): c/r/b.tif; 2 channel(s): c/r/a.tif" in message
    assert "Every image of an experiment must have the same channels" in message


def test_zero_spread_names_the_group_and_channel():
    constant = np.full((2, 4, 4), 7, np.uint16)
    constant[0, 0, 0] = 8
    with pytest.raises(ValueError, match="Group 'p', channel 1: The min-max normalization"):
        fit(
            constant,
            NormalizationSettings(method="minmax", scope="patch"),
            group="p",
            channel_axis=0,
        )


def test_biological_repeat_scope_needs_rows_not_pixels():
    with pytest.raises(MethodNotAllowedError, match="needs the preprocessing brick"):
        fit(np.arange(10, dtype=np.uint8), NormalizationSettings(scope="biological_repeat"))


def test_rgb_waits_for_milestone_3():
    with pytest.raises(MethodNotAllowedError, match="StarDist"):
        fit(np.zeros((3, 4), np.uint8), NormalizationSettings(image_type="rgb", method="zscore"))
    with pytest.raises(NotImplementedError, match="RGB normalization is not available"):
        fit(np.zeros((3, 4), np.uint8), NormalizationSettings(image_type="rgb"), channel_axis=0)


def test_fit_refuses_inputs_it_cannot_read():
    with pytest.raises(ValueError, match="No plane_histograms.csv rows"):
        fit([], NormalizationSettings())
    with pytest.raises(TypeError, match="PlaneRecord"):
        fit([np.zeros(3)], NormalizationSettings())


def test_parameters_from_records_and_existing_parameters_apply_identically():
    _, records = cached_design(seed=19, n_channels=1)
    parameters = fit(records, NormalizationSettings())
    existing = percentile_range(PooledHistogram(records))
    pixels = np.random.default_rng(20).integers(0, 5000, (8, 8)).astype(np.uint16)
    np.testing.assert_array_equal(parameters.apply(pixels), existing.apply(pixels))
    assert isinstance(existing, NormalizationParameters)
