import numpy as np
import pytest

from morphobricks_normalization import (
    NormalizationSettings,
    PlaneRecord,
    PooledHistogram,
    WeightedHistogram,
    compute_plane_histogram,
    fit,
    group_name,
    scope_groups,
)
from morphobricks_normalization.normalization import NORMAL_CONSISTENCY
from morphobricks_normalization.settings import FLUORESCENCE_METHODS
from simulation import cached_design


def repeat_settings(method, pooling):
    return NormalizationSettings(method=method, scope="biological_repeat", pooling=pooling)


def plane(condition, pixels, repeat="r1", name="img.tif"):
    histogram = compute_plane_histogram(np.asarray(pixels))
    return PlaneRecord(f"{condition}/{repeat}/{name}", condition, repeat, "single", histogram)


def weighted_direct(pixels_by_condition, method, percentile=(1.0, 99.8)):
    """Contract §4.3 on raw pixels: each pixel of a condition weighs 1 / its pixel count."""
    values = np.concatenate([np.asarray(p, np.float64).ravel() for p in pixels_by_condition])
    weights = np.concatenate([np.full(np.size(p), 1 / np.size(p)) for p in pixels_by_condition])

    def quantile(x, w, q):
        order = np.argsort(x, kind="stable")
        cumulative = np.cumsum(w[order])
        index = np.searchsorted(cumulative, q / 100 * cumulative[-1] * (1 - 1e-12))
        return x[order][index]

    if method == "minmax":
        return values.min(), values.max() - values.min()
    if method == "percentile":
        low, high = (quantile(values, weights, q) for q in percentile)
        return low, high - low
    if method == "zscore":
        mean = np.average(values, weights=weights)
        return mean, np.sqrt(np.average((values - mean) ** 2, weights=weights))
    median = quantile(values, weights, 50)
    return median, quantile(np.abs(values - median), weights, 50) * NORMAL_CONSISTENCY


def test_conditions_weighting_gives_each_condition_equal_weight():
    # 1000 dim control pixels against 10 bright treated pixels.
    records = [
        plane("control", np.full(1000, 10, np.uint16)),
        plane("treated", np.full(10, 20, np.uint16)),
    ]
    by_pixels = fit(records, repeat_settings("zscore", "pixels"))
    by_conditions = fit(records, repeat_settings("zscore", "conditions"))
    assert by_pixels.offset[0] == pytest.approx((1000 * 10 + 10 * 20) / 1010)
    assert by_conditions.offset[0] == 15.0 and by_conditions.scale[0] == 5.0
    high = NormalizationSettings(
        scope="biological_repeat", pooling="conditions", percentile=(1, 75)
    )
    assert fit(records, high).offset[0] + fit(records, high).scale[0] == 20.0
    assert fit(records, repeat_settings("minmax", "conditions")) == fit(
        records, repeat_settings("minmax", "pixels")
    )


@pytest.mark.parametrize("method", FLUORESCENCE_METHODS)
def test_conditions_weighting_matches_a_direct_weighted_implementation(method):
    rng = np.random.default_rng(30)
    sizes = {"control": (40, 40), "treated": (12, 9), "drug": (25, 31)}
    pixels = {
        condition: rng.normal(1000 * (k + 1), 150 * (k + 1), size).round().astype(np.uint16)
        for k, (condition, size) in enumerate(sizes.items())
    }
    records = [plane(condition, p) for condition, p in pixels.items()]
    parameters = fit(records, repeat_settings(method, "conditions"))
    offset, scale = weighted_direct(list(pixels.values()), method)
    assert parameters.offset[0] == pytest.approx(offset, rel=1e-12)
    assert parameters.scale[0] == pytest.approx(scale, rel=1e-12)
    assert parameters.exact


@pytest.mark.parametrize("method", FLUORESCENCE_METHODS)
@pytest.mark.parametrize("dtype, bins", [(np.uint16, None), (np.float32, 400)])
def test_both_weights_agree_when_conditions_have_equal_pixel_counts(method, dtype, bins):
    # simulate_design gives every condition of a repeat the same images and sizes.
    _, records = cached_design(seed=31, dtype=dtype, bin_count=bins)
    by_pixels = repeat_settings(method, "pixels")
    by_conditions = repeat_settings(method, "conditions")
    groups, _ = scope_groups(records, by_pixels)
    for key, members in groups.items():
        group = group_name(key)
        assert fit(members, by_conditions, group=group) == fit(members, by_pixels, group=group)


@pytest.mark.parametrize("method", FLUORESCENCE_METHODS)
def test_both_weights_differ_when_conditions_have_unequal_pixel_counts(method):
    _, records = cached_design(seed=32, n_channels=1)
    records = [r for r in records if not (r.condition == "treated" and "tech_rep2" in r.plane_id)]
    groups, _ = scope_groups(records, repeat_settings(method, "pixels"))
    members = next(iter(groups.values()))
    by_pixels = fit(members, repeat_settings(method, "pixels"))
    by_conditions = fit(members, repeat_settings(method, "conditions"))
    if method == "minmax":  # extrema do not depend on the weights
        assert by_pixels == by_conditions
    else:
        assert by_pixels != by_conditions


@pytest.mark.parametrize("method", FLUORESCENCE_METHODS)
@pytest.mark.parametrize("scope", ["image", "patch"])
def test_both_weights_agree_at_the_image_and_patch_scopes(method, scope):
    planes, all_records = cached_design(seed=33, n_times=2)
    path = planes[0][1]
    stack = np.stack([p for _, q, c, t, p in planes if q == path]).reshape(2, 2, 32, 32)
    by_pixels = NormalizationSettings(method=method, scope=scope, pooling="pixels")
    by_conditions = NormalizationSettings(method=method, scope=scope, pooling="conditions")
    assert by_pixels == by_conditions  # the pooling weight has no effect at these scopes
    assert fit(stack, by_conditions, channel_axis=0) == fit(stack, by_pixels, channel_axis=0)
    records = [r for r in all_records if r.source_relative_path == path]
    if scope == "image":
        assert fit(records, by_conditions) == fit(records, by_pixels)


def test_unit_weights_give_the_pooled_histogram_exactly():
    _, records = cached_design(seed=34, n_channels=1, bin_count=1000)
    pooled = PooledHistogram(records)
    weighted = WeightedHistogram([pooled], [1.0])
    split = WeightedHistogram(
        [PooledHistogram(records[:5]), PooledHistogram(records[5:])], [1.0, 1.0]
    )
    for other in (weighted, split):
        assert (other.mean, other.std, other.median, other.mad) == (
            pooled.mean,
            pooled.std,
            pooled.median,
            pooled.mad,
        )
        assert [other.percentile(q) for q in (0, 0.1, 1, 37.5, 99.8, 100)] == [
            pooled.percentile(q) for q in (0, 0.1, 1, 37.5, 99.8, 100)
        ]
        assert (other.minimum, other.maximum) == (pooled.minimum, pooled.maximum)
        assert other.max_representative_error == pooled.max_representative_error


def test_weighted_histogram_refuses_invalid_weights():
    pooled = PooledHistogram([compute_plane_histogram(np.arange(5, dtype=np.uint8))])
    with pytest.raises(ValueError, match="positive and finite"):
        WeightedHistogram([pooled], [0.0])
    with pytest.raises(ValueError, match="one weight"):
        WeightedHistogram([pooled], [1.0, 2.0])


def test_a_condition_without_finite_pixels_is_left_out_of_the_weights():
    records = [
        plane("control", np.arange(10.0)),
        plane("treated", np.full(4, np.nan)),
    ]
    parameters = fit(records, repeat_settings("minmax", "conditions"))
    assert (parameters.offset, parameters.scale) == ((0.0,), (9.0,))
    assert "planes_without_finite_pixels" in [i.code for i in parameters.issues]
