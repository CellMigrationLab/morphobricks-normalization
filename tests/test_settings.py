import itertools
import json

import numpy as np
import pytest

from morphobricks_normalization import (
    MethodNotAllowedError,
    NormalizationSettings,
    PooledHistogram,
    __version__,
    check_settings,
    compute_plane_histogram,
    min_max,
    percentile_range,
    robust_z_score,
    z_score,
)
from morphobricks_normalization.normalization import NORMAL_CONSISTENCY
from morphobricks_normalization.settings import FLUORESCENCE_METHODS, POOLING_WEIGHTS, SCOPES

SCOPE_MESSAGE = (
    "Scope '{}' does not exist. Choose 'biological_repeat' (one biological repeat across all "
    "conditions), 'image' or 'patch'."
)
RGB_MESSAGE = (
    "RGB images are normalized with StarDist's percentile method only, which scales the three "
    "colour channels together. Use method 'percentile'."
)
NO_FILE_MESSAGE = (
    "The biological-repeat scope needs the preprocessing brick's plane_histograms.csv, which "
    "was not given. Run the preprocessing brick on this experiment, or use the image or patch "
    "scope."
)


def all_settings():
    for method, scope, pooling, per_time_point in itertools.product(
        FLUORESCENCE_METHODS, SCOPES, POOLING_WEIGHTS, (False, True)
    ):
        yield NormalizationSettings(
            method=method,
            percentile=(0.5, 99.9),
            consistency=1.25,
            scope=scope,
            pooling=pooling,
            per_time_point=per_time_point,
        )
    yield NormalizationSettings(image_type="rgb")


def test_defaults_are_those_of_the_contract():
    settings = NormalizationSettings()
    assert settings.to_dict() == {
        "library_version": __version__,
        "image_type": "fluorescence",
        "method": "percentile",
        "percentile": [1.0, 99.8],
        "scope": "image",
        "per_time_point": False,
    }
    assert settings.consistency == NORMAL_CONSISTENCY == 1.482602218505602
    assert settings.pooling == "pixels"


@pytest.mark.parametrize("settings", list(all_settings()), ids=repr)
def test_to_dict_and_from_dict_round_trip_exactly(settings):
    data = settings.to_dict()
    assert NormalizationSettings.from_dict(data) == settings
    assert NormalizationSettings.from_dict(data).to_dict() == data
    # JSON-compatible: the run configuration is a JSON file.
    assert NormalizationSettings.from_dict(json.loads(json.dumps(data))) == settings


def test_keys_that_do_not_apply_are_omitted_and_their_options_reset():
    zscore = NormalizationSettings(method="zscore", percentile=(5, 95), consistency=2.0)
    assert "percentile" not in zscore.to_dict() and "consistency" not in zscore.to_dict()
    assert zscore == NormalizationSettings(method="zscore")
    robust = NormalizationSettings(method="robust_zscore", consistency=2)
    assert robust.to_dict()["consistency"] == 2.0 and "percentile" not in robust.to_dict()
    image = NormalizationSettings(pooling="conditions")
    assert "pooling" not in image.to_dict() and image.pooling == "pixels"
    repeat = NormalizationSettings(scope="biological_repeat", pooling="conditions")
    assert repeat.to_dict()["pooling"] == "conditions"
    assert NormalizationSettings(percentile=[2, 98]).percentile == (2.0, 98.0)


@pytest.mark.parametrize("scope", ["condition", "technical_repeat", "experiment", "batch", ""])
def test_unknown_and_removed_scopes_are_refused(scope):
    with pytest.raises(ValueError) as error:
        NormalizationSettings(scope=scope)
    assert str(error.value) == SCOPE_MESSAGE.format(scope)
    with pytest.raises(ValueError, match="does not exist"):
        NormalizationSettings.from_dict({"scope": scope})


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"image_type": "brightfield"}, "Image type 'brightfield' does not exist"),
        ({"method": "histogram_matching"}, "Method 'histogram_matching' does not exist"),
        ({"scope": "biological_repeat", "pooling": "images"}, "Pooling 'images' does not exist"),
        ({"percentile": (99, 1)}, r"Expected 0 <= low < high <= 100; got low=99.0, high=1.0."),
        ({"percentile": (1, 101)}, r"Expected 0 <= low < high <= 100"),
        ({"percentile": (1, 50, 99)}, "percentile must be two numbers"),
        ({"percentile": "1-99"}, "percentile must be two numbers"),
        ({"method": "robust_zscore", "consistency": 0}, "consistency must be a positive"),
        ({"method": "robust_zscore", "consistency": float("nan")}, "consistency must be"),
        ({"per_time_point": "yes"}, "per_time_point must be true or false"),
    ],
)
def test_invalid_values_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        NormalizationSettings(**kwargs)


def test_from_dict_refuses_unknown_keys_and_ignores_the_version():
    with pytest.raises(ValueError, match="Unknown normalization setting.*: purpose"):
        NormalizationSettings.from_dict({"purpose": "segmentation"})
    assert NormalizationSettings.from_dict({"library_version": "0.0.1"}) == (
        NormalizationSettings()
    )
    assert NormalizationSettings.from_dict({}) == NormalizationSettings()


@pytest.mark.parametrize("method", ["minmax", "zscore", "robust_zscore"])
def test_rgb_allows_only_the_percentile_method(method):
    with pytest.raises(MethodNotAllowedError) as error:
        NormalizationSettings(image_type="rgb", method=method)
    assert str(error.value) == RGB_MESSAGE
    assert isinstance(error.value, ValueError)
    with pytest.raises(MethodNotAllowedError, match="StarDist's percentile method only"):
        NormalizationSettings.from_dict({"image_type": "rgb", "method": method})
    # check_settings refuses the choice too, for settings that skipped the constructor.
    settings = NormalizationSettings(method=method)
    object.__setattr__(settings, "image_type", "rgb")
    with pytest.raises(MethodNotAllowedError) as error:
        check_settings(settings, statistics_file=True)
    assert str(error.value) == RGB_MESSAGE
    assert check_settings(NormalizationSettings(image_type="rgb"), statistics_file=True) == []


def test_biological_repeat_scope_needs_the_statistics_file():
    settings = NormalizationSettings(scope="biological_repeat")
    with pytest.raises(MethodNotAllowedError) as error:
        check_settings(settings, statistics_file=False)
    assert str(error.value) == NO_FILE_MESSAGE
    issues = check_settings(settings, statistics_file=True)
    assert [issue.code for issue in issues] == ["biological_repeat_scope"]
    assert issues[0].severity == "warning"
    assert "differences between conditions within a repeat are kept" in issues[0].message


@pytest.mark.parametrize("scope", ["image", "patch"])
@pytest.mark.parametrize("statistics_file", [False, True])
def test_image_and_patch_scopes_work_with_or_without_the_file(scope, statistics_file):
    settings = NormalizationSettings(scope=scope)
    assert check_settings(settings, statistics_file=statistics_file) == []


def test_written_method_names_are_those_of_the_methods():
    pooled = PooledHistogram([compute_plane_histogram(np.arange(100, dtype=np.uint8))])
    expected = {
        NormalizationSettings(method="minmax"): min_max(pooled).method,
        NormalizationSettings(method="zscore"): z_score(pooled).method,
        NormalizationSettings(method="robust_zscore"): robust_z_score(pooled).method,
        NormalizationSettings(): percentile_range(pooled).method,
        NormalizationSettings(percentile=(0.5, 99)): percentile_range(pooled, 0.5, 99).method,
    }
    for settings, name in expected.items():
        assert settings.method_name == name
    assert NormalizationSettings().method_name == "percentile 1-99.8"
    assert NormalizationSettings(image_type="rgb").method_name == "rgb percentile 1-99.8"
