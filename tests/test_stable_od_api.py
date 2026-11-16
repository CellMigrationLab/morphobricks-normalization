"""The stable optical-density helpers other bricks call (contract §3.6, preprocessing P2).

Pins the import paths, the signatures and one behaviour of each of ``rgb_yxc``,
``check_white`` and ``optical_density`` (which returns a pair), and the exception type of each
refusal. A failure here is a change of public API: it needs the owner's approval, a plan §12
entry and a **Breaking** CHANGELOG entry (contract §3.6), not an update of this test.
"""

import importlib
import inspect
import types

import numpy as np
import pytest

import morphobricks_normalization
from morphobricks_normalization.optical_density import check_white, optical_density
from morphobricks_normalization.scalar_inputs import rgb_yxc

POSITIONAL = inspect.Parameter.POSITIONAL_OR_KEYWORD

# (module, name, [(parameter, kind, annotation)], return annotation). The modules use postponed
# evaluation of annotations, so the annotations are the source text.
STABLE = [
    (
        "morphobricks_normalization.optical_density",
        "optical_density",
        [("rgb", POSITIONAL, "np.ndarray"), ("white", POSITIONAL, "np.ndarray")],
        "tuple[np.ndarray, float]",
    ),
    (
        "morphobricks_normalization.optical_density",
        "check_white",
        [("white", POSITIONAL, "object")],
        "np.ndarray",
    ),
    (
        "morphobricks_normalization.scalar_inputs",
        "rgb_yxc",
        [("pixels", POSITIONAL, "np.ndarray"), ("axes", POSITIONAL, "str")],
        "np.ndarray",
    ),
]


@pytest.mark.parametrize(("module_name", "name", "parameters", "returns"), STABLE)
def test_stable_signatures(module_name, name, parameters, returns):
    function = getattr(importlib.import_module(module_name), name)
    assert callable(function) and function.__module__ == module_name
    signature = inspect.signature(function)
    found = [(p.name, p.kind, p.annotation) for p in signature.parameters.values()]
    assert found == parameters
    assert all(p.default is inspect.Parameter.empty for p in signature.parameters.values())
    assert signature.return_annotation == returns


def test_the_modules_stay_importable_by_their_paths():
    # Not re-exported from the root: a root ``optical_density`` would shadow the module.
    assert isinstance(morphobricks_normalization.optical_density, types.ModuleType)
    assert isinstance(morphobricks_normalization.scalar_inputs, types.ModuleType)
    for name in ("optical_density", "check_white", "rgb_yxc"):
        assert name not in morphobricks_normalization.__all__


def test_rgb_yxc_returns_a_new_yxc_uint8_copy():
    yxc = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    cyx = np.ascontiguousarray(np.moveaxis(yxc, 2, 0))
    before = cyx.copy()
    result = rgb_yxc(cyx, "CYX")
    assert type(result) is np.ndarray and result.dtype == np.uint8 and result.shape == (2, 3, 3)
    assert result.flags.c_contiguous and not np.shares_memory(result, cyx)
    np.testing.assert_array_equal(result, yxc)
    np.testing.assert_array_equal(cyx, before)
    assert not np.shares_memory(rgb_yxc(yxc, "YXC"), yxc)


def test_check_white_returns_a_new_float64_triple():
    given = np.array([255, 128, 1], dtype=np.uint8)
    white = check_white(given)
    assert white.dtype == np.float64 and white.shape == (3,)
    assert white.tolist() == [255.0, 128.0, 1.0] and not np.shares_memory(white, given)
    assert check_white((0.5, 200, 255.0)).tolist() == [0.5, 200.0, 255.0]


def test_optical_density_returns_od_and_floored_fraction():
    rgb = np.array([[[255, 255, 255], [0, 255, 255]], [[25, 255, 255], [255, 255, 255]]], np.uint8)
    white = check_white((255, 255, 255))
    result = optical_density(rgb, white)
    assert type(result) is tuple and len(result) == 2
    od, floored_fraction = result
    assert type(od) is np.ndarray and od.dtype == np.float64 and od.shape == rgb.shape
    assert type(floored_fraction) is float and floored_fraction == 0.25  # one pixel of four
    expected = -np.log10(np.maximum(rgb.astype(np.float64), 1.0) / white)
    np.testing.assert_array_equal(od, expected)
    assert od[0, 1, 0] == -np.log10(1.0 / 255.0)  # 0 is floored at 1, never infinite
    assert np.isfinite(od).all()


RGB = np.zeros((2, 2, 3), np.uint8)


@pytest.mark.parametrize(
    ("pixels", "axes", "error", "start"),
    [
        (RGB.tolist(), "YXC", TypeError, "pixels must be a NumPy array"),
        (RGB, "XYC", ValueError, "axes must be exactly 'YXC' or 'CYX'"),
        (RGB[..., 0], "YXC", ValueError, "Expected a 3-D array"),
        (np.zeros((2, 2, 4), np.uint8), "YXC", ValueError, "Expected exactly 3 samples"),
        (np.ma.MaskedArray(RGB), "YXC", TypeError, "A masked array is refused"),
        (np.zeros((0, 2, 3), np.uint8), "YXC", ValueError, "The image has no pixels"),
        (RGB.astype(np.uint16), "YXC", TypeError, "Only 8-bit sRGB (uint8)"),
    ],
)
def test_rgb_yxc_refusals(pixels, axes, error, start):
    with pytest.raises(error) as raised:
        rgb_yxc(pixels, axes)
    assert type(raised.value) is error and str(raised.value).startswith(start)


@pytest.mark.parametrize(
    ("white", "error", "start"),
    [
        (np.ma.MaskedArray([255.0, 255.0, 255.0]), TypeError, "white is a masked array"),
        ([255, (255, np.ma.masked)], TypeError, "white is a masked array"),
        ([255, 255], ValueError, "white must be three numbers"),
        ((True, True, True), ValueError, "white must be three numbers"),
        (None, ValueError, "white must be three numbers"),
        ((255, 0, 255), ValueError, "Each value of white must be finite and in (0, 255]"),
        ((255, 256, 255), ValueError, "Each value of white must be finite and in (0, 255]"),
        ((255, float("nan"), 255), ValueError, "Each value of white must be finite"),
    ],
)
def test_check_white_refusals(white, error, start):
    with pytest.raises(error) as raised:
        check_white(white)
    assert type(raised.value) is error and str(raised.value).startswith(start)
