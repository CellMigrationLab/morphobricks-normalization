"""check_options: every to_scalar check of the preset and its options, without pixels (#97)."""

import dataclasses
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from morphobricks_normalization import stain
from morphobricks_normalization.stain import PRESETS, check_options, to_scalar
from stain_parity_cases import (
    CHANGED,
    HE_VECTORS,
    NEAR_PLANE,
    PSR_VECTORS,
    REFUSALS,
    WHITE,
    conversion_id,
    conversions,
    frozen_output,
    output_digests,
    valid_pixels,
)

# What to_scalar of v0.6.0 (commit 95e1d6d) did with the inputs of tests/stain_parity_cases.py:
# full exception type and message of every refusal, digests of every conversion. Recorded on that
# commit in a temporary worktree (the command is in the docstring of stain_parity_cases.py), so
# the expected values do not come from the code under test.
V060 = json.loads((Path(__file__).parent / "data" / "stain_v060_parity.json").read_text())
CASES = [pytest.param(*case, id=name) for name, case in REFUSALS.items()]


def object_values(*items):
    """An object array holding ``items`` as they are (``np.array`` would convert masked ones)."""
    if len(items) == 1:
        value = np.empty((), dtype=object)
        value[()] = items[0]
        return value
    values = np.empty(len(items), dtype=object)
    values[:] = items
    return values


def v060_refusal(name):
    """The v0.6.0 refusal of REFUSALS[name] as ``(exception type, full message)``."""
    expected = V060["refusals"][name]
    return {"ValueError": ValueError, "TypeError": TypeError}[expected["type"]], expected["message"]


def test_the_frozen_table_covers_every_case():
    assert len(REFUSALS) == 40
    assert V060["refusals"].keys() == REFUSALS.keys()
    assert V060["changed"].keys() == CHANGED.keys()
    assert len(V060["conversions"]) == 33
    assert all(outcome["type"] != "accepted" for outcome in V060["refusals"].values())


@pytest.mark.parametrize("name", list(REFUSALS))
def test_check_options_refuses_without_pixels_with_the_v060_message(name):
    preset, options = REFUSALS[name]
    error, message = v060_refusal(name)
    with pytest.raises(Exception) as refused:
        check_options(preset, **options)
    assert type(refused.value) is error
    assert str(refused.value) == message


@pytest.mark.parametrize("name", list(REFUSALS))
def test_to_scalar_refuses_with_the_v060_message(name):
    preset, options = REFUSALS[name]
    error, message = v060_refusal(name)
    pixels, axes = valid_pixels(preset)
    with pytest.raises(Exception) as refused:
        to_scalar(pixels, axes=axes, preset=preset, **options)
    assert type(refused.value) is error
    assert str(refused.value) == message


MASKED = "is a masked array (or holds a masked value): its mask would be dropped silently. "
# The inputs whose outcome changed on purpose after v0.6.0 (PR #98 review blockers). B1: a
# polarizer that is not a str is refused, its type named, before the membership test (v0.6.0
# accepted a one-element array and raised NumPy's "truth value is ambiguous" for longer ones; for
# bytes, numbers and lists only the message changes, still a ValueError). B2: masked content inside
# object arrays and nested sequences is refused as masked before np.asarray (v0.6.0 refused these
# later, as not three numbers or as an inhomogeneous sequence). Every other refusal keeps its
# v0.6.0 type and message (above).
NOW = {
    "B1 polarizer one-element array": (
        ValueError,
        "polarizer must be the text 'circular' or 'linear'; got numpy.ndarray "
        "array(['linear'], dtype='<U6').",
    ),
    "B1 polarizer two-element array": (
        ValueError,
        "polarizer must be the text 'circular' or 'linear'; got numpy.ndarray "
        "array(['linear', 'circular'], dtype='<U8').",
    ),
    "B1 polarizer bytes": (
        ValueError,
        "polarizer must be the text 'circular' or 'linear'; got bytes b'linear'.",
    ),
    "B1 polarizer number": (
        ValueError,
        "polarizer must be the text 'circular' or 'linear'; got int 1.",
    ),
    "B1 polarizer list": (
        ValueError,
        "polarizer must be the text 'circular' or 'linear'; got list ['circular'].",
    ),
    "B2 white object array holding masked": (TypeError, f"white {MASKED}Pass plain numbers."),
    "B2 white nested list holding masked": (TypeError, f"white {MASKED}Pass plain numbers."),
    "B2 white list holding a 0-d object array of masked": (
        TypeError,
        f"white {MASKED}Pass plain numbers.",
    ),
    "B2 vector object array holding masked": (
        TypeError,
        f"Stain vector 'eosin' {MASKED}Pass plain numbers.",
    ),
    "B2 vector nested list holding masked": (
        TypeError,
        f"Stain vector 'psr' {MASKED}Pass plain numbers.",
    ),
    "B2 residual tuple of a masked array": (
        TypeError,
        f"Stain vector 'residual' {MASKED}Pass plain numbers.",
    ),
}


def test_every_changed_case_has_its_new_outcome():
    assert NOW.keys() == CHANGED.keys()


@pytest.mark.parametrize("name", list(CHANGED))
def test_the_intentionally_changed_refusals(name):
    preset, options = CHANGED[name]
    error, message = NOW[name]
    before = V060["changed"][name]
    assert (before["type"], before["message"]) != (error.__name__, message)
    pixels, axes = valid_pixels(preset)
    for call in (
        lambda: check_options(preset, **options),
        lambda: to_scalar(pixels, axes=axes, preset=preset, **options),
    ):
        with pytest.raises(Exception) as refused:
            call()
        assert type(refused.value) is error
        assert str(refused.value) == message


def test_a_none_polarizer_is_refused_as_missing_not_by_type():
    with pytest.raises(ValueError) as refused:
        check_options("psr_polarized", polarizer=None)
    assert str(refused.value) == "Preset 'psr_polarized' needs polarizer; it has no default."


@pytest.mark.parametrize("polarizer", ["circular", "linear", np.str_("circular")])
def test_orientation_valid_is_a_python_bool(polarizer):
    pytest.importorskip("skimage")
    result = to_scalar(
        np.full((2, 3, 3), 90, np.uint8), axes="YXC", preset="psr_polarized", polarizer=polarizer
    )
    assert result.provenance["hints"]["orientation_valid"] is (polarizer == "circular")


# Default stain vectors replaced by ones that must be refused: the default path is shown to run
# the same checks as user vectors, with the same exception and message.
BAD_DEFAULTS = {
    "collinear": {"eosin": (0, 1, 0), "hematoxylin": (0, 2, 0)},
    "opposite sign": {"eosin": (0, 1, 0), "hematoxylin": (0, -1, 0.05)},
    "ill-conditioned": {"eosin": (1, 0, 0), "hematoxylin": (0, 1, 0), "residual": NEAR_PLANE},
    "residual collinear": {**HE_VECTORS, "residual": (0.07, 0.99, 0.12)},
    "zero length": {"eosin": (0, 0, 0), "hematoxylin": (0, 1, 0)},
    "not finite": {"eosin": (np.nan, 1, 0), "hematoxylin": (0, 1, 0)},
    "masked": {"eosin": [0.07, np.ma.masked, 0.11], "hematoxylin": (0.65, 0.70, 0.29)},
    "name missing": {"eosin": (0.07, 0.99, 0.11)},
}


def with_he_defaults(monkeypatch, vectors):
    """Replace he_custom's default vectors in a copy of the presets that stain uses."""
    replaced = dataclasses.replace(PRESETS["he_custom"], default_stain_vectors=vectors)
    monkeypatch.setattr(stain, "PRESETS", MappingProxyType({**PRESETS, "he_custom": replaced}))


@pytest.mark.parametrize("vectors", BAD_DEFAULTS.values(), ids=list(BAD_DEFAULTS))
def test_default_vectors_are_refused_as_user_vectors_are(monkeypatch, vectors):
    with pytest.raises(Exception) as as_user:
        check_options("he_custom", white=WHITE, stain_vectors=vectors)
    with_he_defaults(monkeypatch, vectors)
    pixels, axes = valid_pixels("he_custom")
    for call in (
        lambda: check_options("he_custom", white=WHITE),
        lambda: to_scalar(pixels, axes=axes, preset="he_custom", white=WHITE),
    ):
        with pytest.raises(Exception) as as_default:
            call()
        assert type(as_default.value) is type(as_user.value)
        assert str(as_default.value) == str(as_user.value)


def test_the_replaced_default_vectors_are_the_ones_used(monkeypatch):
    vectors = {"eosin": (0.1, 0.9, 0.2), "hematoxylin": (0.7, 0.6, 0.3)}
    with_he_defaults(monkeypatch, vectors)
    pixels, axes = valid_pixels("he_custom")
    result = to_scalar(pixels, axes=axes, preset="he_custom", white=WHITE)
    rows = np.array(result.provenance["stain_matrix"][:2])
    expected = np.array([vectors["eosin"], vectors["hematoxylin"]])
    np.testing.assert_allclose(rows, expected / np.linalg.norm(expected, axis=1)[:, None])
    assert result.provenance["stain_vectors_source"] == stain.PRESET_DEFAULT


CONVERSIONS = [
    pytest.param(case, id=conversion_id(index, case[0], case[2], case[3]))
    for index, case in enumerate(conversions())
]


@pytest.mark.parametrize("case", CONVERSIONS)
def test_outputs_are_those_of_v060(request, case):
    preset, pixels, axes, options = case
    skimage = None
    if PRESETS[preset].method not in ("passthrough", "custom_od"):
        skimage = pytest.importorskip("skimage")
    result = to_scalar(pixels, axes=axes, preset=preset, **options)
    if result.provenance["skimage_version"] is not None:
        # Hashed as a marker (it names the installed scikit-image, not this code).
        assert result.provenance["skimage_version"] == skimage.__version__
    frozen = V060["conversions"][request.node.callspec.id]
    if PRESETS[preset].method != "custom_od":
        assert output_digests(result) == frozen
        return
    # custom_od solves a stain matrix through LAPACK: its last bits depend on the CPU kernel.
    values = frozen_output(preset, result)
    assert values.keys() == frozen.keys()
    assert values["exclusion"] == frozen["exclusion"]
    assert (values["image_dtype"], values["image_shape"]) == (
        frozen["image_dtype"],
        frozen["image_shape"],
    )
    np.testing.assert_allclose(values["image_values"], frozen["image_values"], rtol=RTOL, atol=0)
    assert_close(values["provenance"], frozen["provenance"])


RTOL = 1e-9
"""Relative tolerance of a custom_od value: far above LAPACK's last-bit differences."""


def assert_close(actual: object, expected: object) -> None:
    """Assert that two plain JSON values are equal, floats within :data:`RTOL`."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys()
        for key in expected:
            assert_close(actual[key], expected[key])
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected)
        for item, frozen_item in zip(actual, expected, strict=True):
            assert_close(item, frozen_item)
    elif type(expected) is float:
        assert type(actual) is float
        assert actual == pytest.approx(expected, rel=RTOL, abs=1e-12)
    else:
        assert type(actual) is type(expected) and actual == expected


def test_the_unknown_preset_message_lists_every_known_preset():
    with pytest.raises(ValueError) as refused:
        check_options("nope")
    assert str(refused.value) == f"Unknown preset 'nope'; known presets: {', '.join(PRESETS)}."


# One accepted request per preset family; check_options returns None and needs no pixels.
ACCEPTED = [
    ("fluorescence", {}),
    ("shg", {}),
    ("he_eosin", {}),
    ("he_hematoxylin", {}),
    ("trichrome_blue", {}),
    ("luminance_inverted", {}),
    ("luminance_inverted", {"white": [250, 251, 252.5]}),
    ("lab_a", {"white": np.array([255, 255, 255], dtype=np.uint8)}),
    ("trichrome_green", {}),
    ("psr_polarized", {"polarizer": "circular"}),
    ("psr_polarized", {"polarizer": "linear", "white": WHITE}),
    ("psr_brightfield", {"white": WHITE, "stain_vectors": PSR_VECTORS}),
    (
        "psr_brightfield",
        {"white": WHITE, "stain_vectors": {**PSR_VECTORS, "residual": (0.3, 0.2, 0.9)}},
    ),
    ("he_custom", {"white": WHITE}),
    ("he_custom", {"white": WHITE, "stain_vectors": HE_VECTORS}),
    ("trichrome_custom", {"white": (255, 255, 255)}),
    (
        "trichrome_custom",
        {
            "white": WHITE,
            "stain_vectors": {"collagen": (0.8, 0.59, 0.11), "ponceau_fuchsin": (0.1, 0.74, 0.67)},
        },
    ),
]


@pytest.mark.parametrize(("preset", "options"), ACCEPTED)
def test_check_options_accepts_a_usable_request(preset, options):
    assert check_options(preset, **options) is None


def test_every_preset_has_an_accepted_case():
    assert {preset for preset, _ in ACCEPTED} == set(PRESETS)


@pytest.mark.parametrize(
    "white",
    [
        np.ma.MaskedArray(WHITE, mask=False),
        np.ma.MaskedArray(WHITE, mask=[True, False, False]),
        [240, np.ma.masked, 245],
        (240, 235, np.ma.masked),
        object_values(240, np.ma.masked, 245),
        object_values(240, np.ma.MaskedArray(235.0, mask=False), 245),
        [[240, np.ma.masked, 245]],
        (240, [235, (np.ma.masked,)], 245),
        [240, object_values(np.ma.masked), 245],
        [np.float64(240), object_values(object_values(np.ma.masked)), 245],
    ],
)
@pytest.mark.parametrize("preset", ["he_custom", "lab_a"])
def test_a_masked_white_is_refused_before_np_asarray(monkeypatch, preset, white):
    def asarray(*args, **kwargs):
        raise AssertionError("np.asarray was called on a masked white")

    monkeypatch.setattr(np, "asarray", asarray)
    with pytest.raises(TypeError, match="white is a masked array"):
        check_options(preset, white=white)


def holds_masked(value):
    """Whether ``value`` holds a masked array at any depth (the test's own search)."""
    if isinstance(value, np.ma.MaskedArray):
        return True
    if isinstance(value, np.ndarray) and value.dtype == object:
        return any(holds_masked(item) for item in value.ravel().tolist())
    return isinstance(value, list | tuple) and any(holds_masked(item) for item in value)


@pytest.mark.parametrize(
    "vector",
    [
        np.ma.MaskedArray((0.65, 0.70, 0.29), mask=False),
        [0.65, np.ma.masked, 0.29],
        object_values(0.65, np.ma.masked, 0.29),
        object_values(0.65, np.ma.MaskedArray(0.7, mask=True), 0.29),
        [[0.65, np.ma.masked, 0.29]],
        (0.65, (0.70, [np.ma.masked]), 0.29),
        [0.65, object_values(np.ma.masked), 0.29],
    ],
)
@pytest.mark.parametrize("name", ["counterstain", "residual"])
def test_a_masked_stain_vector_is_refused_before_np_asarray(monkeypatch, name, vector):
    real_asarray = np.asarray

    def asarray(value, *args, **kwargs):
        if holds_masked(value):
            raise AssertionError("np.asarray was called on a masked stain vector")
        return real_asarray(value, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", asarray)
    vectors = {"psr": (0.10, 0.80, 0.55), "counterstain": (0.65, 0.70, 0.29), name: vector}
    with pytest.raises(TypeError, match=f"Stain vector '{name}' is a masked array"):
        check_options("psr_brightfield", white=WHITE, stain_vectors=vectors)


def test_masked_content_is_not_searched_in_strings_or_numeric_arrays():
    # Only lists, tuples and object arrays can hold a masked value; these reach the usual checks.
    with pytest.raises(ValueError, match="white must be three numbers"):
        check_options("lab_a", white="masked")
    with pytest.raises(ValueError, match="white must be three numbers"):
        check_options("lab_a", white=object_values(240, 235.5, np.float32(245)))
    assert check_options("lab_a", white=np.array([240, 235.5, 245], dtype=np.float32)) is None


def test_to_scalar_checks_the_options_before_the_pixels():
    # Neither the pixels nor the axes are usable; the option is refused first.
    with pytest.raises(ValueError, match="does not use white"):
        to_scalar(None, axes="??", preset="he_eosin", white=WHITE)
    with pytest.raises(ValueError, match="'circular' or 'linear'"):
        to_scalar(np.zeros((2, 2, 4), np.uint16), axes="YXC", preset="psr_polarized", polarizer="x")


def test_check_options_leaves_its_arguments_unchanged():
    white = np.array(WHITE)
    vectors = {"psr": np.array((0.1, 0.8, 0.55)), "counterstain": np.array((0.65, 0.7, 0.29))}
    copies = white.copy(), {k: v.copy() for k, v in vectors.items()}
    check_options("psr_brightfield", white=white, stain_vectors=vectors)
    np.testing.assert_array_equal(white, copies[0])
    assert vectors.keys() == copies[1].keys()
    for name, values in vectors.items():
        np.testing.assert_array_equal(values, copies[1][name])


def test_check_options_needs_no_scikit_image():
    code = textwrap.dedent(
        """
        import sys
        sys.modules["skimage"] = None
        sys.modules["skimage.color"] = None
        from morphobricks_normalization.stain import check_options
        check_options("he_eosin")
        check_options("psr_polarized", polarizer="circular", white=(250, 250, 250))
        check_options("lab_a", white=(250, 250, 250))
        try:
            check_options("he_eosin", white=(250, 250, 250))
        except ValueError as error:
            print(error)
        assert "skimage" not in sys.modules or sys.modules["skimage"] is None
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "does not use white" in result.stdout
