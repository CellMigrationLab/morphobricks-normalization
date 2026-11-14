"""check_options: every to_scalar check of the preset and its options, without pixels (#97)."""

import math
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from morphobricks_normalization import stain
from morphobricks_normalization.stain import PRESETS, check_options, to_scalar

WHITE = (240.0, 235.0, 245.0)
PSR_VECTORS = {"psr": (0.10, 0.80, 0.55), "counterstain": (0.65, 0.70, 0.29)}
HE_VECTORS = {"eosin": (0.07, 0.99, 0.11), "hematoxylin": (0.65, 0.70, 0.29)}
COLLINEAR = math.radians(stain.od.MIN_ANGLE_DEGREES - 0.5)
NEAR_PLANE = (1.0, 1.0, 0.02)  # 45 degrees from x and y, but almost in their plane


def psr(**vectors):
    """psr_brightfield options with the given stain vectors."""
    return {"white": WHITE, "stain_vectors": vectors}


# (preset, options, exception, message pattern): every refusal check_options shares with
# to_scalar. The messages are the ones to_scalar gave before #97 (test_messages_are_unchanged).
REFUSALS = {
    "unknown preset": (
        "he_eosine",
        {},
        ValueError,
        r"Unknown preset 'he_eosine'; known presets: he_eosin, ",
    ),
    "white missing (custom_od)": ("he_custom", {}, ValueError, "needs white; it has no default"),
    "vectors missing (psr)": (
        "psr_brightfield",
        {"white": WHITE},
        ValueError,
        "needs stain_vectors",
    ),
    "polarizer missing": ("psr_polarized", {}, ValueError, "needs polarizer"),
    "white unused (he_eosin)": ("he_eosin", {"white": WHITE}, ValueError, "does not use white"),
    "white unused (passthrough)": (
        "fluorescence",
        {"white": WHITE},
        ValueError,
        "does not use white",
    ),
    "polarizer unused (lab_a)": (
        "lab_a",
        {"polarizer": "circular"},
        ValueError,
        "does not use polarizer",
    ),
    "vectors unused (trichrome_blue)": (
        "trichrome_blue",
        {"stain_vectors": PSR_VECTORS},
        ValueError,
        "does not use stain_vectors",
    ),
    "vectors unused (luminance)": (
        "luminance_inverted",
        {"stain_vectors": PSR_VECTORS},
        ValueError,
        "does not use stain_vectors",
    ),
    "polarizer unused (he_custom)": (
        "he_custom",
        {"white": WHITE, "polarizer": "linear"},
        ValueError,
        "does not use polarizer",
    ),
    "unknown polarizer": (
        "psr_polarized",
        {"polarizer": "crossed"},
        ValueError,
        "'circular' or 'linear'",
    ),
    "polarizer wrong case": (
        "psr_polarized",
        {"polarizer": "Circular"},
        ValueError,
        "'circular' or 'linear'",
    ),
    "white zero": ("he_custom", {"white": (0, 200, 200)}, ValueError, r"finite and in \(0, 255\]"),
    "white above 255": (
        "lab_a",
        {"white": (256, 200, 200)},
        ValueError,
        r"finite and in \(0, 255\]",
    ),
    "white negative": (
        "psr_polarized",
        {"white": (-1, 200, 200), "polarizer": "circular"},
        ValueError,
        r"\(0, 255\]",
    ),
    "white nan": (
        "trichrome_custom",
        {"white": (np.nan, 200, 200)},
        ValueError,
        r"finite and in \(0, 255\]",
    ),
    "white inf": (
        "luminance_inverted",
        {"white": (np.inf, 1, 1)},
        ValueError,
        r"finite and in \(0, 255\]",
    ),
    "white two values": (
        "he_custom",
        {"white": (200, 200)},
        ValueError,
        "white must be three numbers",
    ),
    "white text": (
        "he_custom",
        {"white": ("a", "b", "c")},
        ValueError,
        "white must be three numbers",
    ),
    "white scalar": ("trichrome_green", {"white": 240}, ValueError, "white must be three numbers"),
    "white masked array": (
        "he_custom",
        {"white": np.ma.MaskedArray(WHITE, mask=[False, True, False])},
        TypeError,
        "masked",
    ),
    "white all-false mask": (
        "lab_a",
        {"white": np.ma.MaskedArray(WHITE, mask=False)},
        TypeError,
        "white is a masked array",
    ),
    "white list holding masked": (
        "psr_brightfield",
        {"white": [240, np.ma.masked, 245], "stain_vectors": PSR_VECTORS},
        TypeError,
        "white is a masked array",
    ),
    "white tuple holding masked": (
        "luminance_inverted",
        {"white": (np.ma.masked, 235, 245)},
        TypeError,
        "white is a masked array",
    ),
    "vectors not a mapping": (
        "psr_brightfield",
        {"white": WHITE, "stain_vectors": [(1, 0, 0), (0, 1, 0)]},
        TypeError,
        "mapping",
    ),
    "vector name missing": (
        "psr_brightfield",
        psr(psr=(1, 0, 0)),
        ValueError,
        r"missing \['counterstain'\]",
    ),
    "vector name extra": (
        "psr_brightfield",
        psr(**PSR_VECTORS, dab=(0, 0, 1)),
        ValueError,
        r"not allowed \['dab'\]",
    ),
    "vector two values": (
        "psr_brightfield",
        psr(psr=(1, 0), counterstain=(0, 1, 0)),
        ValueError,
        "three numbers",
    ),
    "vector text": (
        "he_custom",
        {"white": WHITE, "stain_vectors": {**HE_VECTORS, "eosin": "red"}},
        ValueError,
        "three numbers",
    ),
    "vector nan": (
        "psr_brightfield",
        psr(psr=(np.nan, 1, 0), counterstain=(0, 1, 0)),
        ValueError,
        "not finite",
    ),
    "vector inf": (
        "trichrome_custom",
        {
            "white": WHITE,
            "stain_vectors": {"collagen": (np.inf, 0, 0), "ponceau_fuchsin": (0, 1, 0)},
        },
        ValueError,
        "not finite",
    ),
    "vector zero length": (
        "psr_brightfield",
        psr(psr=(0, 0, 0), counterstain=(0, 1, 0)),
        ValueError,
        "zero length",
    ),
    "vector masked array": (
        "psr_brightfield",
        psr(psr=np.ma.MaskedArray((1.0, 0, 0), mask=False), counterstain=(0, 1, 0)),
        TypeError,
        "Stain vector 'psr' is a masked array",
    ),
    "vector list holding masked": (
        "he_custom",
        {"white": WHITE, "stain_vectors": {**HE_VECTORS, "hematoxylin": [0.6, np.ma.masked, 0.3]}},
        TypeError,
        "Stain vector 'hematoxylin' is a masked array",
    ),
    "residual masked": (
        "psr_brightfield",
        psr(**PSR_VECTORS, residual=np.ma.MaskedArray((0, 0, 1.0), mask=True)),
        TypeError,
        "Stain vector 'residual' is a masked array",
    ),
    "collinear": (
        "psr_brightfield",
        psr(psr=(1, 0, 0), counterstain=(math.cos(COLLINEAR), math.sin(COLLINEAR), 0)),
        ValueError,
        "4.50 degrees apart",
    ),
    "collinear opposite sign": (
        "psr_brightfield",
        psr(psr=(1, 0, 0), counterstain=(-math.cos(COLLINEAR), -math.sin(COLLINEAR), 0)),
        ValueError,
        "4.50 degrees apart",
    ),
    "collinear identical": (
        "he_custom",
        {"white": WHITE, "stain_vectors": {"eosin": (0, 1, 0), "hematoxylin": (0, 2, 0)}},
        ValueError,
        "collinear",
    ),
    "residual collinear": (
        "psr_brightfield",
        psr(**PSR_VECTORS, residual=(0.20, 1.60, 1.10)),
        ValueError,
        "'psr' and 'residual'",
    ),
    "ill-conditioned": (
        "psr_brightfield",
        psr(psr=(1, 0, 0), counterstain=(0, 1, 0), residual=NEAR_PLANE),
        ValueError,
        "ill-conditioned",
    ),
}
CASES = [pytest.param(*case, id=name) for name, case in REFUSALS.items()]


def valid_pixels(preset):
    """Pixels to_scalar accepts for ``preset``, so only the options can be refused."""
    rng = np.random.default_rng(3)
    if PRESETS[preset].method == "passthrough":
        return rng.normal(100, 10, (5, 6)), "YX"
    return rng.integers(1, 256, (5, 6, 3), dtype=np.uint8), "YXC"


@pytest.mark.parametrize(("preset", "options", "error", "message"), CASES)
def test_check_options_refuses_without_pixels(preset, options, error, message):
    with pytest.raises(error, match=message):
        check_options(preset, **options)


@pytest.mark.parametrize(("preset", "options", "error", "message"), CASES)
def test_to_scalar_refuses_the_same_inputs_with_the_same_message(preset, options, error, message):
    pixels, axes = valid_pixels(preset if preset in PRESETS else "fluorescence")
    with pytest.raises(error) as planned:
        check_options(preset, **options)
    with pytest.raises(error) as converted:
        to_scalar(pixels, axes=axes, preset=preset, **options)
    assert type(converted.value) is type(planned.value)
    assert str(converted.value) == str(planned.value)


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
    ],
)
@pytest.mark.parametrize("preset", ["he_custom", "lab_a"])
def test_a_masked_white_is_refused_before_np_asarray(monkeypatch, preset, white):
    def asarray(*args, **kwargs):
        raise AssertionError("np.asarray was called on a masked white")

    monkeypatch.setattr(np, "asarray", asarray)
    with pytest.raises(TypeError, match="white is a masked array"):
        check_options(preset, white=white)


@pytest.mark.parametrize(
    "vector",
    [np.ma.MaskedArray((0.65, 0.70, 0.29), mask=False), [0.65, np.ma.masked, 0.29]],
)
def test_a_masked_stain_vector_is_refused_before_np_asarray(monkeypatch, vector):
    real_asarray = np.asarray

    def asarray(value, *args, **kwargs):
        if isinstance(value, np.ma.MaskedArray) or (
            isinstance(value, list) and any(v is np.ma.masked for v in value)
        ):
            raise AssertionError("np.asarray was called on a masked stain vector")
        return real_asarray(value, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", asarray)
    vectors = {"psr": (0.10, 0.80, 0.55), "counterstain": vector}
    with pytest.raises(TypeError, match="Stain vector 'counterstain' is a masked array"):
        check_options("psr_brightfield", white=WHITE, stain_vectors=vectors)


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
