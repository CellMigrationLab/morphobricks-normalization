"""The preset catalogue of N68: passthrough, custom_od, polarized PSR, trichrome, hints."""

import math
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from morphobricks_normalization.stain import (
    MASK_HINTS,
    MODALITIES,
    PRESETS,
    PRESETS_BY_MODALITY,
    SCALING_HINTS,
    stain_vectors_from_regions,
    to_scalar,
    white_from_region,
)
from stain_simulation import (
    COUNTERSTAIN,
    DAB,
    EOSIN,
    HEMATOXYLIN,
    PSR,
    WHITE,
    beer_lambert,
    concentration_bound,
    concentrations,
    to_uint8,
    unit_matrix,
)

CUSTOM = {
    "psr_brightfield": ("psr", "counterstain", PSR, COUNTERSTAIN),
    "he_custom": ("eosin", "hematoxylin", EOSIN, HEMATOXYLIN),
    "trichrome_custom": ("collagen", "nuclei", (0.80, 0.59, 0.11), HEMATOXYLIN),
}
SKIMAGE_RGB = ["he_eosin", "he_hematoxylin", "luminance_inverted", "lab_a"]
SKIMAGE_RGB += ["psr_polarized", "trichrome_blue", "trichrome_green"]
RGB = SKIMAGE_RGB + list(CUSTOM)


def arguments(preset):
    """The optional arguments a preset needs, with valid values."""
    if preset in CUSTOM:
        target, exclusion, a, b = CUSTOM[preset]
        return {"white": WHITE, "stain_vectors": {target: a, exclusion: b}}
    if preset == "psr_polarized":
        return {"polarizer": "circular"}
    return {}


def random_rgb(seed=0, shape=(7, 9)):
    return np.random.default_rng(seed).integers(0, 256, (*shape, 3), dtype=np.uint8)


def needs_skimage(preset):
    if preset in SKIMAGE_RGB:
        pytest.importorskip("skimage")


def test_the_registry_holds_the_catalogue_in_order():
    assert list(PRESETS) == [
        "he_eosin",
        "he_hematoxylin",
        "luminance_inverted",
        "lab_a",
        "fluorescence",
        "shg",
        "psr_brightfield",
        "psr_polarized",
        "trichrome_blue",
        "trichrome_green",
        "he_custom",
        "trichrome_custom",
    ]
    assert set(RGB) | {"fluorescence", "shg"} == set(PRESETS)
    with pytest.raises(TypeError):
        PRESETS_BY_MODALITY["shg"] = ()  # type: ignore[index]


def test_presets_by_modality():
    assert dict(PRESETS_BY_MODALITY) == {
        "fluorescence": ("fluorescence",),
        "shg": ("shg",),
        "brightfield": (
            "he_eosin",
            "he_hematoxylin",
            "luminance_inverted",
            "lab_a",
            "psr_brightfield",
            "trichrome_blue",
            "trichrome_green",
            "he_custom",
            "trichrome_custom",
        ),
        "polarized": ("psr_polarized",),
    }


@pytest.mark.parametrize("preset", list(PRESETS))
def test_every_preset_has_valid_read_only_hints(preset):
    recipe = PRESETS[preset]
    assert recipe.modality in MODALITIES and recipe.version == 1 and recipe.fixed_range is None
    assert set(recipe.hints) == {"polarity", "scaling", "mask", "measures"}
    assert recipe.hints["polarity"] == recipe.polarity == "bright"
    assert recipe.hints["scaling"] in SCALING_HINTS and recipe.hints["mask"] in MASK_HINTS
    assert recipe.hints["measures"]
    with pytest.raises(TypeError):
        recipe.hints["scaling"] = "per_image"  # type: ignore[index]


def test_specific_hints_of_the_owner_table():
    hints = {key: (p.hints["scaling"], p.hints["mask"]) for key, p in PRESETS.items()}
    assert hints["fluorescence"] == ("per_image", "optional")
    assert hints["shg"] == ("fixed", "none")
    assert hints["he_eosin"] == ("fixed", "exclude_nuclei_rbc_empty")
    assert hints["psr_brightfield"] == ("fixed", "exclude_counterstain")
    assert hints["psr_polarized"] == ("fixed", "background")
    assert hints["trichrome_blue"] == hints["trichrome_green"] == ("fixed", "exclude_nuclei")
    assert "not collagen-specific" in PRESETS["he_eosin"].hints["measures"]


# --- passthrough ------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["fluorescence", "shg"])
@pytest.mark.parametrize("dtype", ["uint8", "uint16", "int16", "int32", "float16", "float64"])
def test_passthrough_returns_the_same_values_as_float32_copy(preset, dtype):
    plane = (np.arange(30).reshape(5, 6) * 3 - 20).astype(dtype)
    if dtype.startswith("uint"):
        plane = np.arange(30, dtype=dtype).reshape(5, 6)
    before = plane.copy()
    result = to_scalar(plane, axes="YX", preset=preset)
    assert result.image.dtype == np.float32 and result.image.shape == plane.shape
    assert np.array_equal(result.image.astype(np.float64), plane.astype(np.float64))
    assert not np.shares_memory(result.image, plane)
    assert np.array_equal(plane, before)
    assert result.exclusion is None
    provenance = result.provenance
    assert provenance["method"] == "passthrough" and provenance["float32_exact"] is True
    assert provenance["colour_space"] is None and provenance["skimage_version"] is None
    assert provenance["axes"] == "YX" and provenance["input_dtype"] == dtype


def test_passthrough_float32_input_is_copied_not_aliased():
    plane = np.linspace(-1, 1, 12, dtype=np.float32).reshape(3, 4)
    result = to_scalar(plane, axes="YX", preset="fluorescence")
    assert result.image.tobytes() == plane.tobytes()
    assert not np.shares_memory(result.image, plane)


@pytest.mark.parametrize(
    ("plane", "exact"),
    [
        (np.array([[2**24, 2**24 + 1]], np.int32), False),
        (np.array([[2**24, 2**25]], np.int64), True),
        (np.array([[2**63 - 1, 0]], np.int64), False),
        (np.array([[2**64 - 1, 0]], np.uint64), False),
        (np.array([[2**60, -(2**62)]], np.int64), True),
        (np.array([[0.1, 0.5]], np.float64), False),
        (np.array([[0.25, 0.5]], np.float64), True),
    ],
)
def test_passthrough_reports_whether_the_float32_cast_is_exact(plane, exact):
    result = to_scalar(plane, axes="YX", preset="shg")
    assert result.provenance["float32_exact"] is exact


@pytest.mark.parametrize(
    ("plane", "axes", "error", "message"),
    [
        (np.zeros((3, 4)), "YXC", ValueError, "'YX'"),
        (np.zeros((3, 4)), "XY", ValueError, "'YX'"),
        (np.zeros((3, 4, 3)), "YX", ValueError, "2-D"),
        (np.zeros((3, 4), bool), "YX", TypeError, "bool"),
        (np.zeros((3, 4), complex), "YX", TypeError, "complex"),
        (np.array([[np.nan, 1.0]]), "YX", ValueError, "NaN"),
        (np.array([[np.inf, 1.0]]), "YX", ValueError, "NaN or infinite"),
        (np.array([[1e39, 1.0]]), "YX", ValueError, "float32 range"),
        ([[1.0, 2.0]], "YX", TypeError, "NumPy array"),
    ],
)
def test_passthrough_refusals(plane, axes, error, message):
    with pytest.raises(error, match=message):
        to_scalar(plane, axes=axes, preset="fluorescence")


# --- arguments ----------------------------------------------------------------------------

UNUSED = [
    ("lab_a", "white"),
    ("he_eosin", "polarizer"),
    ("he_eosin", "stain_vectors"),
    ("fluorescence", "white"),
    ("shg", "polarizer"),
    ("psr_polarized", "white"),
    ("psr_brightfield", "polarizer"),
    ("trichrome_blue", "stain_vectors"),
]


@pytest.mark.parametrize(("preset", "name"), UNUSED)
def test_unused_arguments_are_refused_not_ignored(preset, name):
    given = arguments(preset)
    given[name] = {"white": WHITE, "stain_vectors": {"a": PSR}, "polarizer": "circular"}[name]
    pixels = np.zeros((3, 4)) if preset in ("fluorescence", "shg") else random_rgb()
    axes = "YX" if preset in ("fluorescence", "shg") else "YXC"
    with pytest.raises(ValueError, match=f"does not use {name}"):
        to_scalar(pixels, axes=axes, preset=preset, **given)


@pytest.mark.parametrize("preset", list(CUSTOM))
@pytest.mark.parametrize("missing", ["white", "stain_vectors"])
def test_custom_presets_need_white_and_vectors(preset, missing):
    given = arguments(preset)
    del given[missing]
    with pytest.raises(ValueError, match=f"needs {missing}"):
        to_scalar(random_rgb(), axes="YXC", preset=preset, **given)


@pytest.mark.parametrize("polarizer", [None, "Circular", "none", "", 1])
def test_psr_polarized_needs_a_declared_polarizer(polarizer):
    with pytest.raises(ValueError, match="polarizer"):
        to_scalar(random_rgb(), axes="YXC", preset="psr_polarized", polarizer=polarizer)


@pytest.mark.parametrize("preset", list(CUSTOM))
def test_custom_presets_refuse_missing_and_extra_names(preset):
    target, exclusion, a, b = CUSTOM[preset]
    with pytest.raises(ValueError, match=f"missing \\['{exclusion}'\\]"):
        to_scalar(random_rgb(), axes="YXC", preset=preset, white=WHITE, stain_vectors={target: a})
    with pytest.raises(ValueError, match="not allowed \\['dab'\\]"):
        vectors = {target: a, exclusion: b, "dab": DAB}
        to_scalar(random_rgb(), axes="YXC", preset=preset, white=WHITE, stain_vectors=vectors)


@pytest.mark.parametrize("white", [(240, 235), (0, 235, 245), (240, 235, 300), (np.nan, 1, 1)])
def test_custom_presets_refuse_an_invalid_white(white):
    vectors = arguments("psr_brightfield")["stain_vectors"]
    with pytest.raises(ValueError, match="white"):
        to_scalar(
            random_rgb(), axes="YXC", preset="psr_brightfield", white=white, stain_vectors=vectors
        )


def test_custom_presets_refuse_collinear_and_ill_conditioned_vectors():
    with pytest.raises(ValueError, match="collinear"):
        to_scalar(
            random_rgb(),
            axes="YXC",
            preset="psr_brightfield",
            white=WHITE,
            stain_vectors={"psr": PSR, "counterstain": np.multiply(PSR, 3)},
        )
    third = np.add(PSR, COUNTERSTAIN) / 2 + (0.003, -0.003, 0.0)
    with pytest.raises(ValueError, match="ill-conditioned"):
        to_scalar(
            random_rgb(),
            axes="YXC",
            preset="psr_brightfield",
            white=WHITE,
            stain_vectors={"psr": PSR, "counterstain": COUNTERSTAIN, "residual": third},
        )


# --- custom_od ----------------------------------------------------------------------------


@pytest.mark.parametrize("preset", list(CUSTOM))
@pytest.mark.parametrize("explicit_third", [False, True])
def test_custom_od_beer_lambert_round_trip(preset, explicit_third):
    target, exclusion, a, b = CUSTOM[preset]
    vectors = {target: a, exclusion: b}
    m = unit_matrix(a, b)
    stains = 2
    if explicit_third:
        vectors["residual"] = DAB
        m = unit_matrix(a, b, DAB)
        stains = 3
    amounts = concentrations(6, (24, 24), stains)
    rgb = to_uint8(beer_lambert(amounts, m))
    cyx = np.moveaxis(rgb, 2, 0).copy()
    result = to_scalar(cyx, axes="CYX", preset=preset, white=WHITE, stain_vectors=vectors)
    bound = concentration_bound(rgb, m)
    assert (np.abs(result.image - amounts[..., 0]) <= bound[..., 0]).all()
    assert (np.abs(result.exclusion - amounts[..., 1]) <= bound[..., 1]).all()
    provenance = result.provenance
    assert provenance["residual_completed"] is (not explicit_third)
    np.testing.assert_allclose(provenance["stain_matrix"], m, rtol=0, atol=1e-15)
    assert provenance["condition_number"] == pytest.approx(np.linalg.cond(m), rel=1e-12)
    assert provenance["stain_names"] == (target, exclusion, "residual")
    assert provenance["white"] == WHITE and provenance["skimage_version"] is None
    assert provenance["floored_fraction"] == provenance["above_white_fraction"] == 0.0


def test_custom_od_keeps_negative_concentrations_and_reports_the_fractions():
    m = unit_matrix(PSR, COUNTERSTAIN)
    image = np.empty((2, 5, 3), np.uint8)
    image[...] = to_uint8(beer_lambert(np.array([0.5, 0.2]), m))  # ordinary stained pixel
    # Target -0.15 with exclusion 0.6: outside the declared stains, kept negative.
    image[0, :2] = to_uint8(beer_lambert(np.array([-0.15, 0.6]), m))
    image[1, 0] = (250, 200, 200)  # brighter than the white in red
    image[1, 1] = (0, 90, 90)  # floored red
    image[1, 2] = (255, 255, 0)  # above white in R and G, floored in B
    vectors = {"psr": PSR, "counterstain": COUNTERSTAIN}
    result = to_scalar(
        image, axes="YXC", preset="psr_brightfield", white=WHITE, stain_vectors=vectors
    )
    provenance = result.provenance
    assert result.image[0, 0] == pytest.approx(-0.15, abs=0.01)
    assert provenance["negative_fraction"] == pytest.approx(np.mean(result.image < 0))
    assert (result.image[0, :2] < 0).all() and provenance["negative_fraction"] >= 0.2
    assert provenance["above_white_fraction"] == pytest.approx(2 / 10)
    assert provenance["floored_fraction"] == pytest.approx(2 / 10)
    assert provenance["zero_fraction"] is None
    assert np.isfinite(result.image).all() and np.isfinite(result.exclusion).all()


def test_custom_od_uses_the_declared_white_not_255():
    # A background of 170 (as in Zenodo 4550300) is stain-free against white 170, not 255.
    image = np.full((3, 3, 3), 170, np.uint8)
    vectors = {"eosin": EOSIN, "hematoxylin": HEMATOXYLIN}
    zero = to_scalar(
        image, axes="YXC", preset="he_custom", white=(170, 170, 170), stain_vectors=vectors
    )
    assert np.array_equal(zero.image, np.zeros((3, 3), np.float32))
    false = to_scalar(
        image, axes="YXC", preset="he_custom", white=(255, 255, 255), stain_vectors=vectors
    )
    assert (false.exclusion > 0.3).all()  # false haematoxylin density in a stain-free area


# --- scikit-image presets -----------------------------------------------------------------


def test_psr_polarized_is_rgb2gray_not_inverted():
    color = pytest.importorskip("skimage.color")
    rgb = random_rgb(7)
    for polarizer, valid in (("circular", True), ("linear", False)):
        result = to_scalar(rgb, axes="YXC", preset="psr_polarized", polarizer=polarizer)
        assert result.image.tobytes() == color.rgb2gray(rgb).astype(np.float32).tobytes()
        assert result.provenance["polarizer"] == polarizer
        assert result.provenance["hints"]["orientation_valid"] is valid
        assert result.provenance["method"] == "luminance" and result.exclusion is None
    bright = to_scalar(
        np.full((2, 2, 3), 250, np.uint8), axes="YXC", preset="psr_polarized", polarizer="circular"
    )
    assert (bright.image > 0.9).all()  # a bright fibre stays bright


def test_trichrome_blue_is_methyl_blue_of_bpx():
    color = pytest.importorskip("skimage.color")
    rgb = random_rgb(8)
    result = to_scalar(rgb, axes="YXC", preset="trichrome_blue")
    expected = color.separate_stains(rgb, color.bpx_from_rgb)[..., 0].astype(np.float32)
    assert result.image.tobytes() == expected.tobytes()
    assert result.exclusion is None
    assert result.provenance["zero_fraction"] == float(np.mean(expected == 0))
    assert result.provenance["matrix"] == "bpx_from_rgb"
    blue = to_scalar(
        np.full((2, 2, 3), (40, 90, 200), np.uint8), axes="YXC", preset="trichrome_blue"
    ).image
    red = to_scalar(
        np.full((2, 2, 3), (200, 60, 90), np.uint8), axes="YXC", preset="trichrome_blue"
    ).image
    assert blue.mean() > red.mean()


def test_trichrome_green_is_minus_lab_a_signed():
    color = pytest.importorskip("skimage.color")
    rgb = random_rgb(9)
    result = to_scalar(rgb, axes="YXC", preset="trichrome_green")
    expected = (-color.rgb2lab(rgb)[..., 1]).astype(np.float32)
    assert result.image.tobytes() == expected.tobytes()
    assert result.image.min() < 0 < result.image.max()
    assert result.provenance["zero_fraction"] is None


# --- shared contract ----------------------------------------------------------------------


@pytest.mark.parametrize("preset", RGB)
def test_yxc_and_cyx_are_bitwise_identical_and_input_is_unchanged(preset):
    needs_skimage(preset)
    yxc = random_rgb(10)
    cyx = np.moveaxis(yxc, 2, 0).copy()
    before = (yxc.copy(), cyx.copy())
    a = to_scalar(yxc, axes="YXC", preset=preset, **arguments(preset))
    b = to_scalar(cyx, axes="CYX", preset=preset, **arguments(preset))
    assert a.image.tobytes() == b.image.tobytes()
    assert (a.exclusion is None) == (b.exclusion is None)
    if a.exclusion is not None:
        assert a.exclusion.tobytes() == b.exclusion.tobytes()
    assert np.array_equal(yxc, before[0]) and np.array_equal(cyx, before[1])
    assert a.image.dtype == np.float32 and a.image.shape == (7, 9)
    assert np.isfinite(a.image).all()
    assert not np.shares_memory(a.image, yxc)
    differing = {"axes"}
    assert {k for k in a.provenance if a.provenance[k] != b.provenance[k]} == differing


@pytest.mark.parametrize("preset", list(PRESETS))
def test_provenance_is_read_only_and_records_the_hints(preset):
    needs_skimage(preset)
    if preset in ("fluorescence", "shg"):
        result = to_scalar(np.ones((2, 3), np.uint16), axes="YX", preset=preset)
    else:
        result = to_scalar(random_rgb(), axes="YXC", preset=preset, **arguments(preset))
    provenance = result.provenance
    assert provenance["preset"] == preset and provenance["modality"] == PRESETS[preset].modality
    for key, value in PRESETS[preset].hints.items():
        assert provenance["hints"][key] == value
    with pytest.raises(TypeError):
        provenance["preset"] = "x"  # type: ignore[index]
    with pytest.raises(TypeError):
        provenance["hints"]["mask"] = "none"  # type: ignore[index]


# --- helpers ------------------------------------------------------------------------------


def test_helpers_give_a_white_and_vectors_that_unmix_the_image():
    m = unit_matrix(PSR, COUNTERSTAIN)
    amounts = np.zeros((30, 30, 2))
    amounts[10:20, :, 0] = 0.4 + concentrations(11, (10, 30), 1, high=0.5)[..., 0]
    amounts[20:, :, 1] = 0.4 + concentrations(12, (10, 30), 1, high=0.5)[..., 0]
    rgb = to_uint8(beer_lambert(amounts, m))
    cyx = np.moveaxis(rgb, 2, 0).copy()
    rows = np.arange(30)[:, None] * np.ones((1, 30), int)
    background, psr, nuclei = rows < 10, (rows >= 10) & (rows < 20), rows >= 20
    white = white_from_region(cyx, axes="CYX", region=background)
    assert white == tuple(float(round(v)) for v in WHITE)
    vectors = stain_vectors_from_regions(
        cyx, axes="CYX", white=white, regions={"psr": psr, "counterstain": nuclei}
    )
    for name, row in (("psr", m[0]), ("counterstain", m[1])):
        angle = math.degrees(math.acos(min(1.0, float(np.dot(vectors[name], row)))))
        assert angle < 0.5, (name, angle)
    result = to_scalar(
        cyx, axes="CYX", preset="psr_brightfield", white=white, stain_vectors=vectors
    )
    assert np.abs(result.image[background]).max() < 0.01
    assert result.image[psr].mean() > 0.5 and np.abs(result.image[nuclei]).mean() < 0.02


def test_helpers_check_their_inputs():
    rgb = random_rgb()
    with pytest.raises(ValueError, match="axes"):
        white_from_region(rgb, axes="YX", region=np.ones((7, 9), bool))
    with pytest.raises(ValueError, match="white"):
        stain_vectors_from_regions(
            rgb, axes="YXC", white=(0, 1, 1), regions={"psr": np.ones((7, 9), bool)}
        )


def test_numpy_only_presets_work_without_scikit_image():
    code = textwrap.dedent(
        """
        import sys
        sys.modules["skimage"] = None
        sys.modules["skimage.color"] = None
        import numpy as np
        from morphobricks_normalization.stain import to_scalar
        plane = to_scalar(np.arange(6.0).reshape(2, 3), axes="YX", preset="fluorescence")
        rgb = np.full((2, 2, 3), 120, np.uint8)
        vectors = {"psr": (0.1, 0.8, 0.55), "counterstain": (0.65, 0.7, 0.29)}
        od = to_scalar(rgb, axes="YXC", preset="psr_brightfield", white=(240, 235, 245),
                       stain_vectors=vectors)
        assert "skimage" not in [m for m in sys.modules if sys.modules[m] is not None]
        for preset, extra in (("trichrome_blue", {}), ("psr_polarized", {"polarizer": "linear"})):
            try:
                to_scalar(rgb, axes="YXC", preset=preset, **extra)
            except ImportError as error:
                assert "[histology]" in str(error)
            else:
                raise SystemExit("no ImportError for " + preset)
        print(plane.image.dtype, od.image.dtype)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["float32", "float32"]


# Review fixes (ChatGPT review of b152472): masked arrays, empty images, fixed reference values.
@pytest.mark.parametrize("axes", ["YXC", "CYX"])
def test_a_masked_rgb_array_is_refused_not_unmasked(axes):
    data = np.zeros((4, 5, 3) if axes == "YXC" else (3, 4, 5), dtype=np.uint8)
    masked = np.ma.MaskedArray(data, mask=np.zeros(data.shape, dtype=bool))
    with pytest.raises(TypeError, match="masked array"):
        to_scalar(masked, axes=axes, preset="luminance_inverted")


def test_a_masked_plane_is_refused():
    masked = np.ma.MaskedArray(np.ones((4, 5), dtype=np.float32), mask=False)
    with pytest.raises(TypeError, match="masked array"):
        to_scalar(masked, axes="YX", preset="fluorescence")


@pytest.mark.parametrize(
    ("shape", "axes"),
    [((0, 5, 3), "YXC"), ((4, 0, 3), "YXC"), ((3, 0, 5), "CYX"), ((3, 4, 0), "CYX")],
)
def test_an_rgb_image_without_pixels_is_refused(shape, axes):
    with pytest.raises(ValueError, match="no pixels"):
        to_scalar(np.zeros(shape, dtype=np.uint8), axes=axes, preset="he_eosin")


@pytest.mark.parametrize("shape", [(0, 5), (4, 0)])
def test_a_plane_without_pixels_is_refused(shape):
    with pytest.raises(ValueError, match="no pixels"):
        to_scalar(np.zeros(shape, dtype=np.float32), axes="YX", preset="shg")


def test_fixed_reference_values_of_the_colour_projections():
    pytest.importorskip("skimage")
    rgb = np.array([[[255, 255, 255], [0, 0, 0], [128, 128, 128]]], dtype=np.uint8)
    lum = to_scalar(rgb, axes="YXC", preset="luminance_inverted").image
    assert lum[0, 0] == pytest.approx(0.0, abs=1e-6)  # white is not dark
    assert lum[0, 1] == pytest.approx(1.0, abs=1e-6)  # black is fully dark
    lab = to_scalar(rgb, axes="YXC", preset="lab_a").image
    # neutral greys have no redness; scikit-image gives |a*| ~ 0.002 from its D65 white point
    assert np.abs(lab).max() < 1e-2
    eosin = to_scalar(rgb, axes="YXC", preset="he_eosin")
    assert eosin.image[0, 0] == 0.0  # white: no stain
    assert eosin.provenance["zero_fraction"] >= 1 / 3
