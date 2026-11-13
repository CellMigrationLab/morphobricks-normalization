"""Default stain vectors and the optional white of the colour projections (plan N69)."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from morphobricks_normalization import optical_density as od
from morphobricks_normalization.stain import (
    BPX_CITATION,
    BPX_DEFAULT_VECTORS,
    HE_CITATION,
    HE_DEFAULT_VECTORS,
    PRESET_DEFAULT,
    PRESETS,
    USER,
    WHITE_RESCALE,
    to_scalar,
)
from stain_simulation import (
    EOSIN,
    HEMATOXYLIN,
    WHITE,
    beer_lambert,
    concentration_bound,
    concentrations,
    to_uint8,
    unit_matrix,
)

WHITE_PRESETS = ["luminance_inverted", "lab_a", "psr_polarized", "trichrome_green"]
P0_WHITE = ["luminance_inverted", "lab_a"]
DEFAULTS = {
    "he_custom": (HE_DEFAULT_VECTORS, HE_CITATION, "eosin", "hematoxylin"),
    "trichrome_custom": (BPX_DEFAULT_VECTORS, BPX_CITATION, "collagen", "ponceau_fuchsin"),
}


def random_rgb(seed=0, shape=(7, 9)):
    return np.random.default_rng(seed).integers(0, 256, (*shape, 3), dtype=np.uint8)


def extra(preset):
    return {"polarizer": "circular"} if preset == "psr_polarized" else {}


# --- default stain vectors ----------------------------------------------------------------


def test_he_defaults_are_the_hematoxylin_and_eosin_rows_of_rgb_from_hed():
    color = pytest.importorskip("skimage.color")
    assert list(HE_DEFAULT_VECTORS) == ["hematoxylin", "eosin"]
    assert np.array_equal(HE_DEFAULT_VECTORS["hematoxylin"], color.rgb_from_hed[0])
    assert np.array_equal(HE_DEFAULT_VECTORS["eosin"], color.rgb_from_hed[1])


def test_trichrome_defaults_are_the_methyl_blue_and_ponceau_rows_of_rgb_from_bpx():
    color = pytest.importorskip("skimage.color")
    assert list(BPX_DEFAULT_VECTORS) == ["collagen", "ponceau_fuchsin"]
    assert np.array_equal(BPX_DEFAULT_VECTORS["collagen"], color.rgb_from_bpx[0])
    assert np.array_equal(BPX_DEFAULT_VECTORS["ponceau_fuchsin"], color.rgb_from_bpx[1])
    # scikit-image completes the third row by the cross product, as custom_od does.
    assert np.array_equal(
        color.rgb_from_bpx[2], np.cross(color.rgb_from_bpx[0], color.rgb_from_bpx[1])
    )


@pytest.mark.parametrize("preset", list(DEFAULTS))
def test_preset_records_hold_read_only_defaults_matching_their_stain_names(preset):
    vectors, citation, target, exclusion = DEFAULTS[preset]
    recipe = PRESETS[preset]
    assert recipe.default_stain_vectors is vectors
    assert recipe.default_stain_vectors_citation == citation
    assert set(vectors) == {recipe.target_stain, recipe.exclusion_stain} == {target, exclusion}
    assert recipe.needs == {"white"} and recipe.accepts == {"stain_vectors"}
    with pytest.raises(TypeError):
        vectors["eosin"] = (1.0, 0.0, 0.0)  # type: ignore[index]
    assert all(isinstance(v, tuple) for v in vectors.values())


def test_psr_brightfield_has_no_default_and_needs_vectors():
    recipe = PRESETS["psr_brightfield"]
    assert recipe.default_stain_vectors is None and recipe.default_stain_vectors_citation is None
    assert recipe.needs == {"white", "stain_vectors"}
    with pytest.raises(ValueError, match="needs stain_vectors"):
        to_scalar(random_rgb(), axes="YXC", preset="psr_brightfield", white=WHITE)


def test_only_he_custom_and_trichrome_custom_have_defaults():
    with_defaults = [k for k, p in PRESETS.items() if p.default_stain_vectors is not None]
    assert with_defaults == ["he_custom", "trichrome_custom"]


@pytest.mark.parametrize("preset", list(DEFAULTS))
def test_white_is_still_required_with_default_vectors(preset):
    with pytest.raises(ValueError, match="needs white"):
        to_scalar(random_rgb(), axes="YXC", preset=preset)


@pytest.mark.parametrize("preset", list(DEFAULTS))
def test_default_vs_user_vectors_provenance(preset):
    vectors, citation, target, exclusion = DEFAULTS[preset]
    rgb = random_rgb(3)
    default = to_scalar(rgb, axes="YXC", preset=preset, white=WHITE)
    assert default.provenance["stain_vectors_source"] == PRESET_DEFAULT
    assert default.provenance["stain_vectors_citation"] == citation
    np.testing.assert_allclose(
        default.provenance["stain_matrix"],
        unit_matrix(vectors[target], vectors[exclusion]),
        rtol=0,
        atol=1e-15,
    )
    # The same numbers given by the user: same outputs, other source.
    same = to_scalar(rgb, axes="YXC", preset=preset, white=WHITE, stain_vectors=dict(vectors))
    assert same.image.tobytes() == default.image.tobytes()
    assert same.exclusion.tobytes() == default.exclusion.tobytes()
    assert same.provenance["stain_vectors_source"] == USER
    assert same.provenance["stain_vectors_citation"] is None
    # A user override replaces the defaults.
    override = {target: (0.2, 0.9, 0.4), exclusion: (0.7, 0.6, 0.3)}
    user = to_scalar(rgb, axes="YXC", preset=preset, white=WHITE, stain_vectors=override)
    np.testing.assert_allclose(
        user.provenance["stain_matrix"],
        unit_matrix(override[target], override[exclusion]),
        rtol=0,
        atol=1e-15,
    )
    assert user.image.tobytes() != default.image.tobytes()


def test_he_custom_beer_lambert_round_trip_with_the_default_vectors():
    # Rows target (eosin), exclusion (haematoxylin), residual = their normalised cross product.
    m = unit_matrix(EOSIN, HEMATOXYLIN)
    assert dict(HE_DEFAULT_VECTORS) == {"hematoxylin": HEMATOXYLIN, "eosin": EOSIN}
    amounts = concentrations(21, (24, 24), 2)
    rgb = to_uint8(beer_lambert(amounts, m))
    result = to_scalar(rgb, axes="YXC", preset="he_custom", white=WHITE)
    bound = concentration_bound(rgb, m)
    assert (np.abs(result.image - amounts[..., 0]) <= bound[..., 0]).all()
    assert (np.abs(result.exclusion - amounts[..., 1]) <= bound[..., 1]).all()
    assert result.provenance["stain_vectors_source"] == PRESET_DEFAULT
    assert result.provenance["residual_completed"] is True


def test_trichrome_custom_beer_lambert_round_trip_with_the_default_vectors():
    vectors = BPX_DEFAULT_VECTORS
    m = unit_matrix(vectors["collagen"], vectors["ponceau_fuchsin"])
    amounts = concentrations(22, (24, 24), 2)
    rgb = to_uint8(beer_lambert(amounts, m))
    result = to_scalar(rgb, axes="YXC", preset="trichrome_custom", white=WHITE)
    bound = concentration_bound(rgb, m)
    assert (np.abs(result.image - amounts[..., 0]) <= bound[..., 0]).all()
    assert (np.abs(result.exclusion - amounts[..., 1]) <= bound[..., 1]).all()
    assert result.provenance["stain_names"] == ("collagen", "ponceau_fuchsin", "residual")


def test_default_vectors_work_without_scikit_image():
    code = textwrap.dedent(
        """
        import sys
        sys.modules["skimage"] = None
        sys.modules["skimage.color"] = None
        import numpy as np
        from morphobricks_normalization.stain import to_scalar
        rgb = np.full((2, 2, 3), 120, np.uint8)
        for preset in ("he_custom", "trichrome_custom"):
            result = to_scalar(rgb, axes="YXC", preset=preset, white=(240, 235, 245))
            assert result.provenance["stain_vectors_source"] == "preset_default"
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["ok"]


# --- white for the colour projections -----------------------------------------------------


def test_rescale_to_white_rounds_half_to_even_and_clips():
    rgb = np.array([[[1, 3, 170], [200, 201, 255]]], dtype=np.uint8)
    white = np.array([170.0, 170.0, 200.0])
    rescaled, clipped = od.rescale_to_white(rgb, white)
    # 1 * 255 / 170 = 1.5 -> 2; 3 * 255 / 170 = 4.5 -> 4 (half to even); 170 * 255 / 200
    # = 216.75 -> 217; 200 * 255 / 170 = 300 -> 255; 201 * 255 / 170 = 301.5 -> 255;
    # 255 * 255 / 200 = 325.125 -> 255.
    assert rescaled.dtype == np.uint8
    assert rescaled.tolist() == [[[2, 4, 217], [255, 255, 255]]]
    assert clipped == 0.5


def test_rescale_to_white_counts_only_values_rounded_above_255():
    # 255 * 255 / 254.6 = 255.40 rounds to 255: above the white, but not clipped.
    rgb = np.array([[[255, 255, 255], [254, 254, 254]]], dtype=np.uint8)
    white = np.array([254.6, 254.6, 254.6])
    rescaled, clipped = od.rescale_to_white(rgb, white)
    assert rescaled.tolist() == [[[255, 255, 255], [254, 254, 254]]]
    assert clipped == 0.0
    assert od.above_white_fraction(rgb, white) == 0.5


def test_rescale_to_white_255_is_the_identity_for_every_uint8_value():
    values = np.arange(256, dtype=np.uint8)
    rgb = np.stack([values, values[::-1], values], axis=-1)[None]
    rescaled, clipped = od.rescale_to_white(rgb, np.full(3, 255.0))
    assert np.array_equal(rescaled, rgb) and clipped == 0.0


@pytest.mark.parametrize("preset", WHITE_PRESETS)
def test_white_rescaling_equals_the_manual_formula(preset):
    color = pytest.importorskip("skimage.color")
    rgb = random_rgb(4, shape=(11, 13))
    white = (170.0, 200.0, 230.5)
    manual = np.clip(np.rint(rgb.astype(np.float64) * 255 / np.array(white)), 0, 255)
    manual = manual.astype(np.uint8)
    expected = {
        "luminance_inverted": lambda x: 1.0 - color.rgb2gray(x),
        "lab_a": lambda x: color.rgb2lab(x)[..., 1],
        "psr_polarized": color.rgb2gray,
        "trichrome_green": lambda x: -color.rgb2lab(x)[..., 1],
    }[preset](manual)
    result = to_scalar(rgb, axes="YXC", preset=preset, white=white, **extra(preset))
    assert result.image.tobytes() == expected.astype(np.float32).tobytes()
    provenance = result.provenance
    assert provenance["white"] == white and provenance["white_applied"] is True
    assert provenance["white_rescale"] == WHITE_RESCALE
    clipped = (np.rint(rgb.astype(np.float64) * 255 / np.array(white)) > 255).any(axis=-1)
    assert provenance["white_clipped_fraction"] == float(np.mean(clipped)) > 0
    above = (rgb.astype(np.float64) > np.array(white)).any(axis=-1)
    assert provenance["above_white_fraction"] == float(np.mean(above))


@pytest.mark.parametrize("preset", WHITE_PRESETS)
def test_white_255_gives_bitwise_the_outputs_without_white(preset):
    pytest.importorskip("skimage")
    rgb = random_rgb(5, shape=(13, 17))
    without = to_scalar(rgb, axes="YXC", preset=preset, **extra(preset))
    with_white = to_scalar(rgb, axes="YXC", preset=preset, white=(255, 255, 255), **extra(preset))
    assert with_white.image.tobytes() == without.image.tobytes()
    assert with_white.provenance["white_applied"] is True
    assert with_white.provenance["white_clipped_fraction"] == 0.0
    assert with_white.provenance["above_white_fraction"] == 0.0


@pytest.mark.parametrize("preset", WHITE_PRESETS)
def test_without_white_the_provenance_says_so(preset):
    pytest.importorskip("skimage")
    provenance = to_scalar(random_rgb(), axes="YXC", preset=preset, **extra(preset)).provenance
    assert provenance["white"] is None and provenance["white_applied"] is False
    assert provenance["white_rescale"] is None
    assert provenance["white_clipped_fraction"] is None
    assert provenance["above_white_fraction"] is None


@pytest.mark.parametrize("preset", P0_WHITE)
def test_a_background_at_the_declared_white_becomes_neutral(preset):
    pytest.importorskip("skimage")
    # A grey-blue background of (170, 180, 190) reads as dark and bluish against 255, and as
    # white (no darkness, no redness) against its declared white.
    rgb = np.full((3, 4, 3), (170, 180, 190), np.uint8)
    without = to_scalar(rgb, axes="YXC", preset=preset).image
    declared = to_scalar(rgb, axes="YXC", preset=preset, white=(170, 180, 190)).image
    assert np.abs(declared).max() < 1e-2
    assert np.abs(without).min() > 0.2


@pytest.mark.parametrize("preset", WHITE_PRESETS)
@pytest.mark.parametrize("white", [(240, 235), (0, 235, 245), (240, 235, 300), (np.nan, 1, 1)])
def test_an_invalid_white_is_refused_for_the_projections(preset, white):
    with pytest.raises(ValueError, match="white"):
        to_scalar(random_rgb(), axes="YXC", preset=preset, white=white, **extra(preset))


@pytest.mark.parametrize("preset", P0_WHITE)
def test_yxc_and_cyx_with_white_are_bitwise_identical(preset):
    pytest.importorskip("skimage")
    yxc = random_rgb(6)
    cyx = np.moveaxis(yxc, 2, 0).copy()
    before = yxc.copy()
    a = to_scalar(yxc, axes="YXC", preset=preset, white=(200, 210, 220))
    b = to_scalar(cyx, axes="CYX", preset=preset, white=(200, 210, 220))
    assert a.image.tobytes() == b.image.tobytes()
    assert np.array_equal(yxc, before)
    assert {k for k in a.provenance if a.provenance[k] != b.provenance[k]} == {"axes"}


# --- Review of 3a7af3e: masked auxiliary inputs and floored reference regions ---------


def _plain_rgb() -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(60, 230, (8, 8, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    "white",
    [
        np.ma.array([240.0, 235.0, 245.0], mask=[True, False, False]),
        np.ma.array([240.0, 235.0, 245.0], mask=False),  # an all-false mask is refused too
        [240.0, np.ma.masked, 245.0],
    ],
)
def test_a_masked_white_is_refused_through_to_scalar(white):
    with pytest.raises(TypeError, match="white is a masked array"):
        to_scalar(_plain_rgb(), axes="YXC", preset="he_custom", white=white)


def test_a_masked_stain_vector_component_is_refused_through_to_scalar():
    vectors = {
        "eosin": np.ma.array([0.07, 0.99, 0.11], mask=[False, True, False]),
        "hematoxylin": (0.65, 0.70, 0.29),
    }
    with pytest.raises(TypeError, match="Stain vector 'eosin' is a masked array"):
        to_scalar(_plain_rgb(), axes="YXC", preset="he_custom", white=WHITE, stain_vectors=vectors)


def test_masked_regions_are_refused_by_both_region_helpers():
    from morphobricks_normalization.stain import stain_vectors_from_regions, white_from_region

    region = np.zeros((8, 8), bool)
    region[:2, :2] = True
    masked = np.ma.array(region, mask=np.zeros((8, 8), bool))
    masked.mask[0, 0] = True  # one selected pixel marked invalid, three valid ones
    with pytest.raises(TypeError, match="masked array"):
        white_from_region(_plain_rgb(), axes="YXC", region=masked)
    with pytest.raises(TypeError, match="masked array"):
        stain_vectors_from_regions(_plain_rgb(), axes="YXC", white=WHITE, regions={"eosin": masked})


def test_a_reference_region_with_a_floored_pixel_is_refused_and_zeros_outside_do_not_matter():
    from morphobricks_normalization.stain import stain_vectors_from_regions

    rgb = _plain_rgb()
    region = np.zeros((8, 8), bool)
    region[2:5, 2:5] = True
    rgb[7, 7, 1] = 0  # outside the region: allowed
    vectors = stain_vectors_from_regions(rgb, axes="YXC", white=WHITE, regions={"eosin": region})
    assert set(vectors) == {"eosin"}
    rgb[3, 3, 0] = 0  # inside the region: its OD would be taken at the floor
    with pytest.raises(ValueError, match="holds 1 pixel"):
        stain_vectors_from_regions(rgb, axes="YXC", white=WHITE, regions={"eosin": region})
