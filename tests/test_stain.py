import subprocess
import sys
import textwrap

import numpy as np
import pytest

from morphobricks_normalization.stain import PRESETS, ScalarResult, StainPreset, rgb_to_scalar

EOSIN = (230, 130, 190)
HEMATOXYLIN = (80, 60, 150)


def random_image(seed=0, shape=(7, 9)):
    return np.random.default_rng(seed).integers(0, 256, (*shape, 3), dtype=np.uint8)


def patches(left, right):
    """A 4x8 YXC image: left colour in the first four columns, right colour in the rest."""
    image = np.empty((4, 8, 3), dtype=np.uint8)
    image[:, :4] = left
    image[:, 4:] = right
    return image


def test_the_module_imports_and_fails_clearly_without_skimage():
    code = textwrap.dedent(
        """
        import sys
        sys.modules["skimage"] = None
        sys.modules["skimage.color"] = None
        import numpy as np
        import morphobricks_normalization
        from morphobricks_normalization.stain import rgb_to_scalar
        try:
            rgb_to_scalar(np.zeros((2, 2, 3), np.uint8), axes="YXC", preset="he_eosin")
        except ImportError as error:
            print(error)
        else:
            raise SystemExit("no ImportError")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "pip install morphobricks-normalization[histology]" in result.stdout


def test_the_registry_has_exactly_the_four_p0_presets():
    assert list(PRESETS) == ["he_eosin", "he_hematoxylin", "luminance_inverted", "lab_a"]
    assert all(
        p.version == 1 and p.polarity == "bright" and p.fixed_range is None
        for p in PRESETS.values()
    )


def test_presets_are_immutable():
    with pytest.raises(TypeError):
        PRESETS["new"] = StainPreset("new")  # type: ignore[index]
    with pytest.raises(AttributeError):
        PRESETS["he_eosin"].version = 2  # type: ignore[misc]


@pytest.mark.parametrize("preset", list(PRESETS))
def test_yxc_and_cyx_give_bitwise_identical_outputs(preset):
    pytest.importorskip("skimage")
    yxc = random_image()
    cyx = np.moveaxis(yxc, 2, 0)
    a = rgb_to_scalar(yxc, axes="YXC", preset=preset)
    b = rgb_to_scalar(cyx, axes="CYX", preset=preset)
    assert a.image.tobytes() == b.image.tobytes()
    if a.exclusion is not None:
        assert a.exclusion.tobytes() == b.exclusion.tobytes()
    assert a.provenance["clipped_fraction"] == b.provenance["clipped_fraction"]


@pytest.mark.parametrize("axes", ["yxc", "XYC", "YXCZ", "", "CXY", None, ("Y", "X", "C")])
def test_wrong_axes_are_refused(axes):
    with pytest.raises(ValueError, match="axes"):
        rgb_to_scalar(random_image(), axes=axes, preset="he_eosin")


def test_axes_are_required():
    with pytest.raises(TypeError):
        rgb_to_scalar(random_image(), preset="he_eosin")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("array", "axes", "message"),
    [
        (np.zeros((5, 5), np.uint8), "YXC", "3-D"),
        (np.zeros((5, 5, 4), np.uint8), "YXC", "alpha"),
        (np.zeros((4, 5, 5), np.uint8), "CYX", "alpha"),
        (np.zeros((5, 5, 2), np.uint8), "YXC", "exactly 3"),
        (np.zeros((5, 5, 3), np.uint8), "CYX", "exactly 3"),
        (np.zeros((2, 5, 5, 3), np.uint8), "YXC", "3-D"),
    ],
)
def test_wrong_shapes_are_refused(array, axes, message):
    with pytest.raises(ValueError, match=message):
        rgb_to_scalar(array, axes=axes, preset="he_eosin")


@pytest.mark.parametrize("dtype", ["float32", "float64", "uint16", "int16", "bool"])
def test_other_dtypes_are_refused(dtype):
    with pytest.raises(TypeError, match="uint8"):
        rgb_to_scalar(np.zeros((4, 4, 3), dtype), axes="YXC", preset="he_eosin")


def test_a_non_array_is_refused():
    with pytest.raises(TypeError, match="NumPy array"):
        rgb_to_scalar([[[0, 0, 0]]], axes="YXC", preset="he_eosin")  # type: ignore[arg-type]


def test_unknown_preset_is_refused_listing_the_known_ones():
    with pytest.raises(ValueError, match="he_eosin.*he_hematoxylin.*luminance_inverted.*lab_a"):
        rgb_to_scalar(random_image(), axes="YXC", preset="trichrome_blue_bpx")


@pytest.mark.parametrize("preset", list(PRESETS))
def test_input_is_unchanged_and_output_is_float32_yx_finite(preset):
    pytest.importorskip("skimage")
    image = random_image(shape=(6, 11))
    before = image.copy()
    result = rgb_to_scalar(image, axes="YXC", preset=preset)
    assert isinstance(result, ScalarResult)
    assert np.array_equal(image, before) and image.dtype == np.uint8
    assert result.image.dtype == np.float32 and result.image.shape == (6, 11)
    assert np.isfinite(result.image).all()
    if result.exclusion is not None:
        assert result.exclusion.dtype == np.float32 and result.exclusion.shape == (6, 11)
        assert np.isfinite(result.exclusion).all()
        assert not np.shares_memory(result.exclusion, image)
    assert not np.shares_memory(result.image, image)


def test_cyx_input_is_unchanged():
    pytest.importorskip("skimage")
    cyx = np.moveaxis(random_image(), 2, 0).copy()
    before = cyx.copy()
    rgb_to_scalar(cyx, axes="CYX", preset="he_eosin")
    assert np.array_equal(cyx, before)


def test_eosin_and_hematoxylin_presets_separate_the_two_stains():
    pytest.importorskip("skimage")
    image = patches(EOSIN, HEMATOXYLIN)  # eosin on the left, haematoxylin on the right
    eosin = rgb_to_scalar(image, axes="YXC", preset="he_eosin").image
    hema = rgb_to_scalar(image, axes="YXC", preset="he_hematoxylin").image
    assert eosin[:, :4].mean() > eosin[:, 4:].mean()
    assert hema[:, 4:].mean() > hema[:, :4].mean()


def test_equal_luminance_colours_are_separated_by_lab_a_not_by_luminance():
    pytest.importorskip("skimage")
    # rgb2gray = 0.2125 R + 0.7154 G + 0.0721 B: both colours give 121.3 / 255 within 0.001.
    red, green = (200, 100, 100), (100, 130, 100)
    image = patches(red, green)
    luminance = rgb_to_scalar(image, axes="YXC", preset="luminance_inverted").image
    lab_a = rgb_to_scalar(image, axes="YXC", preset="lab_a").image
    # 0.005 is about one uint8 grey step (1/255 = 0.0039) of the weighted sum.
    assert luminance.max() - luminance.min() < 0.005
    assert lab_a[:, :4].mean() - lab_a[:, 4:].mean() > 40  # CIELAB a* units, red vs green


def test_lab_a_is_signed_and_not_clipped():
    pytest.importorskip("skimage")
    result = rgb_to_scalar(patches(EOSIN, (60, 160, 60)), axes="YXC", preset="lab_a")
    assert result.image.min() < 0 < result.image.max()
    assert result.provenance["clipped_fraction"] is None


def test_he_eosin_exclusion_is_the_hematoxylin_image_and_others_have_none():
    pytest.importorskip("skimage")
    image = random_image()
    eosin = rgb_to_scalar(image, axes="YXC", preset="he_eosin")
    hema = rgb_to_scalar(image, axes="YXC", preset="he_hematoxylin")
    assert eosin.exclusion is not None
    assert eosin.exclusion.tobytes() == hema.image.tobytes()
    for preset in ("he_hematoxylin", "luminance_inverted", "lab_a"):
        assert rgb_to_scalar(image, axes="YXC", preset=preset).exclusion is None


def test_provenance_contents_and_clipped_fraction():
    skimage = pytest.importorskip("skimage")
    image = patches(EOSIN, HEMATOXYLIN)
    result = rgb_to_scalar(image, axes="YXC", preset="he_eosin")
    expected = {
        "preset": "he_eosin",
        "preset_version": 1,
        "stain": "he",
        "method": "fixed_deconvolution",
        "matrix": "hed_from_rgb",
        "target_component": 1,
        "exclusion_component": 0,
        "polarity": "bright",
        "input_dtype": "uint8",
        "axes": "YXC",
        "colour_space": "srgb",
        "skimage_version": skimage.__version__,
        "clipped_fraction": float(np.mean(result.image == 0)),
    }
    assert dict(result.provenance) == expected
    assert 0.0 <= result.provenance["clipped_fraction"] <= 1.0
    lab = rgb_to_scalar(image, axes="YXC", preset="lab_a").provenance
    assert lab["stain"] is None and lab["matrix"] is None and lab["method"] == "cielab"
    assert (
        rgb_to_scalar(image, axes="YXC", preset="luminance_inverted").provenance["method"]
        == "luminance"
    )


def test_the_clipped_fraction_counts_zeros_of_the_target_component():
    pytest.importorskip("skimage")
    # Left half pure white: every component is 0 there; right half the haematoxylin colour.
    image = patches((255, 255, 255), HEMATOXYLIN)
    result = rgb_to_scalar(image, axes="YXC", preset="he_hematoxylin")
    assert result.provenance["clipped_fraction"] == pytest.approx(0.5)


def test_provenance_is_immutable():
    pytest.importorskip("skimage")
    provenance = rgb_to_scalar(random_image(), axes="YXC", preset="he_eosin").provenance
    with pytest.raises(TypeError):
        provenance["preset"] = "other"  # type: ignore[index]


@pytest.mark.parametrize("preset", list(PRESETS))
def test_the_conversion_is_deterministic(preset):
    pytest.importorskip("skimage")
    image = random_image(3)
    a = rgb_to_scalar(image, axes="YXC", preset=preset)
    b = rgb_to_scalar(image, axes="YXC", preset=preset)
    assert a.image.tobytes() == b.image.tobytes()
    assert dict(a.provenance) == dict(b.provenance)
