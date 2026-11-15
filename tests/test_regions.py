"""The one region checker, region estimates and the resolver-version refusal (Spec ADR 0005).

Fixtures F04, F05 and F06 (normalization part) of MorphoBricks-Spec ADR 0005, the exact
``MASK_REGION_INVALID`` and ``RESOLVER_VERSION_UNSUPPORTED`` messages, the public surface of the
two new modules, and the parity of every non-empty region result with v0.6.1 (``b8aa65b``).
"""

import ast
import json
import math
import warnings
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from morphobricks_normalization import contracts, regions, stain
from morphobricks_normalization import optical_density as od
from morphobricks_normalization.contracts import (
    RESOLVER_VERSION_UNSUPPORTED,
    ResolverVersionError,
    refuse_resolver_version,
)
from morphobricks_normalization.regions import (
    EMPTY_REGION,
    MASK_REGION_INVALID,
    REGION_REASONS,
    RegionEstimate,
    RegionMaskError,
    check_region,
)
from region_parity_cases import outputs, vector_cases, white_cases

ROOT = Path(__file__).resolve().parents[1]
WHITE = (240.0, 235.0, 245.0)
TAIL = (
    ". A region is a 2-D integer or boolean array of the plane's shape (Y, X); its non-zero pixels "
    "are inside."
)
# What v0.6.1 (commit b8aa65b) returned for the non-empty regions of tests/region_parity_cases.py,
# recorded on that commit in a temporary worktree (command in that module's docstring), so the
# expected values do not come from the code under test.
V061 = json.loads((Path(__file__).parent / "data" / "region_v061_parity.json").read_text())


def blob() -> np.ndarray:
    """F04's pixel set: a 3 x 3 block and a separate pixel of an 8 x 8 plane."""
    region = np.zeros((8, 8), bool)
    region[1:4, 1:4] = True
    region[6, 5] = True
    return region


def rgb8() -> np.ndarray:
    """An (8, 8, 3) uint8 image without zeros, below every white of the tests."""
    rng = np.random.default_rng(5)
    return rng.integers(20, 200, (8, 8, 3)).astype(np.uint8)


# --- F05: the region checker's refusals, exact messages ---------------------------------------


@pytest.mark.parametrize(
    ("mask", "reason", "detail"),
    [
        (np.ones((8, 8), np.float32), "dtype", "has pixel type float32"),
        (np.ones((1, 8, 8), np.uint8), "not_2d", "has 3 dimensions"),
        (np.ones((8, 9), np.uint8), "shape", "has shape (8, 9), not the plane's (8, 8)"),
        (np.ma.array(np.ones((8, 8), np.uint8)), "masked_array", "is a masked array"),
        ([[0, 1]], "not_an_array", "is not a NumPy array (got list)"),
    ],
    ids=["F05-float32", "F05-stack", "F05-shape", "F05-masked", "F05-list"],
)
def test_f05_refusals_have_their_reason_and_exact_message(mask, reason, detail):
    with pytest.raises(RegionMaskError) as caught:
        check_region(mask, (8, 8), name="TumorRegion")
    error = caught.value
    assert isinstance(error, ValueError)
    assert error.code == "MASK_REGION_INVALID" == MASK_REGION_INVALID
    assert error.reason == reason
    assert str(error) == f"MASK_REGION_INVALID ({reason}): the region 'TumorRegion' {detail}" + TAIL


@pytest.mark.parametrize(
    ("mask", "reason", "detail"),
    [
        (None, "not_an_array", "is not a NumPy array (got NoneType)"),
        ((1, 0), "not_an_array", "is not a NumPy array (got tuple)"),
        (np.uint8(1), "not_an_array", "is not a NumPy array (got uint8)"),
        (np.ones(8, bool), "not_2d", "has 1 dimensions"),
        (np.ones((2, 8, 8), bool), "not_2d", "has 3 dimensions"),
        (np.ones((8, 8, 1), bool), "not_2d", "has 3 dimensions"),
        (np.ones((8, 8), bool).T[:4], "shape", "has shape (4, 8), not the plane's (8, 8)"),
        (np.ones((8, 8), np.float64), "dtype", "has pixel type float64"),
        (np.ones((8, 8), np.float16), "dtype", "has pixel type float16"),
        (np.ones((8, 8), np.complex64), "dtype", "has pixel type complex64"),
        (np.ones((8, 8), object), "dtype", "has pixel type object"),
        (np.full((8, 8), "a"), "dtype", "has pixel type <U1"),
        (np.ones((8, 8), "datetime64[s]"), "dtype", "has pixel type datetime64[s]"),
        (np.ma.array(np.ones((1, 8, 8), np.float32)), "masked_array", "is a masked array"),
    ],
)
def test_every_other_refusal(mask, reason, detail):
    with pytest.raises(RegionMaskError) as caught:
        check_region(mask, (8, 8), name="r")
    assert caught.value.reason == reason
    assert str(caught.value) == f"MASK_REGION_INVALID ({reason}): the region 'r' {detail}" + TAIL


def test_reasons_are_tested_in_the_documented_order():
    assert regions.INVALID_REASONS == ("not_an_array", "masked_array", "not_2d", "shape", "dtype")
    # A float stack of the wrong shape is refused for its dimensions, before shape or dtype.
    with pytest.raises(RegionMaskError) as caught:
        check_region(np.ones((1, 8, 9), np.float32), (8, 8), name="r")
    assert caught.value.reason == "not_2d"
    with pytest.raises(RegionMaskError) as caught:
        check_region(np.ones((8, 9), np.float32), (8, 8), name="r")
    assert caught.value.reason == "shape"


@pytest.mark.parametrize("shape", [(8,), (8, 8, 1), "88", None, (8.0, 8), (True, 8), (-1, 8)])
def test_a_malformed_plane_shape_is_a_programming_error_without_a_code(shape):
    with pytest.raises(ValueError, match="shape must be the plane's") as caught:
        check_region(np.ones((8, 8), bool), shape, name="r")
    assert not isinstance(caught.value, RegionMaskError)


def test_numpy_integer_plane_sizes_are_accepted():
    assert check_region(np.ones((8, 8), bool), (np.int64(8), np.uint16(8)), name="r").all()


# --- F04: a region is mask != 0, whatever its dtype and representation ------------------------


def f04_masks():
    """The same pixel set as bool, uint8 0/255 and int32 instances with labels 1 and 7."""
    region = blob()
    as_uint8 = np.where(region, 255, 0).astype(np.uint8)
    instances = np.zeros((8, 8), np.int32)
    instances[1:4, 1:4] = 1
    instances[6, 5] = 7
    return [region, as_uint8, instances]


def test_f04_three_representations_give_one_boolean_region():
    masks = f04_masks()
    before = [m.copy() for m in masks]
    results = [check_region(m, (8, 8), name="TumorRegion") for m in masks]
    for result, mask, original in zip(results, masks, before, strict=True):
        assert type(result) is np.ndarray and result.dtype == np.bool_
        assert np.array_equal(result, blob())
        assert result is not mask and not np.shares_memory(result, mask)
        assert np.array_equal(mask, original) and mask.dtype == original.dtype  # unchanged


def test_f04_a_region_value_is_bit_identical_for_each_representation():
    image = rgb8()
    whites = [od.region_white(image, m) for m in f04_masks()]
    assert whites[0].value is not None
    assert all(w == whites[0] for w in whites)
    white = np.array(WHITE)
    vectors = [od.region_vectors(image, white, {"stain": m})["stain"] for m in f04_masks()]
    hexes = {tuple(v.hex() for v in estimate.value) for estimate in vectors}
    assert len(hexes) == 1


@pytest.mark.parametrize(
    "dtype",
    [np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64],
)
def test_every_integer_and_boolean_dtype_is_a_region(dtype):
    mask = np.zeros((3, 4), dtype)
    mask[1, 2] = 1
    assert np.array_equal(check_region(mask, (3, 4), name="r"), mask != 0)


def test_negative_values_are_inside_and_nothing_is_labelled_or_cast():
    mask = np.array([[0, -1, 3], [7, 0, -128]], np.int8)
    result = check_region(mask, (2, 3), name="r")
    assert result.tolist() == [[False, True, True], [True, False, True]]


def test_an_empty_or_zero_size_region_is_returned_not_refused():
    assert not check_region(np.zeros((8, 8), np.uint8), (8, 8), name="r").any()
    assert check_region(np.zeros((0, 8), bool), (0, 8), name="r").shape == (0, 8)


def test_an_ndarray_subclass_gives_a_plain_boolean_array(tmp_path):
    mapped = np.memmap(tmp_path / "region.dat", dtype=np.uint8, mode="w+", shape=(4, 4))
    mapped[1, 1] = 9
    result = check_region(mapped, (4, 4), name="r")
    assert type(result) is np.ndarray and result.sum() == 1
    del mapped


def test_a_non_contiguous_view_is_read_as_is():
    base = np.zeros((8, 16), np.uint8)
    base[2, 4] = 1
    view = base[:, ::2]
    assert np.argwhere(check_region(view, (8, 8), name="r")).tolist() == [[2, 2]]


# --- RegionEstimate ---------------------------------------------------------------------------


def test_region_estimate_holds_a_value_or_a_reason():
    assert REGION_REASONS == ("empty_region",) and EMPTY_REGION == "empty_region"
    defined = RegionEstimate((1.0, 2.0, 3.0), None)
    assert defined.value == (1.0, 2.0, 3.0) and defined.reason is None
    undefined = RegionEstimate(None, "empty_region")
    assert undefined.value is None and undefined.reason == EMPTY_REGION
    with pytest.raises(AttributeError):
        defined.value = None  # frozen


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (None, None),
        ((1.0,), "empty_region"),
        (None, "sampling_mask_empty"),
        (None, ""),
        ((math.nan, 1.0, 1.0), None),
        ((math.inf, 1.0, 1.0), None),
        ((), None),
        ([1.0, 2.0, 3.0], None),
        (np.array([1.0, 2.0, 3.0]), None),
        ((np.float64(1.0), 2.0, 3.0), None),
        ((1, 2.0, 3.0), None),
        ((True, 2.0, 3.0), None),
    ],
)
def test_region_estimate_refuses_a_broken_invariant(value, reason):
    with pytest.raises(ValueError, match="RegionEstimate"):
        RegionEstimate(value, reason)


# --- F06 (normalization part): an empty region is undefined, never NaN, never a crash ---------


def f06_image_and_regions():
    """An (8, 8, 3) uint8 image, an all-zero region and a non-empty one."""
    return rgb8(), np.zeros((8, 8), np.uint8), blob()


def test_f06_region_white_of_an_empty_region_is_undefined_without_a_warning():
    image, empty, _ = f06_image_and_regions()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            assert od.region_white(image, empty) == RegionEstimate(None, "empty_region")


def test_f06_region_vectors_keep_the_non_empty_value_and_mark_the_empty_one():
    image, empty, full = f06_image_and_regions()
    white = np.array(WHITE)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            result = od.region_vectors(image, white, {"empty": empty, "full": full})
    assert isinstance(result, MappingProxyType) and list(result) == ["empty", "full"]
    assert result["empty"] == RegionEstimate(None, "empty_region")
    expected = v061_region_vectors(image, white, {"full": full})["full"]
    assert [v.hex() for v in result["full"].value] == [v.hex() for v in expected]
    assert result["full"].reason is None


def test_f06_region_white_of_the_non_empty_region_is_v061s_value():
    image, _, full = f06_image_and_regions()
    expected = v061_region_white(image, full)
    assert [v.hex() for v in od.region_white(image, full).value] == [v.hex() for v in expected]


def test_f06_white_from_region_refuses_an_empty_region_with_the_exact_message():
    image, empty, _ = f06_image_and_regions()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError) as caught:
            stain.white_from_region(image, axes="YXC", region=empty)
    assert type(caught.value) is ValueError
    assert str(caught.value) == (
        "The background region is empty (empty_region): no white can be estimated from it, so "
        "nothing is converted. Choose a region with at least one pixel, or declare the white "
        "explicitly."
    )


@pytest.mark.parametrize(
    ("names", "shown"),
    [(["eosin"], "['eosin']"), (["eosin", "hematoxylin"], "['eosin', 'hematoxylin']")],
)
def test_f06_stain_vectors_from_regions_name_every_empty_region_in_mapping_order(names, shown):
    image, empty, full = f06_image_and_regions()
    given = {"first_full": full} | {name: empty for name in names}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError) as caught:
            stain.stain_vectors_from_regions(image, axes="YXC", white=WHITE, regions=given)
    assert type(caught.value) is ValueError
    assert str(caught.value) == (
        f"The regions {shown} are empty (empty_region): no stain vector can be estimated from "
        "them, so nothing is converted. Label at least one pixel of each stain."
    )


def test_the_wrappers_keep_their_return_types():
    image, _, full = f06_image_and_regions()
    white = stain.white_from_region(image, axes="YXC", region=full)
    assert type(white) is tuple and len(white) == 3 and all(type(v) is float for v in white)
    vectors = stain.stain_vectors_from_regions(
        image, axes="YXC", white=WHITE, regions={"eosin": full, "hematoxylin": ~full}
    )
    assert isinstance(vectors, MappingProxyType) and list(vectors) == ["eosin", "hematoxylin"]
    for vector in vectors.values():
        assert type(vector) is tuple and len(vector) == 3
        assert all(type(v) is float for v in vector)


def test_the_wrappers_accept_integer_regions_and_refuse_bad_ones_with_the_checker():
    image, _, full = f06_image_and_regions()
    labels = full.astype(np.int32) * 7
    assert stain.white_from_region(image, axes="YXC", region=labels) == stain.white_from_region(
        image, axes="YXC", region=full
    )
    with pytest.raises(RegionMaskError) as caught:
        stain.white_from_region(image, axes="YXC", region=full.astype(np.float32))
    assert caught.value.reason == "dtype" and "'background'" in str(caught.value)
    with pytest.raises(RegionMaskError) as caught:
        stain.stain_vectors_from_regions(
            image, axes="YXC", white=WHITE, regions={"eosin": np.ones((8, 7), bool)}
        )
    assert caught.value.reason == "shape" and "'eosin'" in str(caught.value)


def test_an_empty_or_non_mapping_regions_argument_stays_a_type_error():
    image, _, full = f06_image_and_regions()
    for given in ({}, [("eosin", full)], None):
        with pytest.raises(TypeError, match="non-empty mapping"):
            stain.stain_vectors_from_regions(image, axes="YXC", white=WHITE, regions=given)


def test_the_dark_or_saturated_median_and_the_floor_are_still_refused():
    image, _, full = f06_image_and_regions()
    dark = image.copy()
    dark[..., 0] = 0
    with pytest.raises(ValueError, match="dark or saturated"):
        od.region_white(dark, full)
    with pytest.raises(ValueError, match="holds 10 pixel"):
        od.region_vectors(dark, np.array(WHITE), {"eosin": full})


# --- Parity: non-empty results bitwise equal to v0.6.1 ----------------------------------------
# The v0.6.1 functions, copied unchanged from optical_density.py at b8aa65b (boolean regions only,
# and an empty region raised), give the same-machine bitwise reference.


def _v061_check_region(region, shape, name):
    if not isinstance(region, np.ndarray) or region.dtype != np.bool_:
        raise TypeError(f"Region {name!r} must be a boolean NumPy array (Y, X).")
    if region.shape != shape:
        raise ValueError(f"Region {name!r} has shape {region.shape}; the image is {shape} (Y, X).")
    if not region.any():
        raise ValueError(f"Region {name!r} is empty; select at least one pixel.")
    return region


def v061_region_white(rgb, region):
    mask = _v061_check_region(region, rgb.shape[:2], "background")
    median = np.median(rgb[mask].astype(np.float64), axis=0)
    return tuple(float(v) for v in median)


def v061_region_vectors(rgb, white, regions):
    measured, _ = od.optical_density(rgb, white)
    vectors = {}
    for name, region in regions.items():
        mask = _v061_check_region(region, rgb.shape[:2], name)
        mean = measured[mask].mean(axis=0)
        vectors[name] = tuple(float(v) for v in mean / float(np.linalg.norm(mean)))
    return vectors


def _hexes(values):
    return [float(v).hex() for v in values]


def test_whites_are_bitwise_equal_to_v061_recorded_and_recomputed():
    for case, image, region in white_cases():
        recorded = V061["outputs"][case]
        estimate = od.region_white(image, region)
        assert _hexes(estimate.value) == recorded["region_white"], case
        assert _hexes(estimate.value) == _hexes(v061_region_white(image, region)), case
        cyx = np.ascontiguousarray(np.moveaxis(image, 2, 0))
        assert (
            _hexes(stain.white_from_region(cyx, axes="CYX", region=region))
            == (recorded["white_from_region_cyx"])
        )


def test_vectors_are_bitwise_equal_to_v061_recomputed_and_match_the_recorded_values():
    for case, image, white, given in vector_cases():
        recorded = V061["outputs"][case]
        white_array = np.asarray(white, dtype=np.float64)
        estimates = od.region_vectors(image, white_array, given)
        reference = v061_region_vectors(image, white_array, given)
        wrapped = stain.stain_vectors_from_regions(image, axes="YXC", white=white, regions=given)
        for name, estimate in estimates.items():
            # Bitwise on this machine against the v0.6.1 code.
            assert _hexes(estimate.value) == _hexes(reference[name]), (case, name)
            assert _hexes(wrapped[name]) == _hexes(estimate.value), (case, name)
            # Against the values recorded at b8aa65b: log10 may differ in its last bits between
            # CPUs and NumPy builds (SIMD), so the recorded table is compared to a few ulps.
            frozen = [float.fromhex(v) for v in recorded["region_vectors"][name]]
            np.testing.assert_array_max_ulp(np.array(estimate.value), np.array(frozen), maxulp=4)


def test_the_parity_table_covers_the_recorded_cases_and_every_white_output():
    """Same cases as recorded; every white output (medians: exact on every CPU) bitwise."""
    table = outputs()
    assert table.keys() == V061["outputs"].keys()
    whites = {k: v for k, v in table.items() if k.startswith("white_")}
    assert whites == {k: v for k, v in V061["outputs"].items() if k.startswith("white_")}
    assert len(whites) == 24 and len(table) == 96


# --- RESOLVER_VERSION_UNSUPPORTED -------------------------------------------------------------


@pytest.mark.parametrize("found", [2, 4])
def test_resolver_version_refusal_has_the_exact_message(found):
    error = refuse_resolver_version(found, frozenset({3}))
    assert type(error) is ResolverVersionError and isinstance(error, ValueError)
    assert error.code == "RESOLVER_VERSION_UNSUPPORTED" == RESOLVER_VERSION_UNSUPPORTED
    assert (error.found, error.supported) == (found, (3,))
    assert str(error) == (
        f"RESOLVER_VERSION_UNSUPPORTED: the installed morphobricks-preprocessing resolver has "
        f"contract version {found}; this consumer supports exactly [3]. Install the "
        "morphobricks-preprocessing release it pins; nothing was read or written."
    )


def test_resolver_version_refusal_is_returned_not_raised_and_sorts_the_versions():
    error = refuse_resolver_version(1, [5, 3, 4, 3])
    assert "supports exactly [3, 4, 5]." in str(error) and error.supported == (3, 4, 5)
    with pytest.raises(ResolverVersionError) as caught:
        raise refuse_resolver_version(None, {3})
    assert "contract version None;" in str(caught.value)


@pytest.mark.parametrize("supported", [[], set(), [3.0], ["3"], [True], None])
def test_resolver_version_refusal_needs_integer_versions(supported):
    with pytest.raises((ValueError, TypeError)) as caught:
        refuse_resolver_version(2, supported)
    assert not isinstance(caught.value, ResolverVersionError)


# --- Construction in one place, and the public surface ----------------------------------------


@pytest.mark.parametrize(
    ("fixed_words", "constructor"),
    [
        ("({}): the region {} {}. A region", "regions.py"),
        ("A region is a 2-D integer or boolean array", "regions.py"),
        ("resolver has contract version", "contracts.py"),
        ("Install the morphobricks-preprocessing release it pins", "contracts.py"),
        ("no white can be estimated from it", "stain.py"),
        ("no stain vector can be estimated from them", "stain.py"),
    ],
)
def test_each_message_is_written_only_by_its_constructor(fixed_words, constructor):
    """ADR 0005 'construction and propagation' (4): a source scan for the fixed words."""
    sources = [*(ROOT / "src").rglob("*.py"), *(ROOT / "tools").rglob("*.py")]
    holders = [p.name for p in sources if any(fixed_words in t for t in string_literals(p))]
    assert holders == [constructor]


def string_literals(path):
    """Every string literal of a source file as Python joins it (f-string fields as ``{}``).

    Adjacent literals are one node in the syntax tree, so a message split over lines is found
    whole.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    texts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            texts.append(
                "".join(
                    part.value if isinstance(part, ast.Constant) else "{}" for part in node.values
                )
            )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            texts.append(node.value)
    return texts


def test_public_surface_of_the_new_modules():
    assert regions.__all__ == [
        "EMPTY_REGION",
        "INVALID_REASONS",
        "MASK_REGION_INVALID",
        "REGION_REASONS",
        "RegionEstimate",
        "RegionMaskError",
        "check_region",
    ]
    assert contracts.__all__ == [
        "RESOLVER_VERSION_UNSUPPORTED",
        "ResolverVersionError",
        "refuse_resolver_version",
    ]
    for module in (regions, contracts):
        for name in module.__all__:
            assert hasattr(module, name), (module.__name__, name)
    assert issubclass(RegionMaskError, ValueError) and issubclass(ResolverVersionError, ValueError)
    # The private boolean-only checker is gone: optical_density uses the shared one.
    assert not hasattr(od, "_check_region")
    assert od.check_region is check_region


def test_the_region_functions_return_estimates():
    image, _, full = f06_image_and_regions()
    assert isinstance(od.region_white(image, full), RegionEstimate)
    result = od.region_vectors(image, np.array(WHITE), {"eosin": full})
    assert isinstance(result, Mapping) and isinstance(result["eosin"], RegionEstimate)
