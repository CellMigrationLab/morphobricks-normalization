"""Optical density and stain unmixing (NumPy only): analytic checks of the Beer-Lambert maths."""

import math

import numpy as np
import pytest

from morphobricks_normalization import optical_density as od
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
    unit,
    unit_matrix,
)

WHITE_ARRAY = np.array(WHITE)


def three():
    return {"eosin": EOSIN, "hematoxylin": HEMATOXYLIN, "residual": DAB}


def test_float_round_trip_is_exact_with_three_vectors():
    matrix = od.stain_matrix(three(), "eosin", "hematoxylin")
    amounts = concentrations(0, (8, 9), 3)
    image = beer_lambert(amounts, matrix.array())
    density, floored = od.optical_density(image, WHITE_ARRAY)
    recovered = od.concentrations(density, matrix)
    assert floored == 0.0
    np.testing.assert_allclose(recovered, amounts, rtol=0, atol=1e-12)


def test_float_round_trip_is_exact_with_two_vectors_and_the_cross_product():
    matrix = od.stain_matrix({"psr": PSR, "counterstain": COUNTERSTAIN}, "psr", "counterstain")
    assert matrix.completed and matrix.names == ("psr", "counterstain", "residual")
    expected = unit_matrix(PSR, COUNTERSTAIN)
    np.testing.assert_allclose(matrix.array(), expected, rtol=0, atol=1e-15)
    amounts = concentrations(1, (8, 9), 2)
    density, _ = od.optical_density(beer_lambert(amounts, expected), WHITE_ARRAY)
    recovered = od.concentrations(density, matrix)
    np.testing.assert_allclose(recovered[..., :2], amounts, rtol=0, atol=1e-12)
    np.testing.assert_allclose(recovered[..., 2], 0.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("vectors", [three(), {"eosin": EOSIN, "hematoxylin": HEMATOXYLIN}])
def test_uint8_round_trip_is_within_the_rounding_bound(vectors):
    matrix = od.stain_matrix(vectors, "eosin", "hematoxylin")
    m = matrix.array()
    amounts = concentrations(2, (20, 20), len(vectors))
    rounded = to_uint8(beer_lambert(amounts, m))
    density, _ = od.optical_density(rounded, WHITE_ARRAY)
    error = np.abs(od.concentrations(density, matrix)[..., : len(vectors)] - amounts)
    bound = concentration_bound(rounded, m)[..., : len(vectors)]
    assert (error <= bound).all()
    assert error.max() > 1e-4  # the quantisation is real, so the bound is exercised


def test_optical_density_floors_zero_and_reports_it():
    image = np.full((2, 5, 3), 120, np.uint8)
    image[0, 0] = (0, 50, 50)
    image[1, 4] = (0, 0, 0)
    density, floored = od.optical_density(image, WHITE_ARRAY)
    assert floored == pytest.approx(2 / 10)
    assert density[0, 0, 0] == pytest.approx(-math.log10(1 / WHITE[0]))
    assert np.isfinite(density).all()


def test_above_white_fraction_counts_any_brighter_channel():
    image = np.full((2, 4, 3), 100, np.uint8)
    image[0, 0] = (241, 100, 100)  # red above 240
    image[0, 1] = (240, 235, 245)  # equal to white: not above
    image[1, 3] = (255, 255, 255)
    assert od.above_white_fraction(image, WHITE_ARRAY) == pytest.approx(2 / 8)
    density, _ = od.optical_density(image, WHITE_ARRAY)
    assert density[0, 0, 0] < 0 and np.all(density[0, 1] == 0)


def test_two_vector_condition_number_is_cot_half_angle():
    for degrees in (10.0, 30.0, 60.0, 90.0):
        theta = math.radians(degrees)
        a, b = (1.0, 0.0, 0.0), (math.cos(theta), math.sin(theta), 0.0)
        matrix = od.stain_matrix({"t": a, "e": b}, "t", "e")
        assert matrix.condition == pytest.approx(1 / math.tan(theta / 2), rel=1e-12)
    # The 5-degree rule keeps two-vector matrices well below MAX_CONDITION.
    assert 1 / math.tan(math.radians(od.MIN_ANGLE_DEGREES) / 2) < od.MAX_CONDITION


def test_vectors_are_normalised_and_scale_free():
    a = od.stain_matrix({"t": PSR, "e": COUNTERSTAIN}, "t", "e")
    b = od.stain_matrix({"t": np.multiply(PSR, 7), "e": np.multiply(COUNTERSTAIN, 0.1)}, "t", "e")
    np.testing.assert_allclose(a.array(), b.array(), rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(a.array(), axis=1), 1.0, rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ({"t": PSR}, "missing \\['e'\\]"),
        ({"e": PSR}, "missing \\['t'\\]"),
        ({"t": PSR, "e": COUNTERSTAIN, "dab": DAB}, "not allowed \\['dab'\\]"),
        ({"t": PSR, "e": (0.0, 0.0, 0.0)}, "zero length"),
        ({"t": PSR, "e": (np.nan, 0.5, 0.5)}, "not finite"),
        ({"t": PSR, "e": (np.inf, 0.5, 0.5)}, "not finite"),
        ({"t": PSR, "e": (0.5, 0.5)}, "three numbers"),
        ({"t": PSR, "e": ("a", "b", "c")}, "three numbers"),
        ({"t": PSR, "e": np.multiply(PSR, 2.0)}, "collinear"),
        ({"t": PSR, "e": np.multiply(PSR, -1.0)}, "collinear"),
        ({"t": (1.0, 0.0, 0.0), "e": (1.0, 0.07, 0.0)}, "4.00 degrees"),
        ({"t": PSR, "e": COUNTERSTAIN, "residual": PSR}, "collinear"),
    ],
)
def test_invalid_stain_vectors_are_refused(vectors, message):
    with pytest.raises(ValueError, match=message):
        od.stain_matrix(vectors, "t", "e")


def test_an_angle_just_above_the_limit_is_accepted():
    theta = math.radians(od.MIN_ANGLE_DEGREES + 0.01)
    od.stain_matrix({"t": (1, 0, 0), "e": (math.cos(theta), math.sin(theta), 0)}, "t", "e")


def test_three_nearly_coplanar_vectors_are_refused_as_ill_conditioned():
    third = unit((1.0, 1.0, 0.02))  # 45 degrees from both, but almost in their plane
    rows = np.array([(1, 0, 0), (0, 1, 0), third])
    assert np.linalg.cond(rows) > od.MAX_CONDITION
    with pytest.raises(ValueError, match="ill-conditioned"):
        od.stain_matrix({"t": rows[0], "e": rows[1], "residual": third}, "t", "e")


def test_a_non_mapping_is_refused():
    with pytest.raises(TypeError, match="mapping"):
        od.stain_matrix([PSR, COUNTERSTAIN], "t", "e")


@pytest.mark.parametrize(
    "white",
    [(0, 200, 200), (256, 200, 200), (-1, 200, 200), (np.nan, 200, 200), (np.inf, 1, 1)],
)
def test_white_out_of_range_is_refused(white):
    with pytest.raises(ValueError, match=r"\(0, 255\]"):
        od.check_white(white)


@pytest.mark.parametrize(
    "white", [(200, 200), (200, 200, 200, 255), 240, "white", (True, True, True), [[1, 2, 3]]]
)
def test_white_of_wrong_shape_or_type_is_refused(white):
    with pytest.raises(ValueError, match="three numbers"):
        od.check_white(white)


def test_white_accepts_255_and_fractions():
    assert od.check_white([255, 0.5, 170]).tolist() == [255.0, 0.5, 170.0]


def test_region_white_is_the_background_median():
    rng = np.random.default_rng(3)
    image = rng.integers(20, 120, (10, 12, 3)).astype(np.uint8)
    background = np.zeros((10, 12), bool)
    background[:4] = True
    image[background] = rng.integers(160, 181, (int(background.sum()), 3))
    expected = tuple(np.median(image[background].astype(float), axis=0))
    assert od.region_white(image, background) == expected


@pytest.mark.parametrize("value", [0, 255])
def test_region_white_refuses_dark_or_saturated_backgrounds(value):
    image = np.full((4, 4, 3), 200, np.uint8)
    image[..., 1] = value
    with pytest.raises(ValueError, match="dark or saturated"):
        od.region_white(image, np.ones((4, 4), bool))


@pytest.mark.parametrize(
    ("region", "error", "message"),
    [
        (np.zeros((4, 4), bool), ValueError, "empty"),
        (np.ones((4, 5), bool), ValueError, "shape"),
        (np.ones((4, 4), np.uint8), TypeError, "boolean"),
        ([[True]], TypeError, "boolean"),
    ],
)
def test_region_white_refuses_bad_regions(region, error, message):
    with pytest.raises(error, match=message):
        od.region_white(np.full((4, 4, 3), 200, np.uint8), region)


def test_region_vectors_recover_the_generating_vectors():
    m = unit_matrix(PSR, COUNTERSTAIN)
    amounts = np.zeros((20, 20, 2))
    amounts[:, :10, 0] = concentrations(4, (20, 10), 1, high=0.8)[..., 0] + 0.2
    amounts[:, 10:, 1] = concentrations(5, (20, 10), 1, high=0.8)[..., 0] + 0.2
    image = to_uint8(beer_lambert(amounts, m))
    left = np.zeros((20, 20), bool)
    left[:, :10] = True
    vectors = od.region_vectors(image, WHITE_ARRAY, {"psr": left, "counterstain": ~left})
    for name, row in (("psr", m[0]), ("counterstain", m[1])):
        angle = math.degrees(math.acos(min(1.0, float(np.dot(vectors[name], row)))))
        assert angle < 0.2, (name, angle)
        assert np.linalg.norm(vectors[name]) == pytest.approx(1.0)
    with pytest.raises(TypeError):
        vectors["psr"] = (1.0, 0.0, 0.0)


def test_region_vectors_refuse_background_like_and_empty_regions():
    image = np.empty((4, 4, 3), np.uint8)
    image[...] = (239, 234, 244)  # one grey level below white: OD about 0.002 per channel
    region = np.ones((4, 4), bool)
    with pytest.raises(ValueError, match="background"):
        od.region_vectors(image, WHITE_ARRAY, {"psr": region})
    with pytest.raises(ValueError, match="empty"):
        od.region_vectors(image, WHITE_ARRAY, {"psr": np.zeros((4, 4), bool)})
    with pytest.raises(TypeError, match="non-empty mapping"):
        od.region_vectors(image, WHITE_ARRAY, {})


def test_eosin_dab_hematoxylin_are_well_conditioned():
    matrix = od.stain_matrix(three(), "eosin", "hematoxylin")
    assert matrix.condition < 10
