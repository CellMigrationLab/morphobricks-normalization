"""Optical density and stain unmixing against a declared white (contract §3.4, plan N68).

The in-house ``custom_od`` path of :mod:`morphobricks_normalization.stain`, in NumPy only (no
scikit-image), so that slide-specific stain vectors (the TWOMBLI approach) work without the
``histology`` extra. Every function takes a ``(Y, X, 3)`` RGB array already checked by the
caller and returns new arrays; nothing is modified in place.

Beer-Lambert law per channel ``c``: ``I_c = white_c * 10 ** (-OD_c)`` and ``OD = C @ M``, where
the rows of ``M`` are the unit optical-density vectors of the stains and ``C`` holds one
concentration per stain. Hence:

- ``OD_c = -log10(max(I_c, 1) / white_c)``. A pixel value of 0 has no finite optical density;
  it is floored at 1 (the smallest non-zero 8-bit value) and the fraction of pixels with a
  floored channel is reported, never hidden.
- ``C = OD @ inv(M)``. Negative concentrations are kept, not clipped: they mean the pixel does
  not fit the declared stains (noise, a third colour, a white that is too dark), and the caller
  reports their fraction. Pixels brighter than the white have a negative optical density in
  that channel and are reported too.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

CHANNELS = 3
WHITE_MAX = 255.0
"""Upper limit of a declared white: the largest 8-bit value."""
FLOOR = 1.0
"""Pixel values below 1 (that is, 0) are raised to 1 before the logarithm."""
RESIDUAL = "residual"
"""Name of the third stain vector, declared or completed as a cross product."""
MIN_VECTOR_NORM = 1e-6
"""A stain vector shorter than this has no direction; optical densities are of order 0.1-1."""
MIN_ANGLE_DEGREES = 5.0
"""Two stain vectors closer than this (in either direction) are refused as collinear.

For two unit vectors at angle ``theta`` completed by their cross product the condition number
of ``M`` is ``cot(theta / 2)``: 5 degrees gives 22.9, so this rule is the binding one for two
vectors and :data:`MAX_CONDITION` the one for three nearly coplanar vectors.
"""
MAX_CONDITION = 50.0
"""Largest accepted 2-norm condition number of the unit stain matrix: a heuristic threshold.

The norm-wise relative error of a pixel's concentration vector is at most ``cond(M)`` times the
norm-wise relative error of its optical-density vector. One 8-bit grey level at a mid-range
value (``I`` about 80, ``OD`` about 0.5) is an optical-density error of about 1 %, so at 50 the
worst-case norm-wise amplification is of order 50 %. This is a provisional refusal of nearly
singular matrices, not a guarantee on any one stain map or pixel (the error of a single
component can be smaller or larger than the norm-wise bound suggests); concentration errors are
to be benchmarked on realistic noise and backgrounds. The measured condition number is recorded
in the provenance. Ruifrok-Johnston H&E-DAB has 3.8 and scikit-image's Methyl Blue/Ponceau
Fuchsin matrix 2.0.
"""
MIN_REGION_OD = 0.01
"""Smallest optical-density norm of a region's mean for it to define a stain vector.

0.01 is about 2 % absorption, five grey levels below a white of 240: a region below it is
background, not stain, and its direction would be noise.
"""


@dataclass(frozen=True)
class StainMatrix:
    """Unit stain vectors as rows, in the order target, exclusion, residual.

    Attributes
    ----------
    names
        The three stain names, row by row.
    rows
        The unit vectors, as nested tuples (read-only).
    condition
        2-norm condition number of the matrix.
    completed
        Whether the residual row is the normalised cross product of the first two.
    """

    names: tuple[str, str, str]
    rows: tuple[tuple[float, float, float], ...]
    condition: float
    completed: bool

    def array(self) -> np.ndarray:
        """Return the matrix as a new float64 ``(3, 3)`` array."""
        return np.array(self.rows, dtype=np.float64)


def _holds_masked(value: object) -> bool:
    """Return whether ``value`` is or holds, at any depth, a masked array or ``np.ma.masked``.

    Lists, tuples and object arrays are searched element by element (``np.ma.masked`` is a
    masked array too); numeric arrays cannot hold a masked value. Each container is visited
    once, so a list that holds itself ends the search.
    """
    pending, seen = [value], set()
    while pending:
        item = pending.pop()
        if isinstance(item, np.ma.MaskedArray):
            return True
        is_sequence = isinstance(item, (list, tuple))
        is_object_array = isinstance(item, np.ndarray) and item.dtype == object
        if (is_sequence or is_object_array) and id(item) not in seen:
            seen.add(id(item))
            pending.extend(item.flat if is_object_array else item)
    return False


def _refuse_masked(value: object, what: str) -> None:
    """Refuse a masked array or any masked content, before ``np.asarray`` drops the mask.

    Masked content is found in nested lists and tuples and in object arrays too (PR #98, B2).
    """
    if _holds_masked(value):
        raise TypeError(
            f"{what} is a masked array (or holds a masked value): its mask would be dropped "
            "silently. Pass plain numbers."
        )


def check_white(white: object) -> np.ndarray:
    """Return a declared white as a float64 ``(3,)`` array after checking it.

    Parameters
    ----------
    white
        Background RGB of the 8-bit image: three finite numbers in ``(0, 255]``.

    Returns
    -------
    numpy.ndarray
        float64 ``(3,)``.

    Raises
    ------
    ValueError
        Not three finite numbers in ``(0, 255]``.
    """
    _refuse_masked(white, "white")
    values = np.asarray(white)
    if values.dtype.kind not in "iuf" or values.shape != (CHANNELS,):
        raise ValueError(
            "white must be three numbers (the background R, G, B of the 8-bit image); "
            f"got {white!r}."
        )
    values = values.astype(np.float64)
    if not np.isfinite(values).all() or (values <= 0).any() or (values > WHITE_MAX).any():
        raise ValueError(
            f"Each value of white must be finite and in (0, 255]; got {tuple(values.tolist())}. "
            "Measure it on background, for example with white_from_region."
        )
    return values


def _unit_vector(name: str, values: object) -> np.ndarray:
    """Return the stain vector ``name`` normalised to unit length, or refuse it."""
    _refuse_masked(values, f"Stain vector {name!r}")
    vector = np.asarray(values)
    if vector.dtype.kind not in "iuf" or vector.shape != (CHANNELS,):
        raise ValueError(
            f"Stain vector {name!r} must be three numbers (R, G, B optical densities); "
            f"got {values!r}."
        )
    vector = vector.astype(np.float64)
    if not np.isfinite(vector).all():
        raise ValueError(f"Stain vector {name!r} is not finite: {tuple(vector.tolist())}.")
    norm = float(np.linalg.norm(vector))
    if norm < MIN_VECTOR_NORM:
        raise ValueError(
            f"Stain vector {name!r} has (almost) zero length ({norm:.3g}); it has no direction."
        )
    return vector / norm


def _check_names(vectors: Mapping[str, Any], target: str, exclusion: str) -> None:
    """Refuse a mapping without the target and exclusion stains or with other names."""
    if not isinstance(vectors, Mapping):
        raise TypeError(f"stain_vectors must be a mapping name -> 3 values; got {vectors!r}.")
    allowed = {target, exclusion, RESIDUAL}
    missing = [name for name in (target, exclusion) if name not in vectors]
    extra = sorted(str(name) for name in vectors if name not in allowed)
    if missing or extra:
        raise ValueError(
            f"stain_vectors must name {target!r} and {exclusion!r}, and optionally "
            f"{RESIDUAL!r}; missing {missing}, not allowed {extra}."
        )


def _angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    """Return the angle between two unit vectors, ignoring their sign (0 to 90 degrees)."""
    return float(np.degrees(np.arccos(min(abs(float(a @ b)), 1.0))))


def stain_matrix(vectors: Mapping[str, Any], target: str, exclusion: str) -> StainMatrix:
    """Build and check the unit stain matrix of a ``custom_od`` preset.

    Parameters
    ----------
    vectors
        Mapping stain name -> three optical-density components (R, G, B), any positive scale.
        Exactly ``target`` and ``exclusion``, plus optionally ``"residual"``; without it the
        third row is the normalised cross product of the first two.
    target, exclusion
        The preset's stain names; their rows come first and second.

    Returns
    -------
    StainMatrix

    Raises
    ------
    ValueError
        Missing or extra names; a vector that is not three finite numbers or has zero length;
        two vectors within :data:`MIN_ANGLE_DEGREES`; a condition number above
        :data:`MAX_CONDITION`.
    """
    _check_names(vectors, target, exclusion)
    names = (target, exclusion, RESIDUAL)
    rows = [_unit_vector(name, vectors[name]) for name in names[:2]]
    completed = RESIDUAL not in vectors
    if not completed:
        rows.append(_unit_vector(RESIDUAL, vectors[RESIDUAL]))
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            angle = _angle_degrees(rows[i], rows[j])
            if angle < MIN_ANGLE_DEGREES:
                raise ValueError(
                    f"Stain vectors {names[i]!r} and {names[j]!r} are {angle:.2f} degrees apart "
                    f"(minimum {MIN_ANGLE_DEGREES}); collinear stains cannot be separated."
                )
    if completed:
        cross = np.cross(rows[0], rows[1])
        rows.append(cross / np.linalg.norm(cross))
    matrix = np.array(rows)
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > MAX_CONDITION:
        raise ValueError(
            f"The stain matrix is ill-conditioned (condition number {condition:.3g}, maximum "
            f"{MAX_CONDITION}); the vectors are too close to a plane to be separated."
        )
    as_tuples = tuple(tuple(float(v) for v in row) for row in matrix)
    return StainMatrix(names, as_tuples, condition, completed)


def optical_density(rgb: np.ndarray, white: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ``(OD, floored_fraction)`` of a ``(Y, X, 3)`` image against ``white``.

    ``OD = -log10(max(I, 1) / white)`` per channel, float64. ``floored_fraction`` is the
    fraction of pixels with at least one channel below 1 (raised to 1).
    """
    values = rgb.astype(np.float64)
    floored = values < FLOOR
    od = -np.log10(np.maximum(values, FLOOR) / white)
    return od, float(np.mean(floored.any(axis=-1)))


def above_white_fraction(rgb: np.ndarray, white: np.ndarray) -> float:
    """Return the fraction of pixels brighter than ``white`` in at least one channel."""
    return float(np.mean((rgb.astype(np.float64) > white).any(axis=-1)))


def rescale_to_white(rgb: np.ndarray, white: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ``(rescaled, clipped_fraction)``: ``rgb`` rescaled per channel so ``white`` is 255.

    ``rescaled = clip(rint(I * 255 / white), 0, 255)`` as ``uint8``, computed in float64 in
    that order; ``rint`` rounds half to even. With ``white = (255, 255, 255)`` it is the
    identity (``I * 255 / 255`` is exact in float64). ``clipped_fraction`` is the fraction of
    pixels with at least one channel above 255 after rounding (brighter than the white by more
    than half a grey level), set to 255.
    """
    scaled = np.rint(rgb.astype(np.float64) * WHITE_MAX / white)
    clipped = scaled > WHITE_MAX
    rescaled = np.clip(scaled, 0.0, WHITE_MAX).astype(np.uint8)
    return rescaled, float(np.mean(clipped.any(axis=-1)))


def concentrations(od: np.ndarray, matrix: StainMatrix) -> np.ndarray:
    """Return ``OD @ inv(M)``: one float64 concentration per stain, signed, never clipped."""
    return od @ np.linalg.inv(matrix.array())


def _check_region(region: object, shape: tuple[int, int], name: str) -> np.ndarray:
    """Return a boolean ``(Y, X)`` region; refuse a masked, mistyped, misshaped or empty one."""
    if isinstance(region, np.ma.MaskedArray):
        raise TypeError(
            f"Region {name!r} is a masked array: which pixels belong to it would be ambiguous. "
            "Pass a plain boolean array."
        )
    if not isinstance(region, np.ndarray) or region.dtype != np.bool_:
        raise TypeError(f"Region {name!r} must be a boolean NumPy array (Y, X).")
    if region.shape != shape:
        raise ValueError(f"Region {name!r} has shape {region.shape}; the image is {shape} (Y, X).")
    if not region.any():
        raise ValueError(f"Region {name!r} is empty; select at least one pixel.")
    return region


def region_white(rgb: np.ndarray, region: object) -> tuple[float, float, float]:
    """Return the median R, G, B of a ``(Y, X, 3)`` image over a background region.

    Raises
    ------
    ValueError
        Empty region, wrong shape, or a median of 0 or 255 (no light, or saturated: the
        true white is then unknown).
    TypeError
        The region is not a boolean array.
    """
    mask = _check_region(region, rgb.shape[:2], "background")
    median = np.median(rgb[mask].astype(np.float64), axis=0)
    if (median <= 0).any() or (median >= WHITE_MAX).any():
        raise ValueError(
            f"The background median {tuple(median.tolist())} has a channel at 0 or 255 "
            "(dark or saturated): an unsaturated white cannot be inferred from this region. "
            "Lower the exposure or choose another background region; a white declared "
            "explicitly (for example 255 for a scanner that maps background to 255) is accepted."
        )
    return tuple(float(v) for v in median)


def region_vectors(
    rgb: np.ndarray, white: np.ndarray, regions: Mapping[str, Any]
) -> Mapping[str, tuple[float, float, float]]:
    """Return the unit mean optical-density vector of each labelled region.

    A selected pixel with a channel at 0 would enter at the floor (:data:`FLOOR`, intensity 1)
    and bend the region's direction, so such a region is refused; zeros outside the regions do
    not matter.

    Raises
    ------
    ValueError
        No regions, an empty or wrongly shaped region, a region holding a pixel with a channel at
        0, or a region whose mean optical density is below :data:`MIN_REGION_OD` (background,
        not stain).
    TypeError
        ``regions`` is not a mapping, or a region is not a boolean array.
    """
    if not isinstance(regions, Mapping) or not regions:
        raise TypeError("regions must be a non-empty mapping stain name -> boolean (Y, X) array.")
    od, _ = optical_density(rgb, white)
    vectors = {}
    for name, region in regions.items():
        mask = _check_region(region, rgb.shape[:2], name)
        floored = int((rgb[mask] < FLOOR).any(axis=-1).sum())
        if floored:
            raise ValueError(
                f"Region {name!r} holds {floored} pixel(s) with a channel at 0: their optical "
                "density would be taken at the floor (intensity 1) and bend the stain vector. "
                "Choose a reference region without saturated-dark pixels."
            )
        mean = od[mask].mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm < MIN_REGION_OD:
            raise ValueError(
                f"Region {name!r} has a mean optical density of {norm:.4f} (minimum "
                f"{MIN_REGION_OD}); it looks like background, not stain."
            )
        vectors[name] = tuple(float(v) for v in mean / norm)
    return MappingProxyType(vectors)
