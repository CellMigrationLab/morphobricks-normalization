"""Image-to-scalar conversion with stain and modality presets (contract §3.4, N67, N68).

:func:`to_scalar` turns one declared imaging unit into one float32 ``(Y, X)`` map, chosen by a
named, versioned preset (:data:`PRESETS`), for consumers that need a single channel
(FiberFeatures first). It is a method family of its own: the RGB joint-percentile
normalization of ``rgb.py`` is unchanged and unrelated. :func:`rgb_to_scalar` is the P0 entry
point, kept unchanged; it calls :func:`to_scalar` without the optional arguments.

Methods (a new stain is a new preset record, not a new code path):

- ``passthrough`` (``fluorescence``, ``shg``): a 2-D ``"YX"`` plane of any real numeric dtype,
  returned as a float32 copy. Values that are not finite, or not finite in float32, are refused.
  The cast rounds integers above 2**24 and float64 values to the nearest float32; the
  provenance reports ``float32_exact``.
- ``fixed_deconvolution`` (``he_eosin``, ``he_hematoxylin``, ``trichrome_blue``):
  ``skimage.color.separate_stains`` with a fixed scikit-image matrix. It floors every pixel at
  1e-6, treats 255 as white, divides by ``-log(1e-6)`` and **sets negative component values to
  0**; ``zero_fraction`` is the fraction of pixels whose target component is exactly 0
  after the call (a pixel clipped from a negative value and one that was 0 cannot be told
  apart).
- ``luminance`` (``luminance_inverted``: ``1 - rgb2gray``; ``psr_polarized``: ``rgb2gray``, not
  inverted, since birefringent fibres are bright on a dark field) and ``cielab`` (``lab_a``:
  a*; ``trichrome_green``: minus a*), signed or bounded, not clipped.
- ``custom_od`` (``psr_brightfield``, ``he_custom``, ``trichrome_custom``): optical density
  against a declared ``white`` and user-supplied ``stain_vectors``, in NumPy
  (:mod:`morphobricks_normalization.optical_density`), negative concentrations kept.

RGB input contract. ``axes`` is exactly ``"YXC"`` or ``"CYX"`` and is never guessed; the array
is 3-D with exactly three samples on the colour axis (R, G, B in that order; an alpha channel is
refused, never dropped); the dtype is ``uint8`` (sRGB; 16-bit and ICC handling are later work).
Nothing is reordered, rescaled or repaired silently, and the input array is never modified.

Arguments a preset does not use (``white`` for ``lab_a``, ``polarizer`` for ``he_eosin``) are
refused, never ignored. ``psr_polarized`` needs ``polarizer`` (``"circular"`` or ``"linear"``):
a linear polarizer darkens fibres at some angles to its axis, so the provenance hints record
``orientation_valid = False`` for it and consumers must refuse orientation features then.

Each preset carries consumer ``hints`` (``polarity``, ``scaling``, ``mask``, ``measures``):
recommendations for consumers such as FiberFeatures (#214), recorded in the provenance and
never applied here. ``fixed_range`` is ``None`` until calibrated from a reference set.

scikit-image is the optional ``histology`` extra, imported only by the presets that call it;
``passthrough`` and ``custom_od`` need NumPy only, and importing this module needs NumPy only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType, ModuleType
from typing import Any

import numpy as np

from morphobricks_normalization import optical_density as od
from morphobricks_normalization import scalar_inputs as inputs

AXES = inputs.AXES
PLANE_AXES = inputs.PLANE_AXES
SAMPLE_COUNT = inputs.SAMPLE_COUNT
COLOUR_SPACE = "srgb"
INSTALL_HINT = "pip install morphobricks-normalization[histology]"
MODALITIES = ("fluorescence", "shg", "brightfield", "polarized")
SCALING_HINTS = ("per_image", "fixed")
MASK_HINTS = (
    "exclude_nuclei_rbc_empty",
    "exclude_nuclei",
    "exclude_counterstain",
    "background",
    "none",
    "optional",
)
POLARIZERS = ("circular", "linear")
OPTIONAL_ARGUMENTS = ("white", "stain_vectors", "polarizer")
EMPTY: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True)
class StainPreset:
    """A frozen, versioned recipe for one scalar map from one imaging unit.

    Attributes
    ----------
    id
        Preset identifier, the key in :data:`PRESETS`.
    version
        Version of the recipe; any change of numbers is a new version.
    stain
        Stain as proposed for the MorphoBricks Spec (``"he"``, ``"picrosirius_red"``,
        ``"trichrome_blue"``, ``"trichrome_green"``), or ``None``.
    method
        ``passthrough``, ``fixed_deconvolution``, ``luminance``, ``cielab`` or ``custom_od``.
    matrix
        Name of the scikit-image stain matrix (``hed_from_rgb``, ``bpx_from_rgb``), or ``None``.
    target_component
        Index of the component (deconvolution) or CIELAB channel returned, or ``None``.
    exclusion_component
        Component returned as the optional exclusion map (haematoxylin), or ``None``.
    polarity
        ``"bright"``: the structure of interest has higher values than its surroundings. It
        is the signal direction for ridge detection, not a stain amount (``lab_a`` is redness,
        ``luminance_inverted`` darkness).
    fixed_range
        Range for fixed intensity scaling; ``None`` until calibrated from a reference set.
    modality
        One of :data:`MODALITIES`.
    hints
        Read-only consumer hints: ``polarity``, ``scaling`` (:data:`SCALING_HINTS`), ``mask``
        (:data:`MASK_HINTS`) and ``measures`` (short text). Recommendations, not applied here.
    needs
        The optional arguments of :func:`to_scalar` this preset requires; the others are
        refused.
    inverted
        ``luminance``: ``1 - gray`` instead of ``gray``; ``cielab``: minus the channel.
    target_stain, exclusion_stain
        ``custom_od``: names of the target and exclusion vectors in ``stain_vectors``.
    """

    id: str
    version: int = 1
    stain: str | None = None
    method: str = "fixed_deconvolution"
    matrix: str | None = None
    target_component: int | None = None
    exclusion_component: int | None = None
    polarity: str = "bright"
    fixed_range: tuple[float, float] | None = None
    modality: str = "brightfield"
    hints: Mapping[str, str] = field(default_factory=lambda: EMPTY, hash=False)
    needs: frozenset[str] = frozenset()
    inverted: bool = False
    target_stain: str | None = None
    exclusion_stain: str | None = None


@dataclass(frozen=True)
class ScalarResult:
    """The scalar map of :func:`to_scalar`.

    Attributes
    ----------
    image
        float32 ``(Y, X)``, finite.
    exclusion
        float32 ``(Y, X)`` (haematoxylin for ``he_eosin``, the exclusion stain for
        ``custom_od``), or ``None`` if the preset has none.
    provenance
        Read-only mapping that records how the map was made.
    """

    image: np.ndarray
    exclusion: np.ndarray | None
    provenance: Mapping[str, Any]


def _hints(scaling: str, mask: str, measures: str) -> Mapping[str, str]:
    """Return read-only consumer hints; every preset is ``bright``."""
    return MappingProxyType(
        {"polarity": "bright", "scaling": scaling, "mask": mask, "measures": measures}
    )


_HE_MASK = "exclude_nuclei_rbc_empty"
_CUSTOM = frozenset({"white", "stain_vectors"})

PRESETS: Mapping[str, StainPreset] = MappingProxyType(
    {
        preset.id: preset
        for preset in (
            StainPreset(
                "he_eosin",
                stain="he",
                matrix="hed_from_rgb",
                target_component=1,
                exclusion_component=0,
                hints=_hints("fixed", _HE_MASK, "eosin fibrillar texture, not collagen-specific"),
            ),
            StainPreset(
                "he_hematoxylin",
                stain="he",
                matrix="hed_from_rgb",
                target_component=0,
                hints=_hints("fixed", "none", "haematoxylin, mainly nuclei"),
            ),
            StainPreset(
                "luminance_inverted",
                method="luminance",
                inverted=True,
                hints=_hints("fixed", _HE_MASK, "inverted brightness (baseline), nuclei-dominated"),
            ),
            StainPreset(
                "lab_a",
                method="cielab",
                target_component=1,
                hints=_hints("fixed", _HE_MASK, "CIELAB a* redness (comparator), low contrast"),
            ),
            StainPreset(
                "fluorescence",
                method="passthrough",
                modality="fluorescence",
                hints=_hints("per_image", "optional", "the labelled structure"),
            ),
            StainPreset(
                "shg",
                method="passthrough",
                modality="shg",
                hints=_hints("fixed", "none", "fibrillar collagen (mainly type I)"),
            ),
            StainPreset(
                "psr_brightfield",
                stain="picrosirius_red",
                method="custom_od",
                needs=_CUSTOM,
                target_stain="psr",
                exclusion_stain="counterstain",
                hints=_hints("fixed", "exclude_counterstain", "collagen"),
            ),
            StainPreset(
                "psr_polarized",
                stain="picrosirius_red",
                method="luminance",
                modality="polarized",
                needs=frozenset({"polarizer"}),
                hints=_hints("fixed", "background", "birefringent packed collagen"),
            ),
            StainPreset(
                "trichrome_blue",
                stain="trichrome_blue",
                matrix="bpx_from_rgb",
                target_component=0,
                hints=_hints("fixed", "exclude_nuclei", "collagen (methyl blue)"),
            ),
            StainPreset(
                "trichrome_green",
                stain="trichrome_green",
                method="cielab",
                target_component=1,
                inverted=True,
                hints=_hints("fixed", "exclude_nuclei", "collagen (green, minus CIELAB a*)"),
            ),
            StainPreset(
                "he_custom",
                stain="he",
                method="custom_od",
                needs=_CUSTOM,
                target_stain="eosin",
                exclusion_stain="hematoxylin",
                hints=_hints("fixed", _HE_MASK, "eosin fibrillar texture, not collagen-specific"),
            ),
            StainPreset(
                "trichrome_custom",
                method="custom_od",
                needs=_CUSTOM,
                target_stain="collagen",
                exclusion_stain="nuclei",
                hints=_hints("fixed", "exclude_nuclei", "collagen"),
            ),
        )
    }
)
"""The preset catalogue (issue #94, owner decision of 13 November 2026), read-only."""

PRESETS_BY_MODALITY: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        modality: tuple(key for key, preset in PRESETS.items() if preset.modality == modality)
        for modality in MODALITIES
    }
)
"""Preset ids by modality, read-only, in registry order."""


def _skimage_color() -> tuple[ModuleType, ModuleType]:
    """Import ``skimage`` and ``skimage.color`` lazily; name the extra when missing."""
    try:
        import skimage
        from skimage import color
    except ImportError as error:
        raise ImportError(f"RGB-to-scalar presets need scikit-image: {INSTALL_HINT}") from error
    return skimage, color


def _convert(rgb: np.ndarray, preset: StainPreset) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Return ``(image, exclusion, skimage version)`` as float32, from a ``(Y, X, 3)`` uint8."""
    skimage, color = _skimage_color()
    exclusion = None
    if preset.method == "fixed_deconvolution":
        matrix = getattr(color, preset.matrix)
        components = color.separate_stains(rgb, matrix)
        image = components[..., preset.target_component]
        if preset.exclusion_component is not None:
            exclusion = components[..., preset.exclusion_component]
    elif preset.method == "luminance":
        gray = color.rgb2gray(rgb)
        image = 1.0 - gray if preset.inverted else gray
    elif preset.method == "cielab":
        channel = color.rgb2lab(rgb)[..., preset.target_component]
        image = -channel if preset.inverted else channel
    else:
        raise ValueError(f"Preset {preset.id!r} has an unknown method {preset.method!r}.")
    as_float32 = np.ascontiguousarray(image, dtype=np.float32)
    excluded = None if exclusion is None else np.ascontiguousarray(exclusion, dtype=np.float32)
    return as_float32, excluded, skimage.__version__


def _custom_od(
    rgb: np.ndarray, preset: StainPreset, white: object, stain_vectors: object
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Unmix ``rgb`` with the declared white and vectors; return image, exclusion, provenance."""
    white_values = od.check_white(white)
    matrix = od.stain_matrix(stain_vectors, preset.target_stain, preset.exclusion_stain)
    density, floored = od.optical_density(rgb, white_values)
    amounts = od.concentrations(density, matrix)
    image = np.ascontiguousarray(amounts[..., 0], dtype=np.float32)
    exclusion = np.ascontiguousarray(amounts[..., 1], dtype=np.float32)
    extra = {
        "white": tuple(white_values.tolist()),
        "stain_names": matrix.names,
        "stain_matrix": matrix.rows,
        "condition_number": matrix.condition,
        "residual_completed": matrix.completed,
        "od_definition": "-log10(max(I, 1) / white)",
        "floored_fraction": floored,
        "negative_fraction": float(np.mean(amounts[..., 0] < 0)),
        "above_white_fraction": od.above_white_fraction(rgb, white_values),
    }
    return image, exclusion, extra


def _check_arguments(recipe: StainPreset, given: Mapping[str, Any]) -> None:
    """Refuse a missing required argument, an unused one, or an unknown polarizer."""
    for name in OPTIONAL_ARGUMENTS:
        if name in recipe.needs and given[name] is None:
            raise ValueError(f"Preset {recipe.id!r} needs {name}; it has no default.")
        if name not in recipe.needs and given[name] is not None:
            raise ValueError(
                f"Preset {recipe.id!r} does not use {name}; it is refused rather than ignored."
            )
    polarizer = given["polarizer"]
    if "polarizer" in recipe.needs and polarizer not in POLARIZERS:
        raise ValueError(f"polarizer must be 'circular' or 'linear'; got {polarizer!r}.")


def _provenance(
    recipe: StainPreset, pixels: np.ndarray, axes: str, extra: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Return the read-only provenance of one conversion."""
    hints = dict(recipe.hints)
    if "polarizer" in recipe.needs:
        # A linear polarizer darkens fibres at some angles to its axis (orientation bias).
        hints["orientation_valid"] = extra["polarizer"] == "circular"
    record = {
        "preset": recipe.id,
        "preset_version": recipe.version,
        "stain": recipe.stain,
        "modality": recipe.modality,
        "method": recipe.method,
        "matrix": recipe.matrix,
        "target_component": recipe.target_component,
        "exclusion_component": recipe.exclusion_component,
        "polarity": recipe.polarity,
        "hints": MappingProxyType(hints),
        "input_dtype": str(pixels.dtype),
        "axes": axes,
        "colour_space": None if recipe.method == "passthrough" else COLOUR_SPACE,
        "skimage_version": None,
        "zero_fraction": None,
    }
    record.update(extra)
    return MappingProxyType(record)


def to_scalar(
    pixels: np.ndarray,
    *,
    axes: str,
    preset: str,
    white: object = None,
    stain_vectors: Mapping[str, Any] | None = None,
    polarizer: str | None = None,
) -> ScalarResult:
    """Convert one imaging unit to one float32 map with a preset of :data:`PRESETS`.

    Parameters
    ----------
    pixels
        ``passthrough`` presets: a 2-D real numeric plane. Others: a ``uint8`` 3-D array with
        exactly three samples (R, G, B) on the colour axis. Never modified.
    axes
        ``"YX"`` for ``passthrough``; ``"YXC"`` or ``"CYX"`` otherwise. Required, never guessed.
    preset
        A key of :data:`PRESETS`.
    white
        ``custom_od`` only: background R, G, B of the 8-bit image, three finite values in
        ``(0, 255]`` (see :func:`white_from_region`).
    stain_vectors
        ``custom_od`` only: mapping stain name -> three R, G, B optical-density components; the
        preset's target and exclusion names, and optionally ``"residual"`` (otherwise the cross
        product of the first two). See :func:`stain_vectors_from_regions`.
    polarizer
        ``psr_polarized`` only: ``"circular"`` or ``"linear"``.

    Returns
    -------
    ScalarResult
        ``image`` float32 ``(Y, X)``; ``exclusion`` or ``None``; read-only ``provenance``
        (contract §3.4): preset id, version, stain, modality, method, matrix, components,
        polarity, hints, input dtype, axes, colour space, scikit-image version (``None`` when
        not used), ``zero_fraction``, and the method's own entries (``custom_od``: white,
        stain names, matrix, condition number, floored, negative and above-white fractions;
        ``psr_polarized``: polarizer; ``passthrough``: ``float32_exact``).

    Raises
    ------
    ValueError
        Unknown preset; wrong axes, shape or values; a missing, unused or invalid optional
        argument; stain vectors that are missing, extra, collinear or ill-conditioned.
    TypeError
        ``pixels`` is not a NumPy array or has a refused dtype.
    ImportError
        The preset needs scikit-image and it is not installed.
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; known presets: {', '.join(PRESETS)}.")
    recipe = PRESETS[preset]
    given = {"white": white, "stain_vectors": stain_vectors, "polarizer": polarizer}
    _check_arguments(recipe, given)
    exclusion = None
    if recipe.method == "passthrough":
        image, extra = inputs.plane_float32(pixels, axes)
    elif recipe.method == "custom_od":
        image, exclusion, extra = _custom_od(
            inputs.rgb_yxc(pixels, axes), recipe, white, stain_vectors
        )
    else:
        image, exclusion, version = _convert(inputs.rgb_yxc(pixels, axes), recipe)
        clipped = float(np.mean(image == 0)) if recipe.method == "fixed_deconvolution" else None
        extra = {"skimage_version": version, "zero_fraction": clipped}
        if "polarizer" in recipe.needs:
            extra["polarizer"] = polarizer
    provenance = _provenance(recipe, pixels, axes, extra)
    return ScalarResult(image=image, exclusion=exclusion, provenance=provenance)


def rgb_to_scalar(pixels: np.ndarray, *, axes: str, preset: str) -> ScalarResult:
    """Convert an 8-bit sRGB image to one float32 map with a preset that needs no argument.

    The P0 entry point (N67): :func:`to_scalar` without ``white``, ``stain_vectors`` or
    ``polarizer``, so a preset that needs one of them is refused with a message naming it.
    Outputs of ``he_eosin``, ``he_hematoxylin``, ``luminance_inverted`` and ``lab_a`` are
    unchanged from N67.

    Parameters
    ----------
    pixels
        ``uint8`` array, 3-D, with exactly three samples (R, G, B) on the colour axis. Never
        modified.
    axes
        ``"YXC"`` or ``"CYX"``; required, never guessed.
    preset
        A key of :data:`PRESETS`.

    Returns
    -------
    ScalarResult
        As :func:`to_scalar`.

    Raises
    ------
    ValueError
        Unknown preset, a preset that needs an argument, axes other than ``"YXC"``/``"CYX"``,
        not 3-D, or not three samples (an alpha channel included).
    TypeError
        ``pixels`` is not a NumPy array or its dtype is not ``uint8``.
    ImportError
        scikit-image is not installed (``pip install morphobricks-normalization[histology]``).
    """
    return to_scalar(pixels, axes=axes, preset=preset)


def white_from_region(
    pixels: np.ndarray, *, axes: str, region: np.ndarray
) -> tuple[float, float, float]:
    """Return the median R, G, B of a boolean background region: the declared ``white``.

    Parameters
    ----------
    pixels
        ``uint8`` RGB array as for :func:`to_scalar`. Never modified.
    axes
        ``"YXC"`` or ``"CYX"``.
    region
        Boolean ``(Y, X)`` array selecting background (no tissue).

    Returns
    -------
    tuple of float
        The white, valid for ``to_scalar(..., white=...)``.

    Raises
    ------
    ValueError
        Empty region, shape mismatch, or a median channel at 0 or 255 (dark or saturated).
    TypeError
        Wrong pixel dtype, or the region is not boolean.
    """
    return od.region_white(inputs.rgb_yxc(pixels, axes), region)


def stain_vectors_from_regions(
    pixels: np.ndarray, *, axes: str, white: object, regions: Mapping[str, np.ndarray]
) -> Mapping[str, tuple[float, float, float]]:
    """Return one unit optical-density vector per labelled region (the TWOMBLI approach).

    Each vector is the mean of ``-log10(max(I, 1) / white)`` over the region, normalised to
    unit length: label the matrix stain and the nuclear stain; the background gives ``white``
    (:func:`white_from_region`).

    Parameters
    ----------
    pixels
        ``uint8`` RGB array as for :func:`to_scalar`. Never modified.
    axes
        ``"YXC"`` or ``"CYX"``.
    white
        Declared background R, G, B, three finite values in ``(0, 255]``.
    regions
        Mapping stain name -> boolean ``(Y, X)`` array of pixels of that stain only.

    Returns
    -------
    Mapping
        Read-only, stain name -> unit vector, valid as ``stain_vectors``.

    Raises
    ------
    ValueError
        Invalid white; an empty or wrongly shaped region; a region whose mean optical density
        is about 0 (background, not stain).
    TypeError
        Wrong pixel dtype, ``regions`` not a non-empty mapping, or a region not boolean.
    """
    return od.region_vectors(inputs.rgb_yxc(pixels, axes), od.check_white(white), regions)
