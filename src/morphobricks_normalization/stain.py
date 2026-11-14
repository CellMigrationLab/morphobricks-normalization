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
  a*; ``trichrome_green``: minus a*), signed or bounded, not clipped. They accept an optional
  declared ``white`` (owner decision of 13 November 2026, N69): the RGB is then rescaled per
  channel to ``clip(rint(I * 255 / white), 0, 255)`` as uint8 before the scikit-image call, and
  the fraction of clipped pixels is reported. Without ``white`` nothing changes.
- ``custom_od`` (``psr_brightfield``, ``he_custom``, ``trichrome_custom``): optical density
  against a declared ``white`` and ``stain_vectors``, in NumPy
  (:mod:`morphobricks_normalization.optical_density`), negative concentrations kept.
  ``he_custom`` and ``trichrome_custom`` have published default vectors
  (:data:`HE_DEFAULT_VECTORS`, :data:`BPX_DEFAULT_VECTORS`), used when ``stain_vectors`` is not
  given; ``psr_brightfield`` has none. ``white`` is required for all three.

RGB input contract. ``axes`` is exactly ``"YXC"`` or ``"CYX"`` and is never guessed; the array
is 3-D with exactly three samples on the colour axis (R, G, B in that order; an alpha channel is
refused, never dropped); the dtype is ``uint8`` (sRGB; 16-bit and ICC handling are later work).
Nothing is reordered, rescaled or repaired silently, and the input array is never modified.

Arguments a preset does not use (``white`` for ``he_eosin``, ``polarizer`` for ``lab_a``) are
refused, never ignored. :func:`check_options` makes every check of the preset and of these
arguments without pixels, so a consumer can refuse a conversion before reading an image;
:func:`to_scalar` runs it first (one implementation, the same messages). ``psr_polarized``
needs ``polarizer`` (``"circular"`` or ``"linear"``): a linear polarizer darkens fibres at some
angles to its axis, so the provenance hints record ``orientation_valid = False`` for it and
consumers must refuse orientation features then.

Each preset carries consumer ``hints`` (``polarity``, ``scaling``, ``mask``, ``measures``):
recommendations for consumers such as FiberFeatures (#214), recorded in the provenance and
never applied here. ``fixed_range`` is ``None`` until calibrated from a reference set.

scikit-image is the optional ``histology`` extra, imported only by the presets that call it;
``passthrough`` and ``custom_od`` need NumPy only, and importing this module needs NumPy only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, ModuleType
from typing import Any

import numpy as np

from morphobricks_normalization import optical_density as od
from morphobricks_normalization import scalar_inputs as inputs
from morphobricks_normalization.stain_presets import BPX_CITATION as BPX_CITATION
from morphobricks_normalization.stain_presets import BPX_DEFAULT_VECTORS as BPX_DEFAULT_VECTORS
from morphobricks_normalization.stain_presets import HE_CITATION as HE_CITATION
from morphobricks_normalization.stain_presets import HE_DEFAULT_VECTORS as HE_DEFAULT_VECTORS
from morphobricks_normalization.stain_presets import MASK_HINTS as MASK_HINTS
from morphobricks_normalization.stain_presets import MODALITIES as MODALITIES
from morphobricks_normalization.stain_presets import PRESETS as PRESETS
from morphobricks_normalization.stain_presets import PRESETS_BY_MODALITY as PRESETS_BY_MODALITY
from morphobricks_normalization.stain_presets import SCALING_HINTS as SCALING_HINTS
from morphobricks_normalization.stain_presets import StainPreset as StainPreset

AXES = inputs.AXES
PLANE_AXES = inputs.PLANE_AXES
SAMPLE_COUNT = inputs.SAMPLE_COUNT
COLOUR_SPACE = "srgb"
INSTALL_HINT = "pip install morphobricks-normalization[histology]"
POLARIZERS = ("circular", "linear")
OPTIONAL_ARGUMENTS = ("white", "stain_vectors", "polarizer")
PRESET_DEFAULT = "preset_default"
USER = "user"
"""Values of the provenance entry ``stain_vectors_source``."""
WHITE_RESCALE = "clip(rint(I * 255 / white), 0, 255) as uint8, rint half to even"
"""The rescaling applied by a ``luminance`` or ``cielab`` preset given a ``white``."""


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


def _skimage_color() -> tuple[ModuleType, ModuleType]:
    """Import ``skimage`` and ``skimage.color`` lazily; name the extra when missing."""
    try:
        import skimage
        from skimage import color
    except ImportError as error:
        raise ImportError(f"RGB-to-scalar presets need scikit-image: {INSTALL_HINT}") from error
    return skimage, color


@dataclass(frozen=True)
class _Options:
    """The checked options of one conversion (:func:`_checked_options`).

    Attributes
    ----------
    recipe
        The preset.
    white
        The checked white as float64 ``(3,)``, or ``None`` if not given.
    matrix
        ``custom_od`` only: the checked stain matrix (user or default vectors), else ``None``.
    vectors_source, vectors_citation
        ``custom_od`` only: :data:`USER` or :data:`PRESET_DEFAULT`, and the default's citation.
    """

    recipe: StainPreset
    white: np.ndarray | None
    matrix: od.StainMatrix | None
    vectors_source: str | None
    vectors_citation: str | None


def _white_rescaled(
    rgb: np.ndarray, values: np.ndarray | None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rescale ``rgb`` to a checked white if given; return it and the provenance entries."""
    if values is None:
        entries = {"white": None, "white_applied": False, "white_rescale": None}
        entries |= {"white_clipped_fraction": None, "above_white_fraction": None}
        return rgb, entries
    rescaled, clipped = od.rescale_to_white(rgb, values)
    entries = {
        "white": tuple(values.tolist()),
        "white_applied": True,
        "white_rescale": WHITE_RESCALE,
        "white_clipped_fraction": clipped,
        "above_white_fraction": od.above_white_fraction(rgb, values),
    }
    return rescaled, entries


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


def _custom_od(rgb: np.ndarray, options: _Options) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Unmix ``rgb`` with the checked white and matrix; return image, exclusion, provenance."""
    white_values, matrix = options.white, options.matrix
    source, citation = options.vectors_source, options.vectors_citation
    density, floored = od.optical_density(rgb, white_values)
    amounts = od.concentrations(density, matrix)
    image = np.ascontiguousarray(amounts[..., 0], dtype=np.float32)
    exclusion = np.ascontiguousarray(amounts[..., 1], dtype=np.float32)
    extra = {
        "white": tuple(white_values.tolist()),
        "stain_vectors_source": source,
        "stain_vectors_citation": citation,
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
    """Refuse a missing required argument, an unused one, or a polarizer not in POLARIZERS."""
    for name in OPTIONAL_ARGUMENTS:
        if name in recipe.needs and given[name] is None:
            raise ValueError(f"Preset {recipe.id!r} needs {name}; it has no default.")
        used = recipe.needs | recipe.accepts
        if name not in used and given[name] is not None:
            raise ValueError(
                f"Preset {recipe.id!r} does not use {name}; it is refused rather than ignored."
            )
    polarizer = given["polarizer"]
    if "polarizer" not in recipe.needs:
        return
    # A str first: an array would compare element-wise, so np.array(["linear"]) would pass the
    # membership test and make orientation_valid an array (PR #98, B1).
    if not isinstance(polarizer, str):
        kind = type(polarizer)
        type_name = (
            kind.__qualname__
            if kind.__module__ == "builtins"
            else f"{kind.__module__}.{kind.__qualname__}"
        )
        raise ValueError(
            f"polarizer must be the text 'circular' or 'linear'; got {type_name} {polarizer!r}."
        )
    if polarizer not in POLARIZERS:
        raise ValueError(f"polarizer must be 'circular' or 'linear'; got {polarizer!r}.")


def _checked_options(
    preset: str, white: object, stain_vectors: object, polarizer: object
) -> _Options:
    """Make every check of :func:`check_options` and return the checked values.

    The order among these checks is the one :func:`to_scalar` had before #97 (the preset, the
    presence rules, ``white``, then the stain matrix); they now all come before the pixels.
    Without ``stain_vectors`` a ``custom_od`` preset's published defaults are checked as user
    vectors are; a preset without defaults never gets here without them (the presence rules
    refuse it).
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; known presets: {', '.join(PRESETS)}.")
    recipe = PRESETS[preset]
    given = {"white": white, "stain_vectors": stain_vectors, "polarizer": polarizer}
    _check_arguments(recipe, given)
    values = None if white is None else od.check_white(white)
    matrix, source, citation = None, None, None
    if recipe.method == "custom_od":
        source = USER
        if stain_vectors is None:
            stain_vectors = recipe.default_stain_vectors
            source, citation = PRESET_DEFAULT, recipe.default_stain_vectors_citation
        matrix = od.stain_matrix(stain_vectors, recipe.target_stain, recipe.exclusion_stain)
    return _Options(recipe, values, matrix, source, citation)


def check_options(
    preset: str,
    *,
    white: object = None,
    stain_vectors: Mapping[str, Any] | None = None,
    polarizer: str | None = None,
) -> None:
    """Check a conversion request without pixels: every check :func:`to_scalar` makes of these.

    A consumer calls it when planning, before reading any image, so that an unusable request
    (``white`` for ``he_eosin``, an unknown polarizer, collinear stain vectors) is refused
    before any pixel is read. :func:`to_scalar` runs the same checks first, with the same
    exceptions and messages; it then checks only the pixels and ``axes``.

    Parameters
    ----------
    preset
        A key of :data:`PRESETS`.
    white, stain_vectors, polarizer
        As for :func:`to_scalar`.

    Raises
    ------
    ValueError
        Unknown preset (the known ones are listed); a needed argument missing; an argument the
        preset does not use given (refused, never ignored); a polarizer not in
        :data:`POLARIZERS` or not a ``str`` (an array, bytes or a list is refused with its type
        named, never compared); ``white`` not three finite values in ``(0, 255]``; stain
        vectors with missing or other names, not three finite numbers, of (almost) zero length,
        closer than 5 degrees (either sign) or with a condition number above 50 (contract §3.4).
    TypeError
        ``white`` or a stain vector is a masked array or holds a masked value, at any depth of
        nested lists, tuples or object arrays (refused before ``np.asarray`` would drop the
        mask); ``stain_vectors`` is not a mapping.
    """
    _checked_options(preset, white, stain_vectors, polarizer)


def _provenance(
    recipe: StainPreset, pixels: np.ndarray, axes: str, extra: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Return the read-only provenance of one conversion."""
    hints = dict(recipe.hints)
    if "polarizer" in recipe.needs:
        # A linear polarizer darkens fibres at some angles to its axis (orientation bias).
        hints["orientation_valid"] = bool(extra["polarizer"] == "circular")
    record = {
        "preset": recipe.id,
        "preset_version": recipe.version,
        "image_name": recipe.image_name,
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
        Background R, G, B of the 8-bit image, three finite values in ``(0, 255]`` (see
        :func:`white_from_region`). Required for ``custom_od``; optional for ``luminance`` and
        ``cielab`` (the RGB is then rescaled to it, :data:`WHITE_RESCALE`); refused otherwise.
    stain_vectors
        ``custom_od`` only: mapping stain name -> three R, G, B optical-density components; the
        preset's target and exclusion names, and optionally ``"residual"`` (otherwise the cross
        product of the first two). See :func:`stain_vectors_from_regions`. Optional for
        ``he_custom`` and ``trichrome_custom`` (published defaults), required for
        ``psr_brightfield``.
    polarizer
        ``psr_polarized`` only: ``"circular"`` or ``"linear"``.

    Returns
    -------
    ScalarResult
        ``image`` float32 ``(Y, X)``; ``exclusion`` or ``None``; read-only ``provenance``
        (contract §3.4): preset id, version, stain, modality, method, matrix, components,
        polarity, hints, input dtype, axes, colour space, scikit-image version (``None`` when
        not used), ``zero_fraction``, and the method's own entries (``custom_od``: white,
        ``stain_vectors_source`` and citation, stain names, matrix, condition number, floored,
        negative and above-white fractions; ``luminance`` and ``cielab``: ``white``,
        ``white_applied``, ``white_rescale``, ``white_clipped_fraction``,
        ``above_white_fraction``; ``psr_polarized``: polarizer; ``passthrough``:
        ``float32_exact``).

    Raises
    ------
    ValueError
        Unknown preset; wrong axes, shape or values; a missing, unused or invalid optional
        argument; stain vectors that are missing, extra, collinear or ill-conditioned. The
        checks of :func:`check_options` come first, before any pixel is looked at.
    TypeError
        ``pixels`` is not a NumPy array or has a refused dtype; a masked optional argument
        (as for :func:`check_options`).
    ImportError
        The preset needs scikit-image and it is not installed.
    """
    # check_options' checks, before any pixel is looked at (#97); their values are used below.
    options = _checked_options(preset, white, stain_vectors, polarizer)
    recipe = options.recipe
    exclusion = None
    if recipe.method == "passthrough":
        image, extra = inputs.plane_float32(pixels, axes)
    elif recipe.method == "custom_od":
        image, exclusion, extra = _custom_od(inputs.rgb_yxc(pixels, axes), options)
    else:
        rgb = inputs.rgb_yxc(pixels, axes)
        white_entries: dict[str, Any] = {}
        if "white" in recipe.accepts:
            rgb, white_entries = _white_rescaled(rgb, options.white)
        image, exclusion, version = _convert(rgb, recipe)
        clipped = float(np.mean(image == 0)) if recipe.method == "fixed_deconvolution" else None
        extra = {"skimage_version": version, "zero_fraction": clipped, **white_entries}
        if "polarizer" in recipe.needs:
            extra["polarizer"] = str(polarizer)  # plain str in provenance (np.str_ accepted)
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
