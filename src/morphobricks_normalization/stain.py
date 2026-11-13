"""RGB-to-scalar conversion with stain presets (contract §3.4, plan N67, issue #94).

:func:`rgb_to_scalar` turns an 8-bit sRGB image into one float32 map, chosen by a named,
versioned preset (:data:`PRESETS`), for consumers that need a single channel from a stained
image (FiberFeatures first). It is a method family of its own: the RGB joint-percentile
normalization of ``rgb.py`` is unchanged and unrelated.

Presets (P0): ``he_eosin``, ``he_hematoxylin`` (Ruifrok-Johnston H&E vectors through
``skimage.color.separate_stains`` and ``hed_from_rgb``), ``luminance_inverted`` (baseline) and
``lab_a`` (CIELAB a*, comparator). A new stain is a new preset record, not a new code path.

Input contract. ``axes`` is exactly ``"YXC"`` or ``"CYX"`` and is never guessed; the array is
3-D with exactly three samples on the colour axis (R, G, B in that order; an alpha channel is
refused, never dropped); the dtype is ``uint8`` (sRGB; other dtypes are refused, 16-bit and ICC
handling are later work). Nothing is reordered, rescaled or repaired silently, and the input
array is never modified. ``uint8`` cannot hold NaN or infinity, so the output is finite.

Behaviour inherited from scikit-image, accepted in P0 and recorded in the provenance:
``separate_stains`` floors every pixel value at 1e-6, treats 255 as white, divides by
``-log(1e-6)`` and **sets negative component values to 0**. The provenance reports
``clipped_fraction``: the fraction of the pixels whose target component is exactly 0 after the
call (a pixel that was negative or truly 0 cannot be told apart). ``lab_a`` is signed and not
clipped, and ``luminance_inverted`` is not clipped either; both report ``None``.

``fixed_range`` of a preset is ``None`` for now: the stain-OD range a consumer needs for fixed
intensity scaling must be calibrated from a reference set and recorded, not hard-coded from one
image (issue #94).

scikit-image is the optional ``histology`` extra, imported only inside the conversion; without
it :func:`rgb_to_scalar` raises :class:`ImportError`, and importing this module needs NumPy only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, ModuleType
from typing import Any

import numpy as np

AXES = ("YXC", "CYX")
SAMPLE_COUNT = 3
COLOUR_SPACE = "srgb"
INSTALL_HINT = "pip install morphobricks-normalization[histology]"


@dataclass(frozen=True)
class StainPreset:
    """A frozen, versioned recipe for one scalar map from an RGB image.

    Attributes
    ----------
    id
        Preset identifier, the key in :data:`PRESETS`.
    version
        Version of the recipe; any change of numbers is a new version.
    stain
        Stain declared in the MorphoBricks Spec (``"he"``), or ``None`` for a colour projection.
    method
        ``fixed_deconvolution``, ``luminance`` or ``cielab``.
    matrix
        Name of the scikit-image stain matrix (``hed_from_rgb``), or ``None``.
    target_component
        Index of the component (deconvolution) or CIELAB channel returned, or ``None``.
    exclusion_component
        Component returned as the optional exclusion map (haematoxylin), or ``None``.
    polarity
        ``"bright"``: a higher value means more stain.
    fixed_range
        Stain range for fixed intensity scaling; ``None`` until calibrated from a reference set.
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


@dataclass(frozen=True)
class ScalarResult:
    """The scalar map of :func:`rgb_to_scalar`.

    Attributes
    ----------
    image
        float32 ``(Y, X)``, finite.
    exclusion
        float32 ``(Y, X)`` (haematoxylin for ``he_eosin``), or ``None`` if the preset has none.
    provenance
        Read-only mapping that records how the map was made.
    """

    image: np.ndarray
    exclusion: np.ndarray | None
    provenance: Mapping[str, Any]


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
            ),
            StainPreset("he_hematoxylin", stain="he", matrix="hed_from_rgb", target_component=0),
            StainPreset("luminance_inverted", method="luminance"),
            StainPreset("lab_a", method="cielab", target_component=1),
        )
    }
)
"""The four P0 presets, read-only. No trichrome presets yet (issue #94, P1)."""


def _to_yxc(pixels: np.ndarray, axes: str) -> np.ndarray:
    """Return a C-contiguous ``(Y, X, 3)`` copy after checking the input contract."""
    if not isinstance(pixels, np.ndarray):
        raise TypeError(f"pixels must be a NumPy array; got {type(pixels).__name__}.")
    if not isinstance(axes, str) or axes not in AXES:
        raise ValueError(f"axes must be exactly 'YXC' or 'CYX', never guessed; got {axes!r}.")
    if pixels.ndim != 3:
        raise ValueError(f"Expected a 3-D array with axes {axes}; got {pixels.ndim}-D.")
    colour_axis = axes.index("C")
    if pixels.shape[colour_axis] != SAMPLE_COUNT:
        extra = (
            " An alpha channel is refused, not dropped." if pixels.shape[colour_axis] == 4 else ""
        )
        raise ValueError(
            f"Expected exactly 3 samples (R, G, B) on the colour axis of {axes}; "
            f"got {pixels.shape[colour_axis]}.{extra}"
        )
    if pixels.dtype != np.uint8:
        raise TypeError(
            f"Only 8-bit sRGB (uint8) is supported; got {pixels.dtype}. "
            "Nothing is rescaled or converted silently."
        )
    return np.ascontiguousarray(np.moveaxis(pixels, colour_axis, 2))


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
        image = 1.0 - color.rgb2gray(rgb)
    elif preset.method == "cielab":
        image = color.rgb2lab(rgb)[..., preset.target_component]
    else:
        raise ValueError(f"Preset {preset.id!r} has an unknown method {preset.method!r}.")
    as_float32 = np.ascontiguousarray(image, dtype=np.float32)
    excluded = None if exclusion is None else np.ascontiguousarray(exclusion, dtype=np.float32)
    return as_float32, excluded, skimage.__version__


def rgb_to_scalar(pixels: np.ndarray, *, axes: str, preset: str) -> ScalarResult:
    """Convert an 8-bit sRGB image to one float32 map with a stain preset.

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
        ``image`` float32 ``(Y, X)``; ``exclusion`` (``he_eosin`` only) or ``None``;
        ``provenance`` with preset id and version, stain, method, matrix, target and exclusion
        component, polarity, input dtype, axes, colour space, scikit-image version and
        ``clipped_fraction`` (fraction of target-component pixels exactly 0, ``None`` where the
        preset does not clip).

    Raises
    ------
    ValueError
        Unknown preset, axes other than ``"YXC"``/``"CYX"``, not 3-D, or not three samples
        (an alpha channel included).
    TypeError
        ``pixels`` is not a NumPy array or its dtype is not ``uint8``.
    ImportError
        scikit-image is not installed (``pip install morphobricks-normalization[histology]``).
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; known presets: {', '.join(PRESETS)}.")
    recipe = PRESETS[preset]
    rgb = _to_yxc(pixels, axes)
    image, exclusion, skimage_version = _convert(rgb, recipe)
    clipped = float(np.mean(image == 0)) if recipe.method == "fixed_deconvolution" else None
    provenance = MappingProxyType(
        {
            "preset": recipe.id,
            "preset_version": recipe.version,
            "stain": recipe.stain,
            "method": recipe.method,
            "matrix": recipe.matrix,
            "target_component": recipe.target_component,
            "exclusion_component": recipe.exclusion_component,
            "polarity": recipe.polarity,
            "input_dtype": str(pixels.dtype),
            "axes": axes,
            "colour_space": COLOUR_SPACE,
            "skimage_version": skimage_version,
            "clipped_fraction": clipped,
        }
    )
    return ScalarResult(image=image, exclusion=exclusion, provenance=provenance)
