"""The preset catalogue of :mod:`morphobricks_normalization.stain` (contract §3.4, N68, N69).

Frozen, versioned :class:`StainPreset` records in the read-only registry :data:`PRESETS`, their
consumer hints and the published default stain vectors of the ``custom_od`` presets
(:data:`HE_DEFAULT_VECTORS`, :data:`BPX_DEFAULT_VECTORS`). Data only: the conversions are in
``stain.py``. Importing this module needs nothing but the standard library.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

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
EMPTY: Mapping[str, Any] = MappingProxyType({})

# Ruifrok AC, Johnston DA. Quantification of histochemical staining by color deconvolution.
# Anal Quant Cytol Histol. 2001;23(4):291-299. Rows 0 (haematoxylin) and 1 (eosin) of
# scikit-image's ``skimage.color.rgb_from_hed`` (scikit-image 0.26), copied so that the preset
# needs no scikit-image; tests/test_stain_defaults.py checks them against scikit-image.
HE_CITATION = (
    "Ruifrok & Johnston 2001, Anal Quant Cytol Histol 23:291-299; "
    "skimage.color.rgb_from_hed rows 0 (hematoxylin) and 1 (eosin)"
)
HE_DEFAULT_VECTORS: Mapping[str, tuple[float, float, float]] = MappingProxyType(
    {"hematoxylin": (0.65, 0.70, 0.29), "eosin": (0.07, 0.99, 0.11)}
)
# Methyl Blue and Ponceau Fuchsin (Masson's trichrome), rows 0 and 1 of scikit-image's
# ``skimage.color.rgb_from_bpx`` (scikit-image 0.26), the values of the "Masson Trichrome"
# vectors of G. Landini's Colour Deconvolution plugin for ImageJ/Fiji, with the method of
# Ruifrok & Johnston 2001. Copied here; checked against scikit-image by the tests.
BPX_CITATION = (
    "Ruifrok & Johnston 2001 method; Masson Trichrome vectors of Landini's Colour "
    "Deconvolution (Fiji); skimage.color.rgb_from_bpx rows 0 (methyl blue) and 1 "
    "(ponceau fuchsin)"
)
BPX_DEFAULT_VECTORS: Mapping[str, tuple[float, float, float]] = MappingProxyType(
    {
        "collagen": (0.7995107, 0.5913521, 0.10528667),
        "ponceau_fuchsin": (0.09997159, 0.73738605, 0.6680326),
    }
)


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
        The optional arguments of :func:`to_scalar` this preset requires.
    accepts
        The optional arguments this preset uses when given but does not require (``white``
        for ``luminance`` and ``cielab``, ``stain_vectors`` when there are defaults). Every
        argument in neither ``needs`` nor ``accepts`` is refused.
    inverted
        ``luminance``: ``1 - gray`` instead of ``gray``; ``cielab``: minus the channel.
    target_stain, exclusion_stain
        ``custom_od``: names of the target and exclusion vectors in ``stain_vectors``.
    default_stain_vectors
        ``custom_od``: read-only published stain vectors (target and exclusion names), used
        when ``stain_vectors`` is not given, or ``None`` (then ``stain_vectors`` is needed).
    default_stain_vectors_citation
        Where the default vectors come from, recorded in the provenance.
    image_name
        The CellProfiler-style name of the image the preset makes (as CellProfiler's
        ``UnmixColors`` names its outputs ``Hematoxylin``, ``Eosin``, ``Methyl blue``): a token of
        letters and digits, unique among the presets, which consumers use as the derived
        image's channel name in feature columns (``Eosin__Cells__ridge_support_mean``). It names
        the image, not what it measures: ``Eosin`` is eosin-derived texture, not collagen.
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
    accepts: frozenset[str] = frozenset()
    inverted: bool = False
    target_stain: str | None = None
    exclusion_stain: str | None = None
    default_stain_vectors: Mapping[str, tuple[float, float, float]] | None = field(
        default=None, hash=False
    )
    default_stain_vectors_citation: str | None = None
    image_name: str | None = None


def _hints(scaling: str, mask: str, measures: str) -> Mapping[str, str]:
    """Return read-only consumer hints; every preset is ``bright``."""
    return MappingProxyType(
        {"polarity": "bright", "scaling": scaling, "mask": mask, "measures": measures}
    )


_HE_MASK = "exclude_nuclei_rbc_empty"
_CUSTOM = frozenset({"white", "stain_vectors"})
_WHITE = frozenset({"white"})

PRESETS: Mapping[str, StainPreset] = MappingProxyType(
    {
        preset.id: preset
        for preset in (
            StainPreset(
                "he_eosin",
                image_name="Eosin",
                stain="he",
                matrix="hed_from_rgb",
                target_component=1,
                exclusion_component=0,
                hints=_hints("fixed", _HE_MASK, "eosin fibrillar texture, not collagen-specific"),
            ),
            StainPreset(
                "he_hematoxylin",
                image_name="Hematoxylin",
                stain="he",
                matrix="hed_from_rgb",
                target_component=0,
                hints=_hints("fixed", "none", "haematoxylin, mainly nuclei"),
            ),
            StainPreset(
                "luminance_inverted",
                image_name="InvertedLuminance",
                method="luminance",
                inverted=True,
                accepts=_WHITE,
                hints=_hints("fixed", _HE_MASK, "inverted brightness (baseline), nuclei-dominated"),
            ),
            StainPreset(
                "lab_a",
                image_name="LabA",
                method="cielab",
                target_component=1,
                accepts=_WHITE,
                hints=_hints("fixed", _HE_MASK, "CIELAB a* redness (comparator), low contrast"),
            ),
            StainPreset(
                "fluorescence",
                image_name="Fluorescence",
                method="passthrough",
                modality="fluorescence",
                hints=_hints("per_image", "optional", "the labelled structure"),
            ),
            StainPreset(
                "shg",
                image_name="SHG",
                method="passthrough",
                modality="shg",
                hints=_hints("fixed", "none", "fibrillar collagen (mainly type I)"),
            ),
            StainPreset(
                "psr_brightfield",
                image_name="SiriusRed",
                stain="picrosirius_red",
                method="custom_od",
                needs=_CUSTOM,
                target_stain="psr",
                exclusion_stain="counterstain",
                hints=_hints("fixed", "exclude_counterstain", "collagen"),
            ),
            StainPreset(
                "psr_polarized",
                image_name="SiriusRedPolarized",
                stain="picrosirius_red",
                method="luminance",
                modality="polarized",
                needs=frozenset({"polarizer"}),
                accepts=_WHITE,
                hints=_hints("fixed", "background", "birefringent packed collagen"),
            ),
            StainPreset(
                "trichrome_blue",
                image_name="MethylBlue",
                stain="trichrome_blue",
                matrix="bpx_from_rgb",
                target_component=0,
                hints=_hints("fixed", "exclude_nuclei", "collagen (methyl blue)"),
            ),
            StainPreset(
                "trichrome_green",
                image_name="TrichromeGreen",
                stain="trichrome_green",
                method="cielab",
                target_component=1,
                inverted=True,
                accepts=_WHITE,
                hints=_hints("fixed", "exclude_nuclei", "collagen (green, minus CIELAB a*)"),
            ),
            StainPreset(
                "he_custom",
                image_name="EosinCustom",
                stain="he",
                method="custom_od",
                needs=_WHITE,
                accepts=frozenset({"stain_vectors"}),
                target_stain="eosin",
                exclusion_stain="hematoxylin",
                default_stain_vectors=HE_DEFAULT_VECTORS,
                default_stain_vectors_citation=HE_CITATION,
                hints=_hints("fixed", _HE_MASK, "eosin fibrillar texture, not collagen-specific"),
            ),
            StainPreset(
                "trichrome_custom",
                image_name="TrichromeCustom",
                method="custom_od",
                needs=_WHITE,
                accepts=frozenset({"stain_vectors"}),
                target_stain="collagen",
                # Ponceau Fuchsin (cytoplasm, muscle), not nuclei: the published Masson
                # matrix has no nuclear stain (iron haematoxylin is near black). The mask hint
                # stays exclude_nuclei: a nuclear mask is built in preprocessing (N69).
                exclusion_stain="ponceau_fuchsin",
                default_stain_vectors=BPX_DEFAULT_VECTORS,
                default_stain_vectors_citation=BPX_CITATION,
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
