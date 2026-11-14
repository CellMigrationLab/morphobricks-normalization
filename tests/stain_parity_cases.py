"""Inputs of the v0.6.0 parity tables of tests/test_stain_check_options.py (#97, PR #98).

``tests/data/stain_v060_parity.json`` freezes what ``stain.to_scalar`` of v0.6.0 (commit
``95e1d6d``) did with these inputs: the exception type and full message of every refusal in
:data:`REFUSALS` and :data:`CHANGED`, and the SHA-256 digests of the outputs of every conversion
in :data:`CONVERSIONS` (for ``custom_od``, the image values and provenance, compared with a
tolerance, and the exclusion's digest: :func:`frozen_output`). This module uses only names that
existed in v0.6.0, so the table can be recorded on that commit. To record it again (``<venv>``
with the ``histology`` extra)::

    git worktree add /tmp/v060 95e1d6d
    PYTHONPATH=/tmp/v060/src <venv>/bin/python tests/stain_parity_cases.py \\
        > tests/data/stain_v060_parity.json
    git worktree remove /tmp/v060

The script prints the path of the ``stain`` module it used on standard error: check that it is
the worktree's.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import numpy as np

from morphobricks_normalization import stain

BASE_COMMIT = "95e1d6d"
WHITE = (240.0, 235.0, 245.0)
PSR_VECTORS = {"psr": (0.10, 0.80, 0.55), "counterstain": (0.65, 0.70, 0.29)}
HE_VECTORS = {"eosin": (0.07, 0.99, 0.11), "hematoxylin": (0.65, 0.70, 0.29)}
COLLINEAR = math.radians(stain.od.MIN_ANGLE_DEGREES - 0.5)
NEAR_PLANE = (1.0, 1.0, 0.02)  # 45 degrees from x and y, but almost in their plane
SKIMAGE_VERSION = "<skimage_version>"
"""Stands for ``skimage_version`` in the hashed provenance; the test checks the real value."""


def psr(**vectors: Any) -> dict[str, Any]:
    """Return psr_brightfield options with the given stain vectors."""
    return {"white": WHITE, "stain_vectors": vectors}


def _masked_white_array() -> np.ndarray:
    """Return an object array holding ``np.ma.masked`` (built so NumPy keeps the constant)."""
    values = np.empty(3, dtype=object)
    values[:] = [240, np.ma.masked, 245]
    return values


def _object_scalar(item: object) -> np.ndarray:
    """Return a 0-d object array holding ``item``."""
    value = np.empty((), dtype=object)
    value[()] = item
    return value


def _object_vector(item: object) -> np.ndarray:
    """Return an object array ``(0.6, item, 0.3)``."""
    values = np.empty(3, dtype=object)
    values[:] = [0.6, item, 0.3]
    return values


# name -> (preset, options): every refusal check_options shares with to_scalar. Their v0.6.0
# exception types and messages are frozen in the "refusals" table and do not change.
REFUSALS: dict[str, tuple[str, dict[str, Any]]] = {
    "unknown preset": ("he_eosine", {}),
    "white missing (custom_od)": ("he_custom", {}),
    "vectors missing (psr)": ("psr_brightfield", {"white": WHITE}),
    "polarizer missing": ("psr_polarized", {}),
    "white unused (he_eosin)": ("he_eosin", {"white": WHITE}),
    "white unused (passthrough)": ("fluorescence", {"white": WHITE}),
    "polarizer unused (lab_a)": ("lab_a", {"polarizer": "circular"}),
    "vectors unused (trichrome_blue)": ("trichrome_blue", {"stain_vectors": PSR_VECTORS}),
    "vectors unused (luminance)": ("luminance_inverted", {"stain_vectors": PSR_VECTORS}),
    "polarizer unused (he_custom)": ("he_custom", {"white": WHITE, "polarizer": "linear"}),
    "unknown polarizer": ("psr_polarized", {"polarizer": "crossed"}),
    "polarizer wrong case": ("psr_polarized", {"polarizer": "Circular"}),
    "white zero": ("he_custom", {"white": (0, 200, 200)}),
    "white above 255": ("lab_a", {"white": (256, 200, 200)}),
    "white negative": ("psr_polarized", {"white": (-1, 200, 200), "polarizer": "circular"}),
    "white nan": ("trichrome_custom", {"white": (np.nan, 200, 200)}),
    "white inf": ("luminance_inverted", {"white": (np.inf, 1, 1)}),
    "white two values": ("he_custom", {"white": (200, 200)}),
    "white text": ("he_custom", {"white": ("a", "b", "c")}),
    "white scalar": ("trichrome_green", {"white": 240}),
    "white masked array": ("he_custom", {"white": np.ma.MaskedArray(WHITE, mask=[0, 1, 0])}),
    "white all-false mask": ("lab_a", {"white": np.ma.MaskedArray(WHITE, mask=False)}),
    "white list holding masked": (
        "psr_brightfield",
        {"white": [240, np.ma.masked, 245], "stain_vectors": PSR_VECTORS},
    ),
    "white tuple holding masked": ("luminance_inverted", {"white": (np.ma.masked, 235, 245)}),
    "vectors not a mapping": (
        "psr_brightfield",
        {"white": WHITE, "stain_vectors": [(1, 0, 0), (0, 1, 0)]},
    ),
    "vector name missing": ("psr_brightfield", psr(psr=(1, 0, 0))),
    "vector name extra": ("psr_brightfield", psr(**PSR_VECTORS, dab=(0, 0, 1))),
    "vector two values": ("psr_brightfield", psr(psr=(1, 0), counterstain=(0, 1, 0))),
    "vector text": ("he_custom", {"white": WHITE, "stain_vectors": {**HE_VECTORS, "eosin": "red"}}),
    "vector nan": ("psr_brightfield", psr(psr=(np.nan, 1, 0), counterstain=(0, 1, 0))),
    "vector inf": (
        "trichrome_custom",
        {
            "white": WHITE,
            "stain_vectors": {"collagen": (np.inf, 0, 0), "ponceau_fuchsin": (0, 1, 0)},
        },
    ),
    "vector zero length": ("psr_brightfield", psr(psr=(0, 0, 0), counterstain=(0, 1, 0))),
    "vector masked array": (
        "psr_brightfield",
        psr(psr=np.ma.MaskedArray((1.0, 0, 0), mask=False), counterstain=(0, 1, 0)),
    ),
    "vector list holding masked": (
        "he_custom",
        {"white": WHITE, "stain_vectors": {**HE_VECTORS, "hematoxylin": [0.6, np.ma.masked, 0.3]}},
    ),
    "residual masked": (
        "psr_brightfield",
        psr(**PSR_VECTORS, residual=np.ma.MaskedArray((0, 0, 1.0), mask=True)),
    ),
    "collinear": (
        "psr_brightfield",
        psr(psr=(1, 0, 0), counterstain=(math.cos(COLLINEAR), math.sin(COLLINEAR), 0)),
    ),
    "collinear opposite sign": (
        "psr_brightfield",
        psr(psr=(1, 0, 0), counterstain=(-math.cos(COLLINEAR), -math.sin(COLLINEAR), 0)),
    ),
    "collinear identical": (
        "he_custom",
        {"white": WHITE, "stain_vectors": {"eosin": (0, 1, 0), "hematoxylin": (0, 2, 0)}},
    ),
    "residual collinear": ("psr_brightfield", psr(**PSR_VECTORS, residual=(0.20, 1.60, 1.10))),
    "ill-conditioned": (
        "psr_brightfield",
        psr(psr=(1, 0, 0), counterstain=(0, 1, 0), residual=NEAR_PLANE),
    ),
}

# name -> (preset, options): inputs whose classification changed on purpose after v0.6.0 (PR
# #98 review blockers B1 and B2). The table keeps v0.6.0's outcome as evidence; the test checks
# the new refusal and that it differs.
CHANGED: dict[str, tuple[str, dict[str, Any]]] = {
    "B1 polarizer one-element array": ("psr_polarized", {"polarizer": np.array(["linear"])}),
    "B1 polarizer two-element array": (
        "psr_polarized",
        {"polarizer": np.array(["linear", "circular"])},
    ),
    "B1 polarizer bytes": ("psr_polarized", {"polarizer": b"linear"}),
    "B1 polarizer number": ("psr_polarized", {"polarizer": 1}),
    "B1 polarizer list": ("psr_polarized", {"polarizer": ["circular"]}),
    "B2 white object array holding masked": ("lab_a", {"white": _masked_white_array()}),
    "B2 white nested list holding masked": ("he_custom", {"white": [[240, np.ma.masked, 245]]}),
    "B2 white list holding a 0-d object array of masked": (
        "luminance_inverted",
        {"white": [240, _object_scalar(np.ma.masked), 245]},
    ),
    "B2 vector object array holding masked": (
        "he_custom",
        {"white": WHITE, "stain_vectors": {**HE_VECTORS, "eosin": _object_vector(np.ma.masked)}},
    ),
    "B2 vector nested list holding masked": (
        "psr_brightfield",
        psr(psr=[[0.1, np.ma.masked, 0.5]], counterstain=(0, 1, 0)),
    ),
    "B2 residual tuple of a masked array": (
        "psr_brightfield",
        psr(**PSR_VECTORS, residual=(0.3, (np.ma.MaskedArray(0.2, mask=False),), 0.9)),
    ),
}


def valid_pixels(preset: str) -> tuple[np.ndarray, str]:
    """Return pixels to_scalar accepts for ``preset``, so only the options can be refused."""
    rng = np.random.default_rng(3)
    if preset in stain.PRESETS and stain.PRESETS[preset].method == "passthrough":
        return rng.normal(100, 10, (5, 6)), "YX"
    return rng.integers(1, 256, (5, 6, 3), dtype=np.uint8), "YXC"


def _custom_od_cases(name: str, preset: Any, rgb: np.ndarray) -> Iterator[tuple]:
    """Yield the custom_od conversions of one preset: default or declared, then user vectors."""
    base: dict[str, Any] = {"white": (240, 235, 250)}
    if preset.default_stain_vectors is None:
        base["stain_vectors"] = {"psr": (0.2, 0.9, 0.6), "counterstain": (0.6, 0.7, 0.3)}
    yield name, rgb, "YXC", base
    vectors = {
        preset.target_stain: (0.1, 0.9, 0.2),
        preset.exclusion_stain: (0.7, 0.6, 0.3),
        "residual": (0.3, 0.3, 0.9),
    }
    yield name, rgb, "YXC", {"white": [250.5, 249, 251], "stain_vectors": vectors}


def conversions() -> Iterator[tuple[str, np.ndarray, str, dict[str, Any]]]:
    """Yield ``(preset, pixels, axes, options)``: 33 conversions over every preset.

    YXC and CYX, with and without ``white``, default and user vectors, both polarizers, and
    passthrough of a float64 and a uint16 plane.
    """
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, size=(13, 11, 3), dtype=np.uint8)
    plane = rng.normal(100, 30, size=(13, 11))
    for name, preset in stain.PRESETS.items():
        if preset.method == "passthrough":
            yield name, plane, "YX", {}
            yield name, plane.astype(np.uint16), "YX", {}
            continue
        if preset.method == "custom_od":
            yield from _custom_od_cases(name, preset, rgb)
            continue
        base = {"polarizer": "circular"} if "polarizer" in preset.needs else {}
        yield name, rgb, "YXC", base
        yield name, np.moveaxis(rgb, -1, 0).copy(), "CYX", base
        if "white" in preset.accepts:
            yield name, rgb, "YXC", {**base, "white": (230, 240, 250)}
            yield name, rgb, "YXC", {**base, "white": np.array([255, 255, 255])}
        if "polarizer" in preset.needs:
            yield name, rgb, "YXC", {"polarizer": "linear"}


def conversion_id(index: int, preset: str, axes: str, options: Mapping[str, Any]) -> str:
    """Return a readable, unique id of conversion ``index``."""
    return f"{index:02d} {preset} {axes} {'+'.join(options) or 'plain'}"


def _plain(value: object) -> object:
    """Return ``value`` as plain JSON types; refuse any other type (a NumPy scalar included)."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(f"Provenance value {value!r} of type {type(value).__name__} is not plain.")


def provenance_json(provenance: Mapping[str, Any]) -> str:
    """Return the canonical JSON of a provenance, ``skimage_version`` replaced by a marker."""
    record = dict(provenance)
    if record.get("skimage_version") is not None:
        record["skimage_version"] = SKIMAGE_VERSION
    return json.dumps(_plain(record), sort_keys=True, separators=(",", ":"), allow_nan=False)


def array_digest(values: np.ndarray | None) -> str | None:
    """Return the SHA-256 of an array's dtype, shape and C-order bytes, or ``None``."""
    if values is None:
        return None
    header = f"{values.dtype.str}{list(values.shape)}".encode()
    return hashlib.sha256(header + np.ascontiguousarray(values).tobytes()).hexdigest()


def output_digests(result: Any) -> dict[str, str | None]:
    """Return the digests of a ScalarResult: image, exclusion and canonical provenance."""
    provenance = provenance_json(result.provenance).encode()
    return {
        "image": array_digest(result.image),
        "exclusion": array_digest(result.exclusion),
        "provenance": hashlib.sha256(provenance).hexdigest(),
    }


def output_values(result: Any) -> dict[str, Any]:
    """Return a ScalarResult's image values and plain provenance, with the exclusion's digest.

    Used for ``custom_od`` conversions: they solve a stain matrix through LAPACK, whose last
    bits depend on the CPU kernel OpenBLAS selects, so they are compared with a tolerance.
    """
    image = np.asarray(result.image)
    return {
        "exclusion": array_digest(result.exclusion),
        "image_dtype": image.dtype.str,
        "image_shape": list(image.shape),
        "image_values": image.ravel().tolist(),
        "provenance": json.loads(provenance_json(result.provenance)),
    }


def frozen_output(preset: str, result: Any) -> dict[str, Any]:
    """Return what the table freezes for one conversion: digests, or values for custom_od."""
    if stain.PRESETS[preset].method == "custom_od":
        return output_values(result)
    return output_digests(result)


def outcome(call: Callable[[], object]) -> dict[str, str]:
    """Return the exception type and full message of ``call``, or ``accepted``."""
    try:
        call()
    except Exception as error:  # noqa: BLE001 - every outcome is recorded, none hidden
        return {"type": type(error).__name__, "message": str(error)}
    return {"type": "accepted", "message": ""}


def _skimage_version() -> str | None:
    """Return the installed scikit-image version, or ``None``."""
    try:
        import skimage
    except ImportError:
        return None
    return skimage.__version__


def record() -> dict[str, Any]:
    """Return the parity table of the installed ``stain`` module."""
    import warnings

    def refused(preset: str, options: dict[str, Any]) -> dict[str, str]:
        pixels, axes = valid_pixels(preset)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return outcome(lambda: stain.to_scalar(pixels, axes=axes, preset=preset, **options))

    converted = {}
    for index, (preset, pixels, axes, options) in enumerate(conversions()):
        result = stain.to_scalar(pixels, axes=axes, preset=preset, **options)
        converted[conversion_id(index, preset, axes, options)] = frozen_output(preset, result)
    return {
        "produced_by": (
            f"tests/stain_parity_cases.py on commit {BASE_COMMIT} (v0.6.0), to_scalar with "
            "valid pixels; see the module docstring"
        ),
        "numpy": np.__version__,
        "skimage": _skimage_version(),
        "refusals": {name: refused(*case) for name, case in REFUSALS.items()},
        "changed": {name: refused(*case) for name, case in CHANGED.items()},
        "conversions": converted,
    }


if __name__ == "__main__":
    print(stain.__file__, file=sys.stderr)
    print(json.dumps(record(), indent=1, sort_keys=True))
