"""Inputs of the v0.6.1 region parity table of tests/test_regions.py (MorphoBricks-Spec ADR 0005).

``tests/data/region_v061_parity.json`` freezes what ``region_white``, ``region_vectors``,
``stain.white_from_region`` and ``stain.stain_vectors_from_regions`` of v0.6.1 (commit
``b8aa65b``) returned for these non-empty boolean regions, every float as :meth:`float.hex`.
ADR 0005 requires the values of a non-empty region to stay bitwise equal when the functions
move to ``check_region`` and ``RegionEstimate``. This module reads a result through ``.value``
when it has one, so the same cases run on v0.6.1 (tuples) and after ADR 0005 (estimates). To
record the table again::

    git worktree add /tmp/v061 b8aa65b
    PYTHONPATH=/tmp/v061/src <venv>/bin/python tests/region_parity_cases.py \\
        > tests/data/region_v061_parity.json
    git worktree remove /tmp/v061

The script prints the path of the ``optical_density`` module it used on standard error: check
that it is the worktree's.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from morphobricks_normalization import optical_density as od
from morphobricks_normalization import stain

BASE_COMMIT = "b8aa65b"
SEED = 20261115
WHITES = ((240.0, 235.0, 245.0), (255.0, 255.0, 255.0), (231.5, 228.25, 239.0))
SHAPES = ((1, 1), (5, 7), (16, 16), (33, 20))


def _regions(rng: np.random.Generator, shape: tuple[int, int]) -> Iterator[np.ndarray]:
    """Yield non-empty boolean regions of ``shape``: all, one pixel, a block, random pixels."""
    yield np.ones(shape, bool)
    one = np.zeros(shape, bool)
    one[rng.integers(shape[0]), rng.integers(shape[1])] = True
    yield one
    block = np.zeros(shape, bool)
    block[: max(1, shape[0] // 2), : max(1, shape[1] // 3)] = True
    yield block
    for density in (0.1, 0.5, 0.9):
        scattered = rng.random(shape) < density
        scattered.flat[0] = True
        yield scattered


def white_cases() -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(case ID, (Y, X, 3) uint8 image, region)`` whose medians are in (0, 255)."""
    rng = np.random.default_rng(SEED)
    for shape in SHAPES:
        image = rng.integers(1, 255, (*shape, 3)).astype(np.uint8)
        for index, region in enumerate(_regions(rng, shape)):
            yield f"white_{shape[0]}x{shape[1]}_{index}", image, region


def vector_cases() -> Iterator[tuple[str, np.ndarray, tuple[float, ...], dict[str, np.ndarray]]]:
    """Yield ``(case ID, image, white, regions)`` whose regions hold no 0 and are not background.

    Pixel values stay below 228, so each channel's optical density against every white of
    :data:`WHITES` is above 0.0001 and the mean's norm above :data:`od.MIN_REGION_OD`.
    """
    rng = np.random.default_rng(SEED + 1)
    for shape in SHAPES:
        image = rng.integers(1, 200, (*shape, 3)).astype(np.uint8)
        regions = list(_regions(rng, shape))
        for w, white in enumerate(WHITES):
            for index in range(len(regions)):
                chosen = {
                    "first": regions[index],
                    "second": regions[(index + 1) % len(regions)],
                    "third": ~regions[index] if (~regions[index]).any() else regions[0],
                }
                yield f"vectors_{shape[0]}x{shape[1]}_w{w}_{index}", image, white, chosen


def _hex(values: Any) -> list[str]:
    """Return the floats of a result (a tuple, or an estimate's ``.value``) as hex strings."""
    values = getattr(values, "value", values)
    return [float(v).hex() for v in values]


def _vectors_hex(result: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return a mapping name -> vector (tuple or estimate) as name -> hex strings."""
    return {name: _hex(value) for name, value in result.items()}


def outputs() -> dict[str, Any]:
    """Return every output of the installed module for the cases, floats as hex."""
    table: dict[str, Any] = {}
    for case, image, region in white_cases():
        cyx = np.ascontiguousarray(np.moveaxis(image, 2, 0))
        table[case] = {
            "region_white": _hex(od.region_white(image, region)),
            "white_from_region_yxc": _hex(
                stain.white_from_region(image, axes="YXC", region=region)
            ),
            "white_from_region_cyx": _hex(stain.white_from_region(cyx, axes="CYX", region=region)),
        }
    for case, image, white, regions in vector_cases():
        cyx = np.ascontiguousarray(np.moveaxis(image, 2, 0))
        white_array = np.asarray(white, dtype=np.float64)
        table[case] = {
            "region_vectors": _vectors_hex(od.region_vectors(image, white_array, regions)),
            "stain_vectors_from_regions_yxc": _vectors_hex(
                stain.stain_vectors_from_regions(image, axes="YXC", white=white, regions=regions)
            ),
            "stain_vectors_from_regions_cyx": _vectors_hex(
                stain.stain_vectors_from_regions(cyx, axes="CYX", white=white, regions=regions)
            ),
        }
    return table


def digest(table: Mapping[str, Any]) -> str:
    """Return the SHA-256 of the table's canonical JSON."""
    text = json.dumps(table, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record() -> dict[str, Any]:
    """Return the parity table of the installed module."""
    table = outputs()
    return {
        "produced_by": (
            f"tests/region_parity_cases.py on commit {BASE_COMMIT} (v0.6.1); see the module "
            "docstring"
        ),
        "numpy": np.__version__,
        "sha256": digest(table),
        "outputs": table,
    }


if __name__ == "__main__":
    print(od.__file__, file=sys.stderr)
    print(json.dumps(record(), indent=1, sort_keys=True))
