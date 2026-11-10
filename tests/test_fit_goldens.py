"""Fitted parameters of simulated planes against numbers stored in ``tests/data`` (science golden).

``golden_pixels.npz`` holds simulated planes (uint16 and float32, two channels, two time
points, one float channel without finite pixels at one time point) and
``golden_parameters.json`` the offset, scale and ``exact`` flag that the Milestone 2 code
(commit ``02a0620``) fitted from them for every method, scope, pooling weight and time-point
option (200 fits per pixel type). The numbers were computed by that earlier code, not by this
one, so the test fails if a change of the histograms, the pooling or the formulas moves any
fitted parameter.

Here the same pixels go through the current code (histograms, the ``plane_histograms.csv``
writer and reader, scope groups, ``fit``) and every parameter must equal the stored number.

Parameters built from a mean or standard deviation are sums over the pooled values
(``np.dot``, weighted ``np.bincount``), whose last bits depend on the NumPy build and the
processor (SIMD, pairwise summation). Against the stored numbers they are compared within a
few units in the last place; every other method is compared exactly.
"""

import functools
import json
from pathlib import Path

import numpy as np
import pytest

from morphobricks_normalization import (
    NormalizationSettings,
    NoValidPixelsError,
    PlaneRecord,
    compute_plane_histogram,
    fit,
    group_name,
    read_plane_histograms,
    scope_groups,
    write_plane_histograms,
)

DATA = Path(__file__).parent / "data"
DATASETS = ("uint16", "float32")
# Methods whose offset and scale are sums over the pooled values (see above);
# every other method takes them from single representative values.
SUMMED_METHODS = {"zscore"}
MAX_ULP = 4


@functools.cache
def golden():
    return json.loads((DATA / "golden_parameters.json").read_text(encoding="utf-8"))


@functools.cache
def planes(name):
    """``[(labels, path, c, t, pixels), ...]`` as the golden numbers were computed from."""
    pixels = np.load(DATA / "golden_pixels.npz")
    return [
        (tuple(p["labels"]), p["path"], p["c"], p["t"], pixels[f"{name}_{k:03d}"])
        for k, p in enumerate(golden()["datasets"][name]["planes"])
    ]


@functools.cache
def table_records(name, folder):
    """Write the golden pixels as a ``plane_histograms.csv`` and read its records back."""
    bin_count = golden()["datasets"][name]["bin_count"]
    by_plane = {}
    for labels, path, c, t, pixels in planes(name):
        _, channels = by_plane.setdefault((path, t), (labels, {}))
        channels[c] = compute_plane_histogram(pixels, bin_count)
    records = [
        PlaneRecord(
            path,
            *labels,
            histograms=[channels[c] for c in sorted(channels)],
            t=t,
            position_id=path.rsplit("/", 1)[-1].rsplit(".", 1)[0],
        )
        for (path, t), (labels, channels) in by_plane.items()
    ]
    table = write_plane_histograms(records, Path(folder) / f"golden_{name}.csv")
    return read_plane_histograms(table)


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return str(tmp_path_factory.mktemp("golden"))


@pytest.mark.parametrize("name", DATASETS)
def test_fitted_parameters_equal_the_golden_numbers_at_every_scope(name, folder):
    records = table_records(name, folder)
    fits = golden()["datasets"][name]["fits"]
    groups_by_settings = {}
    checked = 0
    for entry in fits:
        settings = NormalizationSettings.from_dict(entry["settings"])
        group = entry["group"]
        if settings.scope == "patch":
            stack = np.stack([p for _, path, c, t, p in planes(name) if path == group and t == 0])
            statistics, options = stack, {"channel_axis": 0}
        else:
            key = json.dumps(entry["settings"], sort_keys=True)
            if key not in groups_by_settings:
                groups, _ = scope_groups(records, settings)
                groups_by_settings[key] = {group_name(k): v for k, v in groups.items()}
                expected = {e["group"] for e in fits if e["settings"] == entry["settings"]}
                assert set(groups_by_settings[key]) == expected
            statistics, options = groups_by_settings[key][group], {}
        if "error" in entry:
            with pytest.raises(NoValidPixelsError):
                fit(statistics, settings, group=group, **options)
        else:
            parameters = fit(statistics, settings, group=group, **options)
            where = (entry["settings"], group)
            if settings.method in SUMMED_METHODS:
                for value, stored in (("offset", entry["offset"]), ("scale", entry["scale"])):
                    np.testing.assert_array_max_ulp(
                        np.array(getattr(parameters, value)), np.array(stored), maxulp=MAX_ULP
                    )
            else:
                assert parameters.offset == tuple(entry["offset"]), where
                assert parameters.scale == tuple(entry["scale"]), where
            assert parameters.exact == entry["exact"], where
        checked += 1
    assert checked == len(fits) == 200
