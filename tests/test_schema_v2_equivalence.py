"""Schema version 3 holds exactly what version 2 held (contract §1.3, plan §11).

Since Milestone 5 the writer writes schema version 4: version 3 with empty
underflow and overflow columns for the default min-max range (contract §1.4).
The tables written here are therefore version 4, and the parameters below also
show that a version-3-equivalent table (no pixel outside the bins) gives the
parameters it gave before.

The fixture in ``tests/data`` was written by the Milestone 2 code (commit
``02a0620``, ``make_v2_fixture.py``): simulated pixels, the version-2
``plane_histograms.csv`` tables computed from them (one row per plane and
channel), and the parameters that code fitted at every scope. Here the same
pixels go through the current code; every histogram value must be the same
text, channel by channel, and every parameter the same number.

Parameters built from a mean or standard deviation are sums over the pooled
values (``np.dot``, weighted ``np.bincount``), whose last bits depend on the
NumPy build and the processor (SIMD, pairwise summation). Against the stored
fixture they are compared within a few units in the last place; within one
process, version-2 rows and version-3 rows must give them exactly.
"""

import csv
import functools
import json
from pathlib import Path

import numpy as np
import pytest

from morphobricks_normalization import (
    NormalizationSettings,
    NoValidPixelsError,
    PlaneHistogram,
    PlaneRecord,
    compute_plane_histogram,
    fit,
    group_name,
    read_plane_histograms,
    scope_groups,
    write_plane_histograms,
)
from morphobricks_normalization.csv_io import SCHEMA_VERSION
from morphobricks_normalization.dtypes import check_supported, is_integer

DATA = Path(__file__).parent / "data"
DATASETS = ("uint16", "float32")
# Version-2 column -> version-3 column of channel c.
CHANNEL_COLUMNS = {
    "pixel_min": "Intensity_MinIntensity_C{c}",
    "pixel_max": "Intensity_MaxIntensity_C{c}",
    "valid_pixel_count": "Histogram_ValidPixelCount_C{c}",
    "binning_min": "Histogram_BinningMin_C{c}",
    "binning_max": "Histogram_BinningMax_C{c}",
}
# Methods whose offset and scale are sums over the pooled values (see above);
# every other method takes them from single representative values.
SUMMED_METHODS = {"zscore"}
MAX_ULP = 4
SHARED_COLUMNS = (
    "position_id",
    "image_file",
    "z",
    "t",
    "condition",
    "biological_repeat",
    "technical_repeat",
    "source_relative_path",
    "dtype",
    "bit_depth",
    "total_pixel_count",
    "bin_count",
)


@functools.cache
def fixture():
    return json.loads((DATA / "v2_parameters.json").read_text(encoding="utf-8"))


@functools.cache
def planes(name):
    """``[(labels, path, c, t, pixels), ...]`` as the fixture was written from."""
    pixels = np.load(DATA / "v2_pixels.npz")
    return [
        (tuple(p["labels"]), p["path"], p["c"], p["t"], pixels[f"{name}_{k:03d}"])
        for k, p in enumerate(fixture()["datasets"][name]["planes"])
    ]


@functools.cache
def version_3_table(name, folder):
    """Write the fixture's pixels as a version-3 table; return its path and records."""
    bin_count = fixture()["datasets"][name]["bin_count"]
    by_plane = {}
    for labels, path, c, t, pixels in planes(name):
        _, channels = by_plane.setdefault((path, t), (labels, {}))
        channels[c] = compute_plane_histogram(pixels, bin_count)
    records = [
        PlaneRecord(path, *labels, histograms=[channels[c] for c in sorted(channels)], t=t)
        for (path, t), (labels, channels) in by_plane.items()
    ]
    table = write_plane_histograms(records, Path(folder) / f"v3_{name}.csv")
    return table, read_plane_histograms(table)


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@functools.cache
def version_2_records(name):
    """The rows of the version-2 table: one single-channel record per (path, t, c)."""
    records = {}
    for row in read_rows(DATA / f"v2_{name}.csv"):
        dtype = check_supported(row["dtype"])
        number = int if is_integer(dtype) else float
        bin_count = int(row["bin_count"])
        histogram = PlaneHistogram(
            dtype=dtype,
            bin_count=bin_count,
            binning_min=number(row["binning_min"]) if row["binning_min"] else None,
            binning_max=number(row["binning_max"]) if row["binning_max"] else None,
            counts=np.array([row[f"histogram_bin_{i:05d}"] for i in range(bin_count)], np.int64),
            total_pixel_count=int(row["total_pixel_count"]),
            valid_pixel_count=int(row["valid_pixel_count"]),
        )
        labels = (row["condition"], row["biological_repeat"], row["technical_repeat"])
        key = (row["source_relative_path"], int(row["t"]), int(row["c"]))
        records[key] = PlaneRecord(key[0], *labels, histogram, z=int(row["z"]), t=key[1])
    return records


def fit_or_none(statistics, settings, **options):
    try:
        return fit(statistics, settings, **options)
    except NoValidPixelsError:
        return None


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return str(tmp_path_factory.mktemp("v3"))


@pytest.mark.parametrize("name", DATASETS)
def test_version_3_holds_the_histograms_of_version_2_channel_by_channel(name, folder):
    table, _ = version_3_table(name, folder)
    old = {
        (r["source_relative_path"], r["c"], r["t"]): r for r in read_rows(DATA / f"v2_{name}.csv")
    }
    new = read_rows(table)
    assert len(new) * 2 == len(old)  # two channels per plane
    bin_count = int(next(iter(old.values()))["bin_count"])
    seen = set()
    for row in new:
        assert row["schema_version"] == SCHEMA_VERSION and row["channel_count"] == "2"
        for c in range(2):
            key = (row["source_relative_path"], str(c), row["t"])
            before = old[key]
            seen.add(key)
            assert row["plane_id"] == before["plane_id"].replace(f",c={c},", ",")
            for column in SHARED_COLUMNS:
                assert row[column] == before[column], column
            for v2, v3 in CHANNEL_COLUMNS.items():
                assert row[v3.format(c=c)] == before[v2], v3
            for tail in ("Underflow", "Overflow"):  # min-max range: no tails (schema 4)
                assert row[f"Histogram_{tail}Count_C{c}"] == "0"
                assert row[f"Histogram_{tail}Sum_C{c}"] == "0.0"
                assert row[f"Histogram_{tail}SumOfSquares_C{c}"] == "0.0"
            counts = [row[f"Histogram_Count_C{c}_{i:05d}"] for i in range(bin_count)]
            assert counts == [before[f"histogram_bin_{i:05d}"] for i in range(bin_count)]
    assert seen == set(old)


@pytest.mark.parametrize("name", DATASETS)
def test_version_3_gives_the_parameters_of_version_2_at_every_scope(name, folder):
    _, records = version_3_table(name, folder)
    fits = fixture()["datasets"][name]["fits"]
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


@pytest.mark.parametrize("name", DATASETS)
def test_version_2_rows_and_version_3_rows_give_identical_parameters(name, folder):
    # Same process, same NumPy: fitting each channel from the version-2 table's
    # one-channel rows, as Milestone 2 did, and the whole planes of version 3
    # must agree to the last bit for every method, including the z-score.
    _, records = version_3_table(name, folder)
    old = version_2_records(name)
    settings_seen = {
        json.dumps(e["settings"], sort_keys=True)
        for e in fixture()["datasets"][name]["fits"]
        if e["settings"]["scope"] != "patch"
    }
    compared = 0
    for text in sorted(settings_seen):
        settings = NormalizationSettings.from_dict(json.loads(text))
        groups, _ = scope_groups(records, settings)
        for key, members in groups.items():
            group = group_name(key)
            new = fit_or_none(members, settings, group=group)
            per_channel = [
                fit_or_none([old[r.source_relative_path, r.t, c] for r in members], settings)
                for c in range(2)
            ]
            if None in per_channel:
                assert new is None, (text, group)
                continue
            assert new.offset == tuple(p.offset[0] for p in per_channel), (text, group)
            assert new.scale == tuple(p.scale[0] for p in per_channel), (text, group)
            assert new.exact == all(p.exact for p in per_channel)
            compared += 1
    assert compared > 100
