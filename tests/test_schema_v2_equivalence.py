"""Schema version 3 holds exactly what version 2 held (contract §1.3, plan §11).

The fixture in ``tests/data`` was written by the Milestone 2 code (commit
``02a0620``, ``make_v2_fixture.py``): simulated pixels, the version-2
``plane_histograms.csv`` tables computed from them (one row per plane and
channel), and the parameters that code fitted at every scope. Here the same
pixels go through the current code; every histogram value must be the same
text, channel by channel, and every parameter the same number.
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
# Version-2 column -> version-3 column of channel c.
CHANNEL_COLUMNS = {
    "pixel_min": "Intensity_MinIntensity_C{c}",
    "pixel_max": "Intensity_MaxIntensity_C{c}",
    "valid_pixel_count": "Histogram_ValidPixelCount_C{c}",
    "binning_min": "Histogram_BinningMin_C{c}",
    "binning_max": "Histogram_BinningMax_C{c}",
}
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
        assert row["schema_version"] == "3" and row["channel_count"] == "2"
        for c in range(2):
            key = (row["source_relative_path"], str(c), row["t"])
            before = old[key]
            seen.add(key)
            assert row["plane_id"] == before["plane_id"].replace(f",c={c},", ",")
            for column in SHARED_COLUMNS:
                assert row[column] == before[column], column
            for v2, v3 in CHANNEL_COLUMNS.items():
                assert row[v3.format(c=c)] == before[v2], v3
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
            assert parameters.offset == tuple(entry["offset"]), (entry["settings"], group)
            assert parameters.scale == tuple(entry["scale"]), (entry["settings"], group)
            assert parameters.exact == entry["exact"]
        checked += 1
    assert checked == len(fits) == 200
