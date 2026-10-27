"""Write the schema-version-2 fixture of ``test_schema_v2_equivalence.py``.

Run once, from the repository root, with the code of Milestone 2 (commit
``02a0620``, before the channels moved into columns):

    git worktree add ../v2 02a0620
    PYTHONPATH=../v2/src python tests/data/make_v2_fixture.py

It writes, next to this script:

- ``v2_pixels.npz``: the pixels of every simulated plane and channel;
- ``v2_<dtype>.csv``: ``plane_histograms.csv`` of schema version 2 (one row per
  plane and channel) written by that version's writer;
- ``v2_parameters.json``: the planes' labels, and the parameters that version's
  ``fit`` gives for every method, scope, pooling weight and time-point option.

The fixture is committed, so the test compares the current code with outputs
of the earlier code rather than with itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from morphobricks_normalization import (  # noqa: E402
    NormalizationSettings,
    NoValidPixelsError,
    PlaneRecord,
    compute_plane_histogram,
    fit,
    group_name,
    scope_groups,
    write_plane_histograms,
)
from morphobricks_normalization.csv_io import SCHEMA_VERSION  # noqa: E402
from simulation import simulate_design  # noqa: E402

# One table per pixel type: binned 16-bit planes (some exact, some not) and
# float planes, one of whose channels has no finite pixels at one time point.
DATASETS = {"uint16": (np.uint16, 512), "float32": (np.float32, 64)}
METHODS = [
    {"method": "minmax"},
    {"method": "percentile"},
    {"method": "percentile", "percentile": [0.5, 95.0]},
    {"method": "zscore"},
    {"method": "robust_zscore"},
]
FILE_SCOPES = [
    {"scope": "biological_repeat", "pooling": "pixels"},
    {"scope": "biological_repeat", "pooling": "conditions"},
    {"scope": "image"},
]


def simulate(dtype):
    planes = simulate_design(seed=41, n_tech=2, n_images=1, n_times=2, shape=(24, 24), dtype=dtype)
    # Unequal conditions in one repeat group, so the two pooling weights differ.
    planes = [p for p in planes if not p[1].startswith("treated/2024_05_01/tech_rep2/")]
    if np.dtype(dtype).kind == "f":
        labels, path, c, t, pixels = planes[3]
        assert (c, t) == (1, 1)
        planes[3] = (labels, path, c, t, np.full_like(pixels, np.nan))
    return planes


def outcome(statistics, settings, **kwargs):
    try:
        parameters = fit(statistics, settings, **kwargs)
    except NoValidPixelsError:
        return {"error": "NoValidPixelsError"}
    return {
        "offset": list(parameters.offset),
        "scale": list(parameters.scale),
        "exact": parameters.exact,
    }


def main():
    assert SCHEMA_VERSION == "2", "Run this script with the code of schema version 2."
    arrays, fixture = {}, {"datasets": {}}
    for name, (dtype, bin_count) in DATASETS.items():
        planes = simulate(dtype)
        records = [
            PlaneRecord(
                path, *labels, histogram=compute_plane_histogram(pixels, bin_count), c=c, t=t
            )
            for labels, path, c, t, pixels in planes
        ]
        table = write_plane_histograms(records, HERE / f"v2_{name}.csv")
        # The csv module ends lines with CRLF; the repository keeps text files LF.
        table.write_bytes(table.read_bytes().replace(b"\r\n", b"\n"))
        entries = []
        for k, (labels, path, c, t, pixels) in enumerate(planes):
            arrays[f"{name}_{k:03d}"] = pixels
            entries.append({"labels": list(labels), "path": path, "c": c, "t": t})
        results = []
        for method in METHODS:
            for scope in FILE_SCOPES:
                for per_time_point in (False, True):
                    data = method | scope | {"per_time_point": per_time_point}
                    settings = NormalizationSettings.from_dict(data)
                    groups, _ = scope_groups(records, settings)
                    for key, members in groups.items():
                        group = group_name(key)
                        result = outcome(members, settings, group=group)
                        results.append({"settings": data, "group": group} | result)
            # Patch scope: the pixels of one image at t=0, channels on axis 0.
            settings = NormalizationSettings.from_dict(method | {"scope": "patch"})
            for path in sorted({path for _, path, _, _, _ in planes}):
                stack = np.stack([p for _, q, _, t, p in planes if q == path and t == 0])
                result = outcome(stack, settings, group=path, channel_axis=0)
                results.append({"settings": method | {"scope": "patch"}, "group": path} | result)
        fixture["datasets"][name] = {"bin_count": bin_count, "planes": entries, "fits": results}
    np.savez_compressed(HERE / "v2_pixels.npz", **arrays)
    text = json.dumps(fixture, indent=1)
    (HERE / "v2_parameters.json").write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
