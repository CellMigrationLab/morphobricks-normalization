"""Compute every histogram variant of a dataset once and pickle it for the other scripts.

prepare.py DATASET [STUDY_CSV]

Writes WORK/DATASET/prepared.pkl with, per variant (common.VARIANTS), the plane records and
range checks, plus 6 seeded 256 x 256 patches per image (all channels) for the normalized-pixel
comparison, and per channel plane: dtype, minimum, maximum, the width of the range
(max - min + 1) and the number of distinct values (the occupied bins of the exact histogram).

With STUDY_CSV (the plane_histograms.csv of a real preprocessing run of the study data), the
exact records are written with this repository's writer and compared byte for byte with it.
"""

from __future__ import annotations

import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from common import VARIANTS, WORK, channel_ids, iter_images, records_of
from morphobricks_normalization import write_plane_histograms

PATCH = 256
PATCHES_PER_IMAGE = 6


def main() -> None:
    dataset = sys.argv[1]
    out = WORK / dataset
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261116)
    result = {v: {"records": [], "checks": [], "seconds": 0.0} for v in VARIANTS}
    patches, stats = [], []
    for image, data in iter_images(dataset):
        integer16 = data.dtype.itemsize >= 2
        for variant in VARIANTS:
            if variant != "exact" and not integer16:
                continue  # uint8: 256 bins already; 1,024 / 4,096 bins do not apply
            t0 = time.perf_counter()
            records, checks = records_of(image, data, variant)
            result[variant]["seconds"] += time.perf_counter() - t0
            result[variant]["records"] += records
            result[variant]["checks"] += checks
        for t in range(data.shape[0]):
            for c, channel in enumerate(image.channels):
                plane = data[t, c]
                stats.append(
                    {
                        "plane_id": f"{image.rel}::t={t}",
                        "channel": channel,
                        "dtype": plane.dtype.name,
                        "min": int(plane.min()),
                        "max": int(plane.max()),
                        "width": int(plane.max()) - int(plane.min()) + 1,
                        "distinct": int(np.unique(plane).size),
                        "pixels": int(plane.size),
                    }
                )
        for _ in range(PATCHES_PER_IMAGE):
            t = int(rng.integers(data.shape[0]))
            y = int(rng.integers(data.shape[2] - PATCH))
            x = int(rng.integers(data.shape[3] - PATCH))
            patches.append((image.rel, t, image.channels, data[t, :, y : y + PATCH, x : x + PATCH].copy()))
        print(f"{dataset}: {image.rel} {data.dtype} {data.shape}", flush=True)
        del data
    result = {v: r for v, r in result.items() if r["records"]}
    payload = {"variants": result, "patches": patches, "stats": stats}
    with open(out / "prepared.pkl", "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    for v, r in result.items():
        print(f"{v}: {len(r['records'])} records, histograms {r['seconds']:.2f} s")
    if len(sys.argv) > 2:
        records = result["exact"]["records"]
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            path = Path(tmp) / "plane_histograms.csv"
            write_plane_histograms(records, path, channel_ids=channel_ids(records))
            same = path.read_bytes() == Path(sys.argv[2]).read_bytes()
        print(f"exact records written == preprocessing's plane_histograms.csv: {same}")
        if not same:
            sys.exit(1)


if __name__ == "__main__":
    main()
