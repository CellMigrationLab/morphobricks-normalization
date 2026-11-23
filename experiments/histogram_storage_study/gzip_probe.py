"""Size and time of gzip on the two CSV layouts (exact histograms), stdlib only (not for merge).

gzip_probe.py OUT.json [DATASET ...]   (5 repeats; median, min, max)
"""

import gzip
import json
import pickle
import statistics
import sys
import time
from pathlib import Path

from common import WORK, channel_ids
from formats import WRITERS

out = []
for ds in sys.argv[2:] or ["study", "hpmec", "jump"]:
    with open(WORK / ds / "prepared.pkl", "rb") as fh:
        records = pickle.load(fh)["variants"]["exact"]["records"]
    for fmt in ("csv_dense", "csv_sparse"):
        path = WORK / "probe.csv"
        WRITERS[fmt](records, path, channel_ids(records))
        raw = path.read_bytes()
        path.unlink()
        for level in (1, 6):
            times, back = [], []
            for _ in range(5):
                t = time.perf_counter()
                z = gzip.compress(raw, compresslevel=level, mtime=0)
                times.append(time.perf_counter() - t)
                t = time.perf_counter()
                gzip.decompress(z)
                back.append(time.perf_counter() - t)
            row = {"dataset": ds, "format": fmt, "level": level, "raw": len(raw), "gz": len(z),
                   "compress_s": [statistics.median(times), min(times), max(times)],
                   "decompress_s": [statistics.median(back), min(back), max(back)]}
            out.append(row)
            print(ds, fmt, level, len(raw), len(z), [round(x * 1000, 1) for x in row["compress_s"]],
                  [round(x * 1000, 1) for x in row["decompress_s"]], flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
