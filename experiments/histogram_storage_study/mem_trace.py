"""Traced peak memory of one write and one read per format (not for merge).

mem_trace.py OUT.json [DATASET ...]

``tracemalloc`` peak (Python objects and NumPy buffers) of one call, after a warm-up call, for
the exact and 4,096-bin min-max histograms. pyarrow allocates outside tracemalloc: for the
Parquet format its memory pool's peak (``max_memory``) is added. Deterministic, unlike RSS.
"""

from __future__ import annotations

import gc
import json
import pickle
import shutil
import sys
import tracemalloc
from pathlib import Path

from common import WORK, channel_ids
from formats import FORMATS, READERS, SUFFIX, WRITERS


def peak(fn):
    import pyarrow as pa

    fn()  # warm-up
    gc.collect()
    pool = pa.default_memory_pool()
    pool.release_unused()
    arrow_before = pool.max_memory()
    tracemalloc.start()
    out = fn()
    traced = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    del out
    return traced + max(pool.max_memory() - arrow_before, 0)


def main():
    out = []
    for ds in sys.argv[2:] or ["study", "cmu1", "hpmec", "jump"]:
        with open(WORK / ds / "prepared.pkl", "rb") as fh:
            variants = pickle.load(fh)["variants"]
        for v in ("exact", "d4096"):
            if v not in variants:
                continue
            records = variants[v]["records"]
            ids = channel_ids(records)
            for f in FORMATS:
                path = WORK / f"mem{SUFFIX[f]}"

                def write():
                    if path.is_dir():
                        shutil.rmtree(path)
                    WRITERS[f](records, path, ids)

                w = peak(write)
                r = peak(lambda: READERS[f](path))
                shutil.rmtree(path) if path.is_dir() else path.unlink()
                out.append({"dataset": ds, "variant": v, "format": f, "write_peak": w, "read_peak": r})
                print(ds, v, f, f"{w / 2**20:.1f} / {r / 2**20:.1f} MiB", flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
