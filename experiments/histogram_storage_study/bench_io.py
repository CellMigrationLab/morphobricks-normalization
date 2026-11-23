"""Write and read times, peak memory and sizes of every format (not for merge).

bench_io.py OUT.json [REPEATS] [DATASET ...]

For every dataset, histogram variant and format, each operation runs in a fresh Python process
(``--op``), one at a time:

``write``     all records to one file (the final table)
``read``      that file back to ``PlaneRecord`` objects (every PlaneHistogram invariant checked)
``write_sh1`` / ``read_sh1``   one file per image (preprocessing's staging shards), serially
``write_sh4`` / ``read_sh4``   the same with 4 threads (image workers share one process)

In the child, the input records (or the written file) are ready and the format's modules are
imported before anything is measured. After one untimed warm-up call (lazy library loads), the
peak RSS is reset (``/proc/self/clear_refs``) and one call measures peak memory: the rise of
the peak RSS (``VmHWM``) over the RSS just before it, which includes pyarrow's and zarr's own
allocators. Then REPEATS timed calls (default 5); the median, minimum and
maximum are reported. The bytes, the file count and the per-channel-plane bytes come from the
written file.
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPS = ("write", "read", "write_sh1", "read_sh1", "write_sh4", "read_sh4")


def _status(field: str) -> int:
    """VmRSS or VmHWM of this process, in bytes (Linux)."""
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    raise KeyError(field)


def _reset_peak() -> None:
    """Reset the peak RSS (VmHWM) to the current RSS (Linux >= 4.0)."""
    with open("/proc/self/clear_refs", "w") as fh:
        fh.write("5")


def _shards(records):
    by_image = {}
    for r in records:
        by_image.setdefault(r.source_relative_path, []).append(r)
    return list(by_image.values())


def child(records_pkl: str, fmt: str, op: str, folder: str, repeats: int) -> None:
    from common import channel_ids
    from formats import READERS, SUFFIX, WRITERS

    if fmt == "parquet_long":
        import pyarrow.parquet  # noqa: F401
    if fmt == "zarr_trimmed":
        import zarr  # noqa: F401
    with open(records_pkl, "rb") as fh:
        records = pickle.load(fh)
    ids = channel_ids(records)
    folder = Path(folder)
    one = folder / f"table{SUFFIX[fmt]}"
    shards = _shards(records)
    shard_paths = [folder / f"shard{k:03d}{SUFFIX[fmt]}" for k in range(len(shards))]
    write, read = WRITERS[fmt], READERS[fmt]
    if op.startswith("read"):
        if op == "read":
            write(records, one, ids)
        else:
            for s, p in zip(shards, shard_paths, strict=True):
                write(s, p, ids)
        del records, shards
    workers = 4 if op.endswith("sh4") else 1

    def run():
        if op == "write":
            write(records, one, ids)
        elif op == "read":
            return read(one)
        elif op.startswith("write_sh"):
            with ThreadPoolExecutor(workers) as pool:
                list(pool.map(lambda a: write(a[0], a[1], ids), zip(shards, shard_paths)))
        else:
            with ThreadPoolExecutor(workers) as pool:
                return list(pool.map(read, shard_paths))

    def clear():
        if op.startswith("write"):
            for p in [one, *shard_paths]:
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.exists():
                    p.unlink()

    run()  # warm-up: lazy library loads and allocator arenas are not the operation's
    clear()
    gc.collect()
    _reset_peak()
    before = _status("VmRSS")
    out = run()
    peak = _status("VmHWM") - before
    del out
    times = []
    for _ in range(repeats):
        clear()
        gc.collect()
        t0 = time.perf_counter()
        out = run()
        times.append(time.perf_counter() - t0)
        del out
    print(json.dumps({"times": times, "peak_rss_rise": max(peak, 0)}))


def main() -> None:
    from common import WORK, channel_ids
    from formats import FORMATS, SUFFIX, WRITERS, size_of

    out_json = Path(sys.argv[1])
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    datasets = sys.argv[3:] or ["study", "cmu1", "hpmec", "jump"]
    results = json.loads(out_json.read_text()) if out_json.exists() else []
    done = {(r["dataset"], r["variant"], r["format"], r["op"]) for r in results}
    for dataset in datasets:
        with open(WORK / dataset / "prepared.pkl", "rb") as fh:
            prepared = pickle.load(fh)
        for variant, data in prepared["variants"].items():
            records = data["records"]
            pkl = WORK / dataset / f"records_{variant}.pkl"
            with open(pkl, "wb") as fh:
                pickle.dump(records, fh, protocol=pickle.HIGHEST_PROTOCOL)
            planes = sum(len(r.channels) for r in records)
            ids = channel_ids(records)
            for fmt in FORMATS:
                folder = WORK / "io" / f"{dataset}_{variant}_{fmt}"
                shutil.rmtree(folder, ignore_errors=True)
                folder.mkdir(parents=True)
                path = folder / f"size{SUFFIX[fmt]}"
                WRITERS[fmt](records, path, ids)
                nbytes, nfiles = size_of(path)
                shard_bytes = 0
                for k, shard in enumerate(_shards(records)):
                    p = folder / f"sizeshard{k}{SUFFIX[fmt]}"
                    WRITERS[fmt](shard, p, ids)
                    shard_bytes += size_of(p)[0]
                shutil.rmtree(folder)
                for op in OPS:
                    key = (dataset, variant, fmt, op)
                    if key in done:
                        continue
                    folder.mkdir(parents=True)
                    proc = subprocess.run(
                        [sys.executable, __file__, "--op", str(pkl), fmt, op, str(folder),
                         str(repeats if not op.endswith(("sh1", "sh4")) else 3)],
                        capture_output=True, text=True, check=False,
                        env={**os.environ, "PYTHONPATH": str(HERE)},
                    )
                    shutil.rmtree(folder)
                    if proc.returncode:
                        print(proc.stderr, file=sys.stderr)
                        raise SystemExit(f"child failed: {key}")
                    child_out = json.loads(proc.stdout.strip().splitlines()[-1])
                    t = child_out["times"]
                    row = {
                        "dataset": dataset, "variant": variant, "format": fmt, "op": op,
                        "planes": planes, "records": len(records), "bytes": nbytes,
                        "files": nfiles, "shard_bytes": shard_bytes,
                        "median_s": statistics.median(t), "min_s": min(t), "max_s": max(t),
                        "times": t, "peak_rss_rise": child_out["peak_rss_rise"],
                        "load": os.getloadavg()[0],
                    }
                    results.append(row)
                    out_json.write_text(json.dumps(results, indent=1))
                    print(
                        f"{dataset:6s} {variant:7s} {fmt:13s} {op:10s} {nbytes:>10d} B "
                        f"{row['median_s']:.4f} s ({row['min_s']:.4f}-{row['max_s']:.4f}) "
                        f"peak +{row['peak_rss_rise'] / 2**20:.1f} MiB load {row['load']:.1f}",
                        flush=True,
                    )
            pkl.unlink()


if __name__ == "__main__":
    if sys.argv[1] == "--op":
        child(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]))
    else:
        main()
