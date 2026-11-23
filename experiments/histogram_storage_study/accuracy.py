"""Parameters, summaries and decisions from each representation, against exact (not for merge).

accuracy.py OUT.json [DATASET ...]

Everything is computed with this repository's functions: ``PooledHistogram`` (and its
``summary``), ``min_max``, ``percentile_range``, ``z_score``, ``robust_z_score``,
``rgb_percentile`` through ``fit``, ``scope_groups``, ``image_summaries``, ``intensity_checks``,
``range_check_issues`` and ``Parameters.apply``.

1. Identity (options b and c): the exact records are written in every format, read back with
   its converter, and every parameter below is compared with the one from today's CSV as raw
   float64 bits. Any difference is a failure (exit status 1).
2. Error (option d): the same parameters from the 4,096- and 1,024-bin histograms (min-max and
   robust range) against exact, as absolute error and in units of the exact scale (the error
   of a normalized value it causes); normalized pixels of the sampled patches (max absolute
   difference, also over the pixels whose exact normalized value is within [-10, 10]; with
   clipping to [0, 1], pixels whose clipped state changes); and the decisions
   that change (intensity-check warnings, saturated fractions, exactness flags, range checks).

Parameters
    per channel plane: min-max, percentile 1-99.8, z-score, robust z-score (offset and scale),
    and the summary percentiles 0.1 ... 99.9, median, MAD, mean, std;
    pooled: ``fit`` at the image scope and the biological-repeat scope (pooling by pixels and
    by conditions), and over the whole dataset, for every method the image type allows.
"""

from __future__ import annotations

import json
import pickle
import shutil
import struct
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from common import WORK, channel_ids, image_type
from formats import FORMATS, READERS, SUFFIX, WRITERS, same_records
from morphobricks_normalization import (
    NormalizationSettings,
    PooledHistogram,
    image_summaries,
    intensity_checks,
    min_max,
    percentile_range,
    range_check_issues,
    robust_z_score,
    z_score,
)
from morphobricks_normalization.fitting import fit
from morphobricks_normalization.scopes import scope_groups

FLUO_METHODS = ("minmax", "percentile", "zscore", "robust_zscore")
CLIPPED = ("minmax", "percentile")


def bits(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", float(x)))[0]


def _settings(dataset, method, scope, pooling="pixels"):
    rgb = image_type(dataset) == "rgb"
    return NormalizationSettings(
        image_type="rgb" if rgb else "fluorescence",
        method=method,
        scope=scope,
        pooling=pooling if scope == "biological_repeat" else "pixels",
    )


def _methods(dataset):
    return ("percentile",) if image_type(dataset) == "rgb" else FLUO_METHODS


def plane_parameters(records, dataset):
    """Per channel plane: {(plane_id, channel, name): value}, with the scale of each method."""
    out = {}
    rgb = image_type(dataset) == "rgb"
    for r in records:
        for c in r.channels:
            h = r.channel_histogram(c)
            if h.is_empty:
                continue
            p = PooledHistogram([h])
            key = (r.plane_id, c)
            s = p.summary()
            for q, v in s.percentiles.items():
                out[(*key, f"P{q:g}")] = (v, None)
            out[(*key, "median")] = (s.median, None)
            out[(*key, "mad")] = (s.mad, None)
            out[(*key, "mean")] = (s.mean, None)
            out[(*key, "std")] = (s.std, None)
            if rgb:
                out[(*key, "linearP1")] = (p.linear_percentile(1.0), None)
                out[(*key, "linearP99.8")] = (p.linear_percentile(99.8), None)
                continue
            for name, fn in (
                ("minmax", min_max),
                ("percentile", percentile_range),
                ("zscore", z_score),
                ("robust_zscore", robust_z_score),
            ):
                try:
                    prm = fn(p)
                except ValueError:
                    continue
                out[(*key, f"{name}.offset")] = (prm.offset, prm.scale)
                out[(*key, f"{name}.scale")] = (prm.scale, prm.scale)
    return out


def pooled_parameters(records, dataset):
    """``fit`` of every group: {(scope, group, method, channel, offset|scale): value}."""
    out, fitted = {}, {}
    scopes = [("image", "pixels"), ("biological_repeat", "pixels"), ("biological_repeat", "conditions")]
    for method in _methods(dataset):
        for scope, pooling in scopes:
            settings = _settings(dataset, method, scope, pooling)
            groups, _ = scope_groups(records, settings)
            for gkey, rows in groups.items():
                gname = "|".join(map(str, gkey))
                try:
                    prm = fit(rows, settings, group=gname)
                except ValueError as error:  # a refusal is a decision: recorded, compared
                    fitted[(scope, pooling, gname, method)] = None
                    out[(f"{scope}/{pooling}", gname, method, "*", "refused")] = (str(error), None)
                    continue
                fitted[(scope, pooling, gname, method)] = prm
                for c, o, s in zip(prm.channels, prm.offset, prm.scale, strict=True):
                    out[(f"{scope}/{pooling}", gname, method, c, "offset")] = (o, s)
                    out[(f"{scope}/{pooling}", gname, method, c, "scale")] = (s, s)
                out[(f"{scope}/{pooling}", gname, method, "*", "exact")] = (float(prm.exact), None)
        # the whole dataset as one group (scope label biological_repeat, every row)
        settings = _settings(dataset, method, "biological_repeat")
        try:
            prm = fit(records, settings, group="ALL")
        except ValueError as error:
            fitted[("all", "pixels", "ALL", method)] = None
            out[("all", "ALL", method, "*", "refused")] = (str(error), None)
            continue
        fitted[("all", "pixels", "ALL", method)] = prm
        for c, o, s in zip(prm.channels, prm.offset, prm.scale, strict=True):
            out[("all", "ALL", method, c, "offset")] = (o, s)
            out[("all", "ALL", method, c, "scale")] = (s, s)
    return out, fitted


def summaries_and_checks(records):
    sums = image_summaries(records)
    values = {}
    for s in sums:
        for k, c in enumerate(s.channels):
            for name in ("median", "p_low", "p_high", "saturated_fraction", "exact"):
                values[(s.source_relative_path, c, name)] = getattr(s, name)[k]
    issues = sorted(
        (i.code, i.channel or "", "|".join(map(str, i.items))) for i in intensity_checks(sums)
    )
    return values, issues


def everything(records, dataset):
    plane = plane_parameters(records, dataset)
    pooled, fitted = pooled_parameters(records, dataset)
    summ, issues = summaries_and_checks(records)
    return plane, pooled, fitted, summ, issues


def identity(dataset, records, base):
    """Write/read the exact records in every format; compare records and every number's bits."""
    out = {}
    ids = channel_ids(records)
    folder = WORK / "acc_tmp"
    folder.mkdir(parents=True, exist_ok=True)
    for fmt in FORMATS:
        path = folder / f"{dataset}{SUFFIX[fmt]}"
        WRITERS[fmt](records, path, ids)
        back = READERS[fmt](path)
        shutil.rmtree(path) if path.is_dir() else path.unlink()
        diffs = same_records(records, back)
        plane, pooled, _, summ, issues = everything(back, dataset)
        n_values, n_diff = 0, 0
        for ref, got in ((base[0], plane), (base[1], pooled)):
            if ref.keys() != got.keys():
                n_diff += 1
            for k, (v, _) in ref.items():
                n_values += 1
                n_diff += bits(v) != bits(got[k][0])
        for k, v in base[3].items():
            n_values += 1
            w = summ[k]
            n_diff += (v is None) != (w is None) or (v is not None and bits(v) != bits(w))
        n_diff += issues != base[4]
        out[fmt] = {"record_differences": len(diffs), "values": n_values, "bit_differences": n_diff}
        print(f"  identity {dataset} {fmt}: {out[fmt]}", flush=True)
    return out


def _err_table(ref, got):
    """Absolute and scale-relative errors per parameter name."""
    rows = {}
    for k, (v, scale) in ref.items():
        if k not in got or isinstance(v, str) or isinstance(got[k][0], str):
            continue
        w = got[k][0]
        name = k[-1] if len(k) == 3 else f"{k[0]}:{k[2]}.{k[4]}"
        if name.endswith("exact"):
            continue
        e = abs(w - v)
        rel = e / scale if scale else None
        d = rows.setdefault(name, {"n": 0, "max_abs": 0.0, "max_rel_scale": 0.0, "changed": 0})
        d["n"] += 1
        d["changed"] += bits(v) != bits(w)
        d["max_abs"] = max(d["max_abs"], e)
        if rel is not None:
            d["max_rel_scale"] = max(d["max_rel_scale"], rel)
    return rows


def _apply_pair(patches, ref, got, method, agg):
    """Normalize the patches with exact (ref) and coarse (got) parameters; update ``agg``."""
    for _, _, chans, patch in patches:
        a = ref.apply(patch, channel_axis=0, channels=list(chans)).astype(np.float64)
        b = got.apply(patch, channel_axis=0, channels=list(chans)).astype(np.float64)
        agg["pixels"] += a.size
        agg["max_abs"] = max(agg["max_abs"], float(np.abs(a - b).max()))
        core = np.abs(a) <= 10  # pixels within 10 units of the exact normalized scale
        if core.any():
            agg["max_abs_core"] = max(agg["max_abs_core"], float(np.abs(a - b)[core].max()))
        if method in CLIPPED:
            ca, cb = np.clip(a, 0, 1), np.clip(b, 0, 1)
            # clipped state: +1 clipped up to 0, -1 clipped down to 1, 0 inside [0, 1]
            agg["clip_changed"] += int((np.sign(ca - a) != np.sign(cb - b)).sum())
            agg["max_abs_clipped"] = max(agg["max_abs_clipped"], float(np.abs(ca - cb).max()))


def _tail_errors(exact_records, coarse_records):
    """Robust range: tails' mean/std against the exact values from the exact histogram (ulps)."""
    worst = {"tails": 0, "mean_ulps": 0, "std_ulps": 0, "limit_matches_exact_quantile": 0, "limits": 0}
    for re_, rc in zip(exact_records, coarse_records, strict=True):
        for c in rc.channels:
            he, hc = re_.channel_histogram(c), rc.channel_histogram(c)
            if hc.is_exact or he.is_empty:
                continue
            values = he.binning_min + np.arange(he.bin_count, dtype=np.int64)
            pe = PooledHistogram([he])
            for limit, q in ((hc.binning_min, 0.01), (hc.binning_max, 99.99)):
                worst["limits"] += 1
                worst["limit_matches_exact_quantile"] += float(limit) == pe.percentile(q)
            for tail, sel in (
                (hc.underflow, values < hc.binning_min),
                (hc.overflow, values > hc.binning_max),
            ):
                if tail.is_empty:
                    continue
                v, n = values[sel], he.counts[sel]
                count = int(n.sum())
                total = sum(int(a) * int(b) for a, b in zip(v, n, strict=True) if b)
                mean = Fraction(total, count)
                var = sum(Fraction(int(b)) * (int(a) - mean) ** 2 for a, b in zip(v, n, strict=True) if b) / count
                exact_mean = float(mean)
                exact_std = float(np.sqrt(float(var)))
                worst["tails"] += 1
                worst["mean_ulps"] = max(worst["mean_ulps"], abs(bits(tail.mean) - bits(exact_mean)))
                worst["std_ulps"] = max(worst["std_ulps"], abs(bits(tail.std) - bits(exact_std)))
                assert tail.count == count
    return worst


def errors(dataset, prepared, base):
    exact = prepared["variants"]["exact"]["records"]
    patches = prepared["patches"]
    rep = {r.source_relative_path: r.biological_repeat for r in exact}
    out = {}
    for variant, data in prepared["variants"].items():
        if variant == "exact":
            continue
        records = data["records"]
        plane, pooled, fitted, summ, issues = everything(records, dataset)
        apply = {}
        for key, ref in base[2].items():
            scope, pooling, gname, method = key
            if ref is None or fitted[key] is None:
                continue
            if scope == "image":
                sel = [p for p in patches if p[0] == gname]
            elif scope == "biological_repeat":
                sel = [p for p in patches if rep[p[0]] == gname]
            else:
                sel = patches
            agg = apply.setdefault(
                f"{scope}/{pooling}/{method}",
                {"pixels": 0, "max_abs": 0.0, "max_abs_core": 0.0, "clip_changed": 0,
                 "max_abs_clipped": 0.0},
            )
            _apply_pair(sel, ref, fitted[key], method, agg)
        summ_changes = {}
        for k, v in base[3].items():
            w = summ[k]
            name = k[-1]
            d = summ_changes.setdefault(name, {"n": 0, "changed": 0, "to_blank": 0, "max_abs": 0.0})
            d["n"] += 1
            if (v is None) != (w is None):
                d["changed"] += 1
                d["to_blank"] += w is None
            elif v is not None and bits(v) != bits(w):
                d["changed"] += 1
                d["max_abs"] = max(d["max_abs"], abs(float(w) - float(v)))
        ref_issues, got_issues = set(base[4]), set(issues)
        flags = [k for k in pooled if k[-1] == "exact"]
        refused_ref = {k for k in base[1] if k[-1] == "refused"}
        refused_got = {k for k in pooled if k[-1] == "refused"}
        narrow = range_check_issues(data["checks"])
        max_err = max(
            r.channel_histogram(c).max_representative_error for r in records for c in r.channels
        )
        out[variant] = {
            "plane_parameters": _err_table(base[0], plane),
            "pooled_parameters": _err_table(base[1], pooled),
            "normalized_pixels": apply,
            "summaries": summ_changes,
            "intensity_issues": {
                "exact": len(ref_issues),
                "variant": len(got_issues),
                "lost": sorted(ref_issues - got_issues),
                "new": sorted(got_issues - ref_issues),
            },
            "fits_refused_only_here": [
                [*k[:3], pooled[k][0][:160]] for k in sorted(refused_got - refused_ref)
            ],
            "fits_refused_only_exact": [list(k[:3]) for k in sorted(refused_ref - refused_got)],
            "groups_flagged_approximate": sum(pooled[k][0] == 0.0 for k in flags),
            "groups": len(flags),
            "exact_channel_planes": sum(
                r.channel_histogram(c).is_exact for r in records for c in r.channels
            ),
            "channel_planes": sum(len(r.channels) for r in records),
            "max_representative_error": max_err,
            "narrow_range_warnings": len(narrow),
            "narrow_range_planes": sum(1 for _, _, ch in data["checks"] if ch is not None and ch.narrow),
            "tails": _tail_errors(exact, records) if variant.endswith("r") else None,
        }
        print(f"  error {dataset} {variant}: done", flush=True)
    return out


def main() -> None:
    out_json = Path(sys.argv[1])
    datasets = sys.argv[2:] or ["study", "cmu1", "hpmec", "jump"]
    results = json.loads(out_json.read_text()) if out_json.exists() else {}
    for dataset in datasets:
        with open(WORK / dataset / "prepared.pkl", "rb") as fh:
            prepared = pickle.load(fh)
        exact = prepared["variants"]["exact"]["records"]
        base = everything(exact, dataset)
        print(f"{dataset}: {len(base[0])} plane values, {len(base[1])} pooled values, "
              f"{len(base[4])} intensity issues", flush=True)
        results[dataset] = {
            "identity": identity(dataset, exact, base),
            "errors": errors(dataset, prepared, base),
            "exact_issues": base[4],
        }
        out_json.write_text(json.dumps(results, indent=1, default=str))
    bad = [
        (d, f) for d, r in results.items() for f, v in r["identity"].items()
        if v["record_differences"] or v["bit_differences"]
    ]
    print("identity failures:", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
