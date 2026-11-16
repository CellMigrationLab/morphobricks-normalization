"""Tables (results.md) and plots (PNG) from the measurements (not for merge).

report.py IO.json ACCURACY.json OUT_FOLDER

Reads the per-plane statistics from WORK/<dataset>/prepared.pkl.
"""

from __future__ import annotations

import json
import pickle
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import WORK  # noqa: E402

DATASETS = ("study", "cmu1", "hpmec", "jump")
NAMES = {
    "study": "study (synthetic uint16)",
    "cmu1": "CMU-1 (uint8 RGB)",
    "hpmec": "HPMEC (uint16)",
    "jump": "JUMP (uint16)",
}
FORMATS = ("csv_dense", "csv_sparse", "parquet_long", "npz_trimmed", "zarr_trimmed")
LABEL = {
    "csv_dense": "(a) dense CSV",
    "csv_sparse": "(b) sparse cell CSV",
    "parquet_long": "(b) long Parquet",
    "npz_trimmed": "(c) trimmed .npz",
    "zarr_trimmed": "(c) trimmed zarr",
}
# Validated default categorical palette (dataviz skill, light mode), fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


TRACED = {}


def kb(n):
    return f"{n / 1024:,.1f}"


def ms(row):
    return f"{row['median_s'] * 1000:,.1f} ({row['min_s'] * 1000:,.1f}–{row['max_s'] * 1000:,.1f})"


def io_tables(io):
    by = {(r["dataset"], r["variant"], r["format"], r["op"]): r for r in io}
    lines = []
    for ds in DATASETS:
        variants = sorted({r["variant"] for r in io if r["dataset"] == ds},
                          key=["exact", "d4096", "d4096r", "d1024", "d1024r"].index)
        lines += [f"\n### {NAMES[ds]}\n",
                  "| content | format | total KiB | KiB / channel plane | × smaller than (a) exact "
                  "| write ms | read ms | one file per image, write ms, 1 → 4 threads "
                  "| one file per image, read ms, 1 → 4 threads | peak RSS rise MiB write / read | traced peak MiB write / read |",
                  "|---|---|---:|---:|---:|---|---|---|---|---|---|"]
        base = by[(ds, "exact", "csv_dense", "write")]["bytes"]
        for v in variants:
            for f in FORMATS:
                w, r = by[(ds, v, f, "write")], by[(ds, v, f, "read")]
                s1, s4 = by[(ds, v, f, "write_sh1")], by[(ds, v, f, "write_sh4")]
                r1, r4 = by[(ds, v, f, "read_sh1")], by[(ds, v, f, "read_sh4")]
                lines.append(
                    f"| {v} | {LABEL[f]} | {kb(w['bytes'])} | {kb(w['bytes'] / w['planes'])} | "
                    f"{base / w['bytes']:,.0f} | {ms(w)} | {ms(r)} | "
                    f"{s1['median_s'] * 1000:,.0f} → {s4['median_s'] * 1000:,.0f} | "
                    f"{r1['median_s'] * 1000:,.0f} → {r4['median_s'] * 1000:,.0f} | "
                    f"{w['peak_rss_rise'] / 2**20:,.1f} / {r['peak_rss_rise'] / 2**20:,.1f} | "
                    f"{TRACED.get((ds, v, f), '–')} |"
                )
    return lines


def data_table():
    lines = ["| dataset | images | channel planes | dtype | pixels / plane | range width "
             "(max−min+1), median (min–max) | distinct values / plane, median (min–max) "
             "| occupied share of the 65,536 (or 256) bins, median |",
             "|---|---:|---:|---|---:|---|---|---:|"]
    stats_by = {}
    for ds in DATASETS:
        with open(WORK / ds / "prepared.pkl", "rb") as fh:
            p = pickle.load(fh)
        st = p["stats"]
        stats_by[ds] = st
        n_images = len({s["plane_id"].split("::")[0] for s in st})
        widths = [s["width"] for s in st]
        distinct = [s["distinct"] for s in st]
        bins = 256 if st[0]["dtype"] == "uint8" else 65536
        px = sorted({s["pixels"] for s in st})
        lines.append(
            f"| {NAMES[ds]} | {n_images} | {len(st)} | {st[0]['dtype']} | "
            f"{', '.join(f'{x:,}' for x in px)} | "
            f"{statistics.median(widths):,.0f} ({min(widths):,}–{max(widths):,}) | "
            f"{statistics.median(distinct):,.0f} ({min(distinct):,}–{max(distinct):,}) | "
            f"{statistics.median(distinct) / bins:.1%} |"
        )
    return lines, stats_by


def accuracy_tables(acc):
    ident = ["| dataset | format | records equal (every field, counts, tails as bits) "
             "| numbers compared | bit differences |", "|---|---|---|---:|---:|"]
    for ds in DATASETS:
        for f, v in acc[ds]["identity"].items():
            ident.append(f"| {NAMES[ds]} | {LABEL[f]} | {'yes' if not v['record_differences'] else 'NO'} "
                         f"| {v['values']:,} | {v['bit_differences']} |")
    err = ["| dataset | variant | exact channel planes | max representative error "
           "| percentile 1–99.8 offset / scale, max error in scale units (plane; pooled) "
           "| robust z-score offset / scale (plane; pooled) | z-score offset / scale (plane; pooled) "
           "| normalized px, max abs diff: percentile clipped / robust z (|x|≤10) / z (|x|≤10) "
           "| px whose clip state changes (percentile, all scopes) |",
           "|---|---|---:|---:|---|---|---|---|---:|"]
    dec = ["| dataset | variant | groups flagged approximate | fits refused only here "
           "| intensity warnings new / lost | saturated fractions made blank "
           "| image medians / P1 / P99.8 changed | narrow-range warnings | tails: count, mean / std "
           "vs exact (ulps), robust limits = exact quantile |",
           "|---|---|---:|---|---|---:|---|---:|---|"]
    for ds in DATASETS:
        for v, e in acc[ds]["errors"].items():
            pp, po, px = e["plane_parameters"], e["pooled_parameters"], e["normalized_pixels"]

            def pooled_max(method, part):
                vals = [x["max_rel_scale"] for k, x in po.items() if k.endswith(f":{method}.{part}")]
                return max(vals) if vals else 0.0

            def pair(method):
                return (f"{pp[method + '.offset']['max_rel_scale']:.2g} / "
                        f"{pp[method + '.scale']['max_rel_scale']:.2g}; "
                        f"{pooled_max(method, 'offset'):.2g} / {pooled_max(method, 'scale'):.2g}")

            def pxmax(method, key):
                return max(x[key] for k, x in px.items() if k.endswith("/" + method))

            clip = sum(x["clip_changed"] for k, x in px.items() if k.endswith("/percentile"))
            pixels = sum(x["pixels"] for k, x in px.items() if k.endswith("/percentile"))
            err.append(
                f"| {NAMES[ds]} | {v} | {e['exact_channel_planes']} / {e['channel_planes']} | "
                f"{e['max_representative_error']:,.4g} | {pair('percentile')} | {pair('robust_zscore')} | "
                f"{pair('zscore')} | {pxmax('percentile', 'max_abs_clipped'):.2g} / "
                f"{pxmax('robust_zscore', 'max_abs_core'):.2g} / {pxmax('zscore', 'max_abs_core'):.2g} | "
                f"{clip:,} of {pixels:,} |"
            )
            s = e["summaries"]
            issues = e["intensity_issues"]
            new = "; ".join(f"{c} {ch} {it}" for c, ch, it in issues["new"]) or "–"
            lost = "; ".join(f"{c} {ch} {it}" for c, ch, it in issues["lost"]) or "–"
            refused = "; ".join(f"{k[2]} {k[1]}" for k in e["fits_refused_only_here"]) or "–"
            t = e["tails"]
            tails = (f"{t['tails']}, {t['mean_ulps']} / {t['std_ulps']}, "
                     f"{t['limit_matches_exact_quantile']} of {t['limits']}") if t else "–"
            dec.append(
                f"| {NAMES[ds]} | {v} | {e['groups_flagged_approximate']} of {e['groups']} | {refused} | "
                f"new: {new}; lost: {lost} | {s['saturated_fraction']['to_blank']} of "
                f"{s['saturated_fraction']['n']} | {s['median']['changed']} / {s['p_low']['changed']} / "
                f"{s['p_high']['changed']} of {s['median']['n']} | {e['narrow_range_warnings']} | {tails} |"
            )
    return ident, err, dec


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_sizes(io, out):
    by = {(r["dataset"], r["variant"], r["format"]): r for r in io if r["op"] == "write"}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True, facecolor=SURFACE)
    for ax, ds in zip(axes, DATASETS, strict=True):
        style(ax)
        variants = [v for v in ("exact", "d4096", "d1024") if (ds, v, "csv_dense") in by]
        ys, labels = [], []
        y = 0
        for v in variants:
            for k, f in enumerate(FORMATS):
                r = by[(ds, v, f)]
                ax.barh(y, r["bytes"] / r["planes"] / 1024, color=SERIES[k], height=0.8,
                        label=LABEL[f] if (ds == "study" and v == "exact") else None)
                y += 1
            ys.append(y - len(FORMATS) / 2 - 0.5)
            labels.append(v)
            y += 1
        ax.set_xscale("log")
        ax.set_yticks(ys, labels)
        ax.invert_yaxis()
        ax.set_title(NAMES[ds], fontsize=9, color=INK, loc="left")
        ax.set_xlabel("KiB per channel plane (log)", fontsize=8, color=INK2)
    fig.legend(loc="upper center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "size_per_plane.png", dpi=130, facecolor=SURFACE)


def plot_distinct(stats_by, out):
    fig, ax = plt.subplots(figsize=(6.2, 4.2), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
    for k, ds in enumerate(DATASETS):
        st = stats_by[ds]
        ax.scatter([s["width"] for s in st], [s["distinct"] for s in st], s=22, color=SERIES[k],
                   edgecolor=SURFACE, linewidth=0.8, label=NAMES[ds], zorder=3)
    lim = (100, 70000)
    ax.plot(lim, lim, color=INK2, linewidth=0.8, linestyle="--", zorder=2)
    ax.text(30000, 22000, "distinct = range width", fontsize=7, color=INK2, ha="right")
    for y, text in ((65536, "65,536 bins (dense CSV, uint16)"), (4096, "4,096"), (1024, "1,024")):
        ax.axhline(y, color="#c9c8c3", linewidth=0.8, zorder=1)
        ax.text(110, y * 1.08, text, fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(100, 120000)
    ax.set_xlabel("range width per channel plane, max − min + 1 (log)", fontsize=8, color=INK2)
    ax.set_ylabel("distinct values used (log)", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "distinct_values.png", dpi=130, facecolor=SURFACE)


def main():
    io = json.loads(Path(sys.argv[1]).read_text())
    traced = Path(sys.argv[1]).with_name("memory_traced.json")
    if traced.exists():
        for r in json.loads(traced.read_text()):
            TRACED[(r["dataset"], r["variant"], r["format"])] = (
                f"{r['write_peak'] / 2**20:,.1f} / {r['read_peak'] / 2**20:,.1f}"
            )
    acc = json.loads(Path(sys.argv[2]).read_text())
    out = Path(sys.argv[3])
    data, stats_by = data_table()
    ident, err, dec = accuracy_tables(acc)
    text = ["# Results (generated by report.py)\n", "## Data\n", *data,
            "\n## Sizes, times and memory\n", *io_tables(io),
            "\n## Identity of (b) and (c)\n", *ident,
            "\n## Error of (d) against exact\n", *err,
            "\n## Decisions that change with (d)\n", *dec]
    (out / "results.md").write_text("\n".join(text) + "\n")
    plot_sizes(io, out)
    plot_distinct(stats_by, out)


if __name__ == "__main__":
    main()
