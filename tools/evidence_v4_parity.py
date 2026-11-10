"""DRAFT EVIDENCE ONLY (Stage 3 S3.2 pull request): delete before merge.

Shows that for planes whose channels are all present, schema version 5 gives the histograms
and fits of schema version 4 for the same pixels, bit for bit. It is not a test and not a
compatibility path (ADR 0002 AR10, ADR 0003 HV4): it runs once against the base release and
once against this change, in two environments, and the two JSON outputs are compared.

It builds the homogeneous Spec fixture ``v13_homogeneous_legacy`` (3 conditions x 3 repeats,
channels ``DNA``, ``ACTIN``; read from ``tests/spec_fixtures`` with ``tomllib``) and, to make
the ID order differ from the index order, the same tree with the channels listed
``["DNA", "ACTIN", "TUBULIN"]``. For each it writes ``plane_histograms.csv`` with the installed
library, reads it back and fits every method x scope x pooling weight x per-time-point option.
Results are keyed by channel ID: under version 4 the C index is mapped through the unit's
confirmed list (the same for every unit, so the map is exact). Floats are ``float.hex``.

Usage::

    python tools/evidence_v4_parity.py OUT.json   # in each environment, then compare the files
"""

from __future__ import annotations

import csv
import inspect
import json
import sys
import tempfile
import tomllib
import zlib
from pathlib import Path

import numpy as np

import morphobricks_normalization as mn

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/spec_fixtures/fixtures/valid/v13_homogeneous_legacy.toml"
V5 = "channels" in inspect.signature(mn.PlaneRecord).parameters
METHODS = ("minmax", "percentile", "zscore", "robust_zscore")


def units(extra: bool) -> list[tuple[str, str, tuple[str, ...]]]:
    """Return ``(condition, repeat, channels)`` of every unit of v13 (plus TUBULIN)."""
    data = tomllib.loads(FIXTURE.read_text(encoding="utf-8"))
    found = []
    for _, repeat in sorted(data["repeats"].items()):
        for condition, member in sorted(repeat["members"].items()):
            for tech in member["technical_repeats"].values():
                ids = tuple(tech["channels"]) + (("TUBULIN",) if extra else ())
                found.append((condition, member["source_repeat"], ids))
    return found


def build(extra: bool) -> tuple[list, tuple[str, ...]]:
    """Return the records of the tree and its channel list (the same in every unit)."""
    records, ids = [], None
    for condition, repeat, channels in units(extra):
        ids = channels
        for i in range(3):
            path = f"{condition}/{repeat}/img_{i}.tif"
            for t in range(2):
                hists = []
                for c in channels:
                    rng = np.random.default_rng(zlib.crc32(f"{path}|{t}|{c}".encode()))
                    level = {"DNA": 1.0, "ACTIN": 2.5, "TUBULIN": 0.6}[c]
                    pixels = rng.gamma(4.0, 300.0 * level, (24, 24)).astype(np.uint16)
                    hists.append(mn.compute_plane_histogram(pixels, 1024))
                extra_kw = {"channels": channels} if V5 else {}
                records.append(
                    mn.PlaneRecord(
                        path, condition, repeat, "single", hists, t=t, position_id=f"img_{i}",
                        **extra_kw,
                    )
                )  # fmt: skip
    return records, ids


def histogram_cells(path: Path, ids: tuple[str, ...]) -> dict[str, list[str]]:
    """Return each row's per-channel cells, keyed by plane and channel ID."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    cells = {}
    for row in rows:
        for k, channel in enumerate(ids):
            token = channel if V5 else f"C{k}"
            names = [n for n in row if n.endswith(f"_{token}") or f"_{token}_" in n]
            names = [n for n in names if n.split("_")[0] in ("Intensity", "Histogram")]
            cells[f"{row['plane_id']}|{channel}"] = [row[n] for n in sorted(names)]
    return cells


def fits(records: list, ids: tuple[str, ...]) -> dict[str, list]:
    """Return every fit, keyed by settings, group and channel ID."""
    out = {}
    for method in METHODS:
        for scope in ("biological_repeat", "image"):
            for pooling in ("pixels", "conditions"):
                for per_t in (False, True):
                    settings = mn.NormalizationSettings(
                        method=method, scope=scope, pooling=pooling, per_time_point=per_t
                    )
                    groups, _ = mn.scope_groups(records, settings)
                    for key, members in groups.items():
                        p = mn.fit(members, settings, group=mn.group_name(key))
                        order = p.channels if V5 else [ids[c] for c in p.channels]
                        for channel, o, s in zip(order, p.offset, p.scale, strict=True):
                            name = f"{method}|{scope}|{pooling}|{per_t}|{mn.group_name(key)}"
                            out[f"{name}|{channel}"] = [o.hex(), s.hex(), p.exact]
    return out


def main() -> int:
    """Write the evidence JSON named on the command line."""
    result = {}
    with tempfile.TemporaryDirectory() as folder:
        for extra in (False, True):
            records, ids = build(extra)
            path = Path(folder) / f"t{extra}.csv"
            kw = {"channel_ids": ids} if V5 else {}
            mn.write_plane_histograms(records, path, **kw)
            back = mn.read_plane_histograms(path)
            result[f"extra={extra}"] = {
                "cells": histogram_cells(path, ids),
                "fits": fits(back, ids),
            }
    Path(sys.argv[1]).write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    n = sum(len(v["fits"]) for v in result.values())
    print(f"{'v5' if V5 else 'v4'}: {n} fitted channels written to {sys.argv[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
