"""AUDIT TOOL ONLY: throwaway reference validator for MorphoBricks experiment description v1.

This is not a runtime package and no brick may import it. It implements the normative text of
docs/experiment-v1.md (phases 1-5 of section 6 and the warnings), the canonical writer of
section 9.1, the section digests of section 9.3 and the predecessor checks of sections 4.4a,
4.6 and 4.9 that two TOML files can express. The records file of section 4.6 is `refrecords.py`;
`fixtures` runs its goldens too. Where the text is ambiguous the choice made here is
marked "AMBIGUITY" and reported in docs/audits/matrix-audit-2026-11-07.md.

Usage:
    python -I refval.py check FILE [--json]
    python -I refval.py fixtures SPEC_ROOT          # run the fixture catalogue
    python -I refval.py transition OLD NEW
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
import tomllib
import unicodedata
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script folder
import refrecords  # noqa: E402  (records of section 4.6; a sibling audit module)

SENTINELS = {"", "na", "nan", "none", "null", "n/a", "<na>"}
TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
RESERVED = {
    "whole", "cell", "nucleus", "cytoplasm", "ring", "expanded_nucleus",
    "condition", "biological_repeat", "repeat", "technical_repeat", "position_id",
    "image_file", "image_path", "mask_file", "patch_file", "patch_id", "label_id",
    "z", "z0", "z1", "c", "t",
}
FORBIDDEN = {
    "axes", "shared", "plane_scope", "z", "projection", "pixel_size", "calibration", "fovs",
    "files", "paths", "overrides", "exclude", "fingerprints", "bricks", "routes", "features",
    "normalization", "statistics", "contrasts", "binary_mask_policy", "mask_conversion",
    "control", "roles", "experiment", "source_index", "channel_count", "retired_repeats",
    "replaced_by", "labchronicle", "provenance", "source_database",
}
TOP = ("schema", "schema_version", "conditions", "channels", "repeats", "masks", "used_ids")
COND = ("source_folder", "display_name", "reference", "target", "treatment_key")
CHAN = ("display_name", "header_names")
REP = ("display_name", "confirmed", "confirmed_on", "members")
MEM = ("source_repeat", "image_count", "technical_repeats")
UNIT = ("source_folder", "display_name", "channels", "image_type")
MASK = ("source_type", "representation", "source_channels", "display_name")
USED = ("conditions", "channels", "repeats", "technical_repeats")
REPRESENTATIONS = ("instances", "binary")


class Report:
    def __init__(self) -> None:
        self.errors: list[tuple[str, str, object]] = []
        self.warnings: list[tuple[str, str, object]] = []
        self.phase = 0

    def e(self, code: str, path: str, value: object = None) -> None:
        self.errors.append((code, path, value))

    def w(self, code: str, path: str, value: object = None) -> None:
        self.warnings.append((code, path, value))

    @property
    def codes(self) -> set[str]:
        return {c for c, _, _ in self.errors}

    @property
    def warning_codes(self) -> set[str]:
        return {c for c, _, _ in self.warnings}

    def as_dict(self) -> dict:
        return {
            "phase": self.phase,
            "errors": [list(map(str, x)) for x in self.errors],
            "warnings": [list(map(str, x)) for x in self.warnings],
            "error_codes": sorted(self.codes),
            "warning_codes": sorted(self.warning_codes),
        }


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def is_text(x: object) -> bool:
    return type(x) is str


def is_table(x: object) -> bool:
    return type(x) is dict


def kpath(base: str, key: str) -> str:
    return f"{base}.{key}" if re.fullmatch(r"[A-Za-z0-9_-]+", key) else f'{base}."{key}"'


# ------------------------------------------------------------------ phase 1: decode
def phase1(data: bytes, r: Report):
    if data.startswith(b"\xef\xbb\xbf"):
        r.e("E_FILE_ENCODING", "<file>", "byte-order mark")
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        r.e("E_FILE_ENCODING", "<file>", str(exc))
        return None
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        msg = str(exc)
        dup = ("Cannot declare" in msg) or ("Cannot overwrite" in msg) or ("twice" in msg) or ("Duplicate" in msg)
        r.e("E_DUPLICATE_KEY" if dup else "E_TOML_SYNTAX", "<file>", msg)
        return None


# ------------------------------------------------------------------ phase 2: header
def phase2(doc: dict, r: Report) -> None:
    if "schema" not in doc:
        r.e("E_SCHEMA_ID", "schema")
    elif not is_text(doc["schema"]):
        r.e("E_TYPE", "schema", doc["schema"])
    elif doc["schema"] != "morphobricks-experiment":
        r.e("E_SCHEMA_ID", "schema", doc["schema"])
    if "schema_version" not in doc:
        r.e("E_SCHEMA_VERSION_MISSING", "schema_version")
    elif type(doc["schema_version"]) is not int:
        r.e("E_TYPE", "schema_version", doc["schema_version"])
    elif doc["schema_version"] != 1:
        r.e("E_SCHEMA_VERSION_UNSUPPORTED", "schema_version", doc["schema_version"])


# ------------------------------------------------------------------ phase 3: structure
def _fields(tbl: dict, allowed, path: str, r: Report) -> None:
    for k in tbl:
        if k in FORBIDDEN:
            r.e("E_FORBIDDEN_FIELD", kpath(path, k), k)
        elif k not in allowed:
            r.e("E_UNKNOWN_FIELD", kpath(path, k), k)


def _required(tbl: dict, keys, path: str, r: Report) -> None:
    for k in keys:
        if k not in tbl:
            r.e("E_MISSING_FIELD", kpath(path, k))


def _text(tbl: dict, k: str, path: str, r: Report, nonempty: bool = False) -> None:
    if k not in tbl:
        return
    v = tbl[k]
    if not is_text(v):
        r.e("E_TYPE", kpath(path, k), v)
    elif nonempty and v.strip() == "":
        r.e("E_TEXT_EMPTY", kpath(path, k), v)


def _text_array(tbl: dict, k: str, path: str, r: Report, nonempty_items: bool = False,
                min1: bool = False) -> None:
    if k not in tbl:
        return
    v = tbl[k]
    p = kpath(path, k)
    if type(v) is not list:
        r.e("E_TYPE", p, v)
        return
    if min1 and not v:
        r.e("E_EMPTY", p)
    for i, x in enumerate(v):
        if not is_text(x):
            r.e("E_TYPE", f"{p}[{i}]", x)
        elif nonempty_items and x.strip() == "":
            r.e("E_TEXT_EMPTY", f"{p}[{i}]", x)


def _id_table(parent: dict, k: str, path: str, r: Report, required: bool = True,
              min1: bool = True):
    """Return [(id, table)] of an ID-keyed table; forbidden ID keys are reported and skipped."""
    p = kpath(path, k)
    if k not in parent:
        if required:
            r.e("E_MISSING_FIELD", p)
        return []
    t = parent[k]
    if not is_table(t):
        r.e("E_TYPE", p, t)
        return []
    if min1 and not t:
        r.e("E_EMPTY", p)
    out = []
    for key, v in t.items():
        if key in FORBIDDEN:
            r.e("E_FORBIDDEN_FIELD", kpath(p, key), key)
            continue
        if not is_table(v):
            r.e("E_TYPE", kpath(p, key), v)
            continue
        out.append((key, v))
    return out


def phase3(doc: dict, r: Report) -> None:
    _fields(doc, TOP, "", r)
    _required(doc, ("conditions", "channels", "repeats", "used_ids"), "", r)
    for cid, c in _id_table(doc, "conditions", "", r):
        p = kpath("conditions", cid)
        _fields(c, COND, p, r)
        _required(c, ("source_folder",), p, r)
        _text(c, "source_folder", p, r)
        _text(c, "display_name", p, r, nonempty=True)
        _text(c, "reference", p, r)
        _text(c, "target", p, r, nonempty=True)
        _text(c, "treatment_key", p, r, nonempty=True)
    for chid, ch in _id_table(doc, "channels", "", r):
        p = kpath("channels", chid)
        _fields(ch, CHAN, p, r)
        _text(ch, "display_name", p, r, nonempty=True)
        _text_array(ch, "header_names", p, r, nonempty_items=True)
    for rid, rep in _id_table(doc, "repeats", "", r):
        p = kpath("repeats", rid)
        _fields(rep, REP, p, r)
        _required(rep, ("confirmed", "confirmed_on", "members"), p, r)
        _text(rep, "display_name", p, r, nonempty=True)
        if "confirmed" in rep:
            if type(rep["confirmed"]) is not bool:
                r.e("E_TYPE", p + ".confirmed", rep["confirmed"])
            elif rep["confirmed"] is not True:
                r.e("E_REPEAT_UNCONFIRMED", p + ".confirmed", False)
        if "confirmed_on" in rep:
            v = rep["confirmed_on"]
            if not is_text(v):
                r.e("E_TYPE", p + ".confirmed_on", v)
            else:
                ok = bool(DATE.fullmatch(v))
                if ok:
                    try:
                        date.fromisoformat(v)
                    except ValueError:
                        ok = False
                if not ok:
                    r.e("E_DATE_FORMAT", p + ".confirmed_on", v)
        for cid, m in _id_table(rep, "members", p, r):
            mp = kpath(p + ".members", cid)
            _fields(m, MEM, mp, r)
            _required(m, ("source_repeat", "image_count", "technical_repeats"), mp, r)
            _text(m, "source_repeat", mp, r)
            if "image_count" in m:
                v = m["image_count"]
                if type(v) is not int:
                    r.e("E_TYPE", mp + ".image_count", v)
                elif v < 1:
                    r.e("E_VALUE_RANGE", mp + ".image_count", v)
            for tid, u in _id_table(m, "technical_repeats", mp, r):
                up = kpath(mp + ".technical_repeats", tid)
                _fields(u, UNIT, up, r)
                _required(u, ("source_folder", "channels"), up, r)
                _text(u, "source_folder", up, r)
                _text(u, "display_name", up, r, nonempty=True)
                _text_array(u, "channels", up, r, min1=True)
                if "image_type" in u:
                    v = u["image_type"]
                    if not is_text(v):
                        r.e("E_TYPE", up + ".image_type", v)
                    elif v != "rgb":
                        r.e("E_ENUM", up + ".image_type", v)
    for mid, mk in _id_table(doc, "masks", "", r, required=False, min1=False):
        p = kpath("masks", mid)
        _fields(mk, MASK, p, r)
        _required(mk, ("source_type", "representation"), p, r)
        _text(mk, "source_type", p, r)
        _text(mk, "display_name", p, r, nonempty=True)
        _text_array(mk, "source_channels", p, r)
        if "representation" in mk and is_text(mk["representation"]):
            if mk["representation"] not in REPRESENTATIONS:
                r.e("E_ENUM", p + ".representation", mk["representation"])
        elif "representation" in mk:
            r.e("E_TYPE", p + ".representation", mk["representation"])
    if "used_ids" in doc:
        u = doc["used_ids"]
        if not is_table(u):
            r.e("E_TYPE", "used_ids", u)
        else:
            _fields(u, USED, "used_ids", r)
            _required(u, USED, "used_ids", r)
            for k in USED:
                _text_array(u, k, "used_ids", r, min1=True)


# ------------------------------------------------------------------ helpers over a valid structure
def members(doc: dict):
    for rid, rep in doc.get("repeats", {}).items():
        for cid, m in rep.get("members", {}).items():
            yield rid, cid, m


def units(doc: dict):
    for rid, cid, m in members(doc):
        for tid, u in m.get("technical_repeats", {}).items():
            yield rid, cid, tid, u


def upath(rid: str, cid: str, tid: str) -> str:
    return kpath(kpath(kpath("repeats", rid) + ".members", cid) + ".technical_repeats", tid)


# ------------------------------------------------------------------ phase 4: identity
def _label(v: str, path: str, r: Report) -> None:
    if v != v.strip() or v.casefold() in SENTINELS:
        r.e("E_ID_TEXT", path, v)


def _nfc(v: str, path: str, r: Report) -> None:
    if nfc(v) != v:
        r.e("E_ID_NOT_NFC", path, v)


def _token(v: str, path: str, r: Report) -> None:
    _label(v, path, r)
    if not TOKEN.fullmatch(v):
        r.e("E_ID_SYNTAX", path, v)


def _single(v: str, path: str, r: Report) -> None:
    if v.casefold() == "single" and v != "single":
        r.e("E_SINGLE_MISUSE", path, v)


def phase4(doc: dict, r: Report) -> None:
    scopes: dict[str, list[tuple[str, str]]] = {k: [] for k in (
        "conditions", "condition_folders", "repeats", "source_repeats", "technical_repeats",
        "technical_folders", "channels", "masks", "mask_types")}

    def label_id(v, path, scope):
        _label(v, path, r)
        _nfc(v, path, r)
        _single(v, path, r)
        scopes[scope].append((v, path))

    for cid, c in doc["conditions"].items():
        p = kpath("conditions", cid)
        label_id(cid, p, "conditions")
        label_id(c["source_folder"], p + ".source_folder", "condition_folders")
        if (cid == "single") != (c["source_folder"] == "single"):
            r.e("E_SINGLE_MISUSE", p + ".source_folder", c["source_folder"])
        elif c["source_folder"] == "single" and len(doc["conditions"]) > 1:
            r.e("E_SINGLE_MISUSE", p + ".source_folder", "single is not the only condition")
        if "reference" in c:
            _nfc(c["reference"], p + ".reference", r)  # a reference is NFC-checked only
    for chid in doc["channels"]:
        p = kpath("channels", chid)
        _token(chid, p, r)
        _nfc(chid, p, r)
        _single(chid, p, r)
        if chid.casefold() in RESERVED:
            r.e("E_ID_RESERVED", p, chid)
        scopes["channels"].append((chid, p))
    for rid, rep in doc["repeats"].items():
        p = kpath("repeats", rid)
        label_id(rid, p, "repeats")
    cond_members: dict[str, list[tuple[str, str]]] = {}
    for rid, cid, m in members(doc):
        mp = kpath(kpath("repeats", rid) + ".members", cid)
        _nfc(cid, mp, r)  # a member key is a reference to a condition ID
        label_id(m["source_repeat"], mp + ".source_repeat", "source_repeats")
        cond_members.setdefault(cid, []).append((m["source_repeat"], mp))
        tids = list(m["technical_repeats"])
        for tid, u in m["technical_repeats"].items():
            up = upath(rid, cid, tid)
            label_id(tid, up, "technical_repeats")
            label_id(u["source_folder"], up + ".source_folder", "technical_folders")
            if tid == "single" and len(tids) > 1:
                r.e("E_SINGLE_MISUSE", up, "single is not the only technical repeat")
            if (tid == "single") != (u["source_folder"] == "single"):
                r.e("E_SINGLE_MISUSE", up + ".source_folder", u["source_folder"])
            for i, ch in enumerate(u["channels"]):
                _nfc(ch, f"{up}.channels[{i}]", r)
    for cid, lst in cond_members.items():
        if any(s == "single" for s, _ in lst) and len(lst) > 1:
            for s, mp in lst:
                if s == "single":
                    r.e("E_SINGLE_MISUSE", mp + ".source_repeat", "single is not the only member")
    for mid, mk in doc.get("masks", {}).items():
        p = kpath("masks", mid)
        _token(mid, p, r)
        _nfc(mid, p, r)
        _single(mid, p, r)
        parts = mid.casefold().split("_")
        if mid.casefold() in RESERVED or "in" in parts:
            r.e("E_ID_RESERVED", p, mid)
        scopes["masks"].append((mid, p))
        label_id(mk["source_type"], p + ".source_type", "mask_types")
        for i, ch in enumerate(mk.get("source_channels", [])):
            _nfc(ch, f"{p}.source_channels[{i}]", r)
    for kind in USED:
        for i, v in enumerate(doc["used_ids"][kind]):
            p = f"used_ids.{kind}[{i}]"
            if kind == "channels":
                # Owner decision of 7 Nov 2026 (Spec issue 11): a listed channel ID reserves it,
                # so the reserved list of 3.2 applies as to a declared channel ID (4.9).
                _token(v, p, r)
                _nfc(v, p, r)
                if v.casefold() in RESERVED:
                    r.e("E_ID_RESERVED", p, v)
            else:
                _label(v, p, r)
                _nfc(v, p, r)
            _single(v, p, r)
            scopes[kind].append((v, p))
    for scope, items in scopes.items():
        by_fold: dict[str, set[str]] = {}
        for v, _ in items:
            by_fold.setdefault(nfc(v).casefold(), set()).add(nfc(v))
        for v, p in items:
            if len(by_fold[nfc(v).casefold()]) > 1:
                r.e("E_ID_CASE_COLLISION", p, v)


# ------------------------------------------------------------------ phase 5: semantics
def dn_key(x: str) -> str:
    return nfc(nfc(x).casefold())


def phase5(doc: dict, r: Report) -> None:
    conds = doc["conditions"]
    chans = doc["channels"]
    for rid, cid, m in members(doc):
        if cid not in conds:
            r.e("E_UNKNOWN_CONDITION", kpath(kpath("repeats", rid) + ".members", cid), cid)
    for rid, cid, tid, u in units(doc):
        up = upath(rid, cid, tid)
        seen = set()
        for i, ch in enumerate(u["channels"]):
            if ch not in chans:
                r.e("E_UNKNOWN_CHANNEL", f"{up}.channels[{i}]", ch)
            if ch in seen:
                r.e("E_DUPLICATE_CHANNEL_IN_UNIT", f"{up}.channels[{i}]", ch)
            seen.add(ch)
        if u.get("image_type") == "rgb":
            if u["channels"] != ["R", "G", "B"]:
                r.e("E_RGB_UNIT_CHANNELS", up + ".channels", u["channels"])
        else:
            for i, ch in enumerate(u["channels"]):
                if ch.casefold() in ("r", "g", "b"):
                    r.e("E_RGB_TOKEN_MISUSE", f"{up}.channels[{i}]", ch)
    for chid, ch in chans.items():
        hn = ch.get("header_names", [])
        if chid.casefold() in ("r", "g", "b") and hn:
            r.e("E_RGB_ALIAS", kpath("channels", chid) + ".header_names", hn)
        if len({nfc(x) for x in hn}) != len(hn):
            r.e("E_DUPLICATE_VALUE", kpath("channels", chid) + ".header_names", hn)
    for mid, mk in doc.get("masks", {}).items():
        sc = mk.get("source_channels", [])
        for i, ch in enumerate(sc):
            if ch not in chans:
                r.e("E_UNKNOWN_CHANNEL", f"{kpath('masks', mid)}.source_channels[{i}]", ch)
        if len(set(sc)) != len(sc):
            r.e("E_DUPLICATE_VALUE", kpath("masks", mid) + ".source_channels", sc)
    # display names
    groups: dict[str, list[tuple[str, str, str]]] = {k: [] for k in
                                                     ("c", "ch", "r", "t", "m")}
    for k, t in (("c", conds), ("ch", chans), ("r", doc["repeats"]),
                 ("m", doc.get("masks", {}))):
        for i, v in t.items():
            if "display_name" in v:
                groups[k].append((i, v["display_name"], kpath(k, i)))
    for rid, cid, tid, u in units(doc):
        if "display_name" in u:
            groups["t"].append((tid, u["display_name"], upath(rid, cid, tid)))
    for items in groups.values():
        owners: dict[str, set[str]] = {}
        for owner, name, _ in items:
            owners.setdefault(dn_key(name), set()).add(owner)
        for owner, name, p in items:
            if len(owners[dn_key(name)]) > 1:
                r.e("E_DISPLAY_NAME_COLLISION", p + ".display_name", name)
    # used_ids
    used = doc["used_ids"]
    declared = {
        "conditions": set(conds), "channels": set(chans), "repeats": set(doc["repeats"]),
        "technical_repeats": {tid for _, _, tid, _ in units(doc)},
    }
    for kind in USED:
        lst = used[kind]
        if len(set(lst)) != len(lst):
            r.e("E_DUPLICATE_VALUE", f"used_ids.{kind}", lst)
        for i in sorted(declared[kind] - set(lst)):
            r.e("E_USED_ID_MISSING", f"used_ids.{kind}", i)
    # one folder, one ID
    folder_ids: dict[str, set[str]] = {}
    for cid, c in conds.items():
        folder_ids.setdefault(c["source_folder"], set()).add(cid)
    for f, ids in folder_ids.items():
        if len(ids) > 1:
            r.e("E_SOURCE_FOLDER_REUSED", "conditions", (f, sorted(ids)))
    tfolder_ids: dict[str, set[str]] = {}
    tid_folders: dict[str, set[str]] = {}
    for rid, cid, tid, u in units(doc):
        tfolder_ids.setdefault(u["source_folder"], set()).add(tid)
        tid_folders.setdefault(tid, set()).add(u["source_folder"])
    for f, ids in tfolder_ids.items():
        if len(ids) > 1:
            r.e("E_SOURCE_FOLDER_REUSED", "technical_repeats", (f, sorted(ids)))
    for tid, fs in tid_folders.items():
        if len(fs) > 1:
            r.e("E_SOURCE_FOLDER_CONFLICT", tid, sorted(fs))
    # conditions with members
    member_conds = {cid for _, cid, _ in members(doc)}
    for cid in conds:
        if cid not in member_conds:
            r.e("E_CONDITION_WITHOUT_MEMBERS", kpath("conditions", cid), cid)
    # source repeats
    owner: dict[tuple[str, str], set[str]] = {}
    for rid, cid, m in members(doc):
        owner.setdefault((cid, m["source_repeat"]), set()).add(rid)
    for key, rids in owner.items():
        if len(rids) > 1:
            r.e("E_SOURCE_REPEAT_REUSED", "repeats", (key, sorted(rids)))
    for rid in doc["repeats"]:
        for rid2, cid, m in members(doc):
            if rid2 != rid and nfc(m["source_repeat"]).casefold() == nfc(rid).casefold():
                r.e("E_REPEAT_ID_SHADOWS_SOURCE",
                    kpath(kpath("repeats", rid2) + ".members", cid), (rid, m["source_repeat"]))
    types: dict[str, set[str]] = {}
    for mid, mk in doc.get("masks", {}).items():
        types.setdefault(mk["source_type"], set()).add(mid)
    for t, ids in types.items():
        if len(ids) > 1:
            r.e("E_MASK_SOURCE_TYPE_REUSED", "masks", (t, sorted(ids)))
    # reference graph
    edges = {}
    for cid, c in conds.items():
        if "reference" in c:
            ref = c["reference"]
            p = kpath("conditions", cid) + ".reference"
            if ref == cid:
                r.e("E_REFERENCE_SELF", p, ref)
            elif ref not in conds:
                r.e("E_REFERENCE_UNKNOWN", p, ref)
            else:
                edges[cid] = ref
    reported = set()
    for start in edges:
        seen = []
        cur = start
        while cur in edges and cur not in seen:
            seen.append(cur)
            cur = edges[cur]
        if cur in seen:
            cyc = frozenset(seen[seen.index(cur):])
            if cyc not in reported:
                reported.add(cyc)
                r.e("E_REFERENCE_CYCLE", "conditions", sorted(cyc))


def warnings(doc: dict, r: Report) -> None:
    used = {ch for _, _, _, u in units(doc) for ch in u["channels"]}
    for chid in doc["channels"]:
        if chid not in used:
            r.w("W_UNUSED_CHANNEL", kpath("channels", chid), chid)
    conds = set(doc["conditions"])
    rep_conds = {rid: set(rep["members"]) for rid, rep in doc["repeats"].items()}
    for rid, cs in rep_conds.items():
        if cs != conds:
            r.w("W_REPEAT_MISSING_CONDITIONS", kpath("repeats", rid), sorted(conds - cs))
    for cid, c in doc["conditions"].items():
        if "reference" in c:
            if not any(cid in cs and c["reference"] in cs for cs in rep_conds.values()):
                r.w("W_REFERENCE_UNPAIRED", kpath("conditions", cid), c["reference"])


def validate(data: bytes) -> tuple[Report, dict | None]:
    r = Report()
    doc = phase1(data, r)
    if r.errors:
        r.phase = 1
        return r, None
    for n, fn in ((2, phase2), (3, phase3), (4, phase4), (5, phase5)):
        fn(doc, r)
        if r.errors:
            r.phase = n
            return r, doc
    warnings(doc, r)
    return r, doc


# ------------------------------------------------------------------ canonical writer (section 9.1)
def _key(k: str) -> str:
    return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else _str(k)


def _str(s: str) -> str:
    out = ['"']
    esc = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f",
           "\r": "\\r"}
    for ch in s:
        if ch in esc:
            out.append(esc[ch])
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _val(v) -> str:
    if type(v) is bool:
        return "true" if v else "false"
    if type(v) is int:
        return str(v)
    if type(v) is str:
        return _str(v)
    if type(v) is list:
        return "[" + ", ".join(_val(x) for x in v) + "]"
    raise TypeError(type(v))


def _block(header: str, tbl: dict, order, sort_arrays=()) -> list[str]:
    lines = [f"[{header}]"]
    for k in order:
        if k in tbl and not is_table(tbl[k]):
            v = tbl[k]
            if k in sort_arrays:
                v = sorted(v)
            lines.append(f"{k} = {_val(v)}")
    return lines + [""]


def section_lines(doc: dict, sec: str) -> list[str]:
    """The table blocks of one section, each followed by a blank line (section 9.1)."""
    out: list[str] = []
    if sec == "conditions":
        for cid in sorted(doc["conditions"]):
            out += _block("conditions." + _key(cid), doc["conditions"][cid], COND)
    elif sec == "channels":
        for chid in sorted(doc["channels"]):
            out += _block("channels." + _key(chid), doc["channels"][chid], CHAN,
                          ("header_names",))
    elif sec == "repeats":
        for rid in sorted(doc["repeats"]):
            rep = doc["repeats"][rid]
            rh = "repeats." + _key(rid)
            out += _block(rh, rep, REP)
            for cid in sorted(rep["members"]):
                m = rep["members"][cid]
                mh = rh + ".members." + _key(cid)
                out += _block(mh, m, MEM)
                for tid in sorted(m["technical_repeats"]):
                    out += _block(mh + ".technical_repeats." + _key(tid),
                                  m["technical_repeats"][tid], UNIT)
    else:
        for mid in sorted(doc.get("masks", {})):
            out += _block("masks." + _key(mid), doc["masks"][mid], MASK, ("source_channels",))
    return out


def write(doc: dict) -> bytes:
    """Canonical bytes of a valid document (experiment-v1 section 9.1)."""
    out = [f"schema = {_str(doc['schema'])}", f"schema_version = {doc['schema_version']}", ""]
    for sec in ("conditions", "channels", "repeats", "masks"):
        out += section_lines(doc, sec)
    out.append("[used_ids]")
    for k in USED:
        out.append(f"{k} = {_val(sorted(doc['used_ids'][k]))}")
    return ("\n".join(out) + "\n").encode("utf-8")


def model(doc):
    """Parsed document with the arrays whose order is not semantic sorted (section 9.2)."""
    def walk(x, key=None):
        if is_table(x):
            return {k: walk(v, k) for k, v in x.items()}
        if type(x) is list:
            return sorted(x) if key in ("header_names", "source_channels") else list(x)
        return x
    d = walk(doc)
    d["used_ids"] = {k: sorted(v) for k, v in d.get("used_ids", {}).items()}
    return d


# ------------------------------------------------------------------ digests (section 9.3)
DESCRIPTIVE = {"display_name", "target", "treatment_key"}


def _strip(x):
    if is_table(x):
        return {k: _strip(v) for k, v in x.items() if k not in DESCRIPTIVE}
    if type(x) is list:
        return [_strip(v) for v in x]
    return x


def section_digests(doc: dict) -> dict[str, str]:
    """SHA-256 of one section's table blocks as the canonical writer emits them (section 9.3).

    The blocks of the stripped section (no `display_name`, `target`, `treatment_key`), each
    followed by one blank line, nothing before or after; an absent section is the empty bytes.
    """
    s = _strip(doc)
    return {sec: hashlib.sha256("".join(f"{line}\n" for line in section_lines(s, sec))
                                .encode("utf-8")).hexdigest()
            for sec in ("conditions", "channels", "repeats", "masks")}


# ------------------------------------------------------------------ derived CSV (section 11)
CSV_COLUMNS = ("condition", "biological_repeat", "repeat_group", "control", "image_count",
               "technical_repeat_count", "confirmed", "confirmed_on", "schema_version")


def derive_csv(doc: dict) -> bytes:
    """The derived experiment_design.csv of a valid document (experiment-v1 section 11).

    AMBIGUITY: the contract (section 12) fixes UTF-8, LF, the header row, the column order,
    `true`/`false` and the sort; it does not state the delimiter, the quoting, a final line
    terminator or a byte-order mark. Here: comma, minimal quoting, LF after every row including
    the last, no byte-order mark (as today's writer does).
    """
    referenced = {c["reference"] for c in doc["conditions"].values() if "reference" in c}
    rows = []
    for rid, cid, m in members(doc):
        tr = m["technical_repeats"]
        single = len(tr) == 1 and next(iter(tr.values()))["source_folder"] == "single"
        rows.append([doc["conditions"][cid]["source_folder"], m["source_repeat"], rid,
                     "true" if cid in referenced else "false", str(m["image_count"]),
                     "0" if single else str(len(tr)), "true",
                     doc["repeats"][rid]["confirmed_on"], "1"])
    rows.sort(key=lambda r: (r[0], r[1]))
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


# ------------------------------------------------------------------ transitions (R_*)
def transition(old: dict, new: dict) -> set[str]:
    codes: set[str] = set()
    for k in USED:
        ou, nu = set(old["used_ids"][k]), set(new["used_ids"][k])
        if ou - nu:
            codes.add("R_USED_ID_DROPPED")
        old_declared = (set(old[k]) if k != "technical_repeats"
                        else {t for _, _, t, _ in units(old)})
        new_declared = (set(new[k]) if k != "technical_repeats"
                        else {t for _, _, t, _ in units(new)})
        if (ou - old_declared) & new_declared:
            codes.add("R_USED_ID_REUSED")

    def rep_folders(doc, rid):
        return frozenset((c, m["source_repeat"]) for c, m in doc["repeats"][rid]["members"].items())

    def cond_folders(doc, cid):
        return {m["source_repeat"] for _, c, m in members(doc) if c == cid}

    def tech(doc):
        out: dict[str, tuple[set, set]] = {}
        for rid, cid, tid, u in units(doc):
            src = doc["repeats"][rid]["members"][cid]["source_repeat"]
            f, ms = out.setdefault(tid, (set(), set()))
            f.add(u["source_folder"])
            ms.add((cid, src))
        return out

    def positions(doc):
        pos: dict[str, set] = {}
        for rid, cid, tid, u in units(doc):
            src = doc["repeats"][rid]["members"][cid]["source_repeat"]
            for i, ch in enumerate(u["channels"]):
                pos.setdefault(ch, set()).add((cid, src, tid, i))
        return pos

    # (a) same member folders, other repeat ID
    old_by_f = {rep_folders(old, r): r for r in old["repeats"]}
    for rid in new["repeats"]:
        f = rep_folders(new, rid)
        if f in old_by_f and old_by_f[f] != rid:
            codes.add("R_CONFIRMED_ID_CHANGED")
    # (b) channel ID replaced at every position with the same header evidence
    op, np_ = positions(old), positions(new)
    for ch, ps in op.items():
        if ch in new["channels"]:
            continue
        for nch, nps in np_.items():
            if nch not in old["channels"] and nps == ps and \
                    sorted(old["channels"][ch].get("header_names", [])) == \
                    sorted(new["channels"][nch].get("header_names", [])):
                codes.add("R_CONFIRMED_ID_CHANGED")
    # (b2) same source_folder, other condition / technical repeat ID
    old_cf = {c["source_folder"]: cid for cid, c in old["conditions"].items()}
    for cid, c in new["conditions"].items():
        if c["source_folder"] in old_cf and old_cf[c["source_folder"]] != cid:
            codes.add("R_CONFIRMED_ID_CHANGED")
    ot, nt = tech(old), tech(new)
    old_tf = {f: tid for tid, (fs, _) in ot.items() for f in fs}
    for tid, (fs, _) in nt.items():
        for f in fs:
            if f in old_tf and old_tf[f] != tid:
                codes.add("R_CONFIRMED_ID_CHANGED")
    # (c) an ID declared in both names a changed entity (something removed or reassigned)
    for rid in set(old["repeats"]) & set(new["repeats"]):
        if not rep_folders(old, rid) <= rep_folders(new, rid):
            codes.add("R_CONFIRMED_ID_CHANGED")
    for cid in set(old["conditions"]) & set(new["conditions"]):
        if old["conditions"][cid]["source_folder"] != new["conditions"][cid]["source_folder"] \
                or not cond_folders(old, cid) <= cond_folders(new, cid):
            codes.add("R_CONFIRMED_ID_CHANGED")
    for tid in set(ot) & set(nt):
        if ot[tid][0] != nt[tid][0] or not ot[tid][1] <= nt[tid][1]:
            codes.add("R_CONFIRMED_ID_CHANGED")
    for ch in set(old["channels"]) & set(new["channels"]):
        if not op.get(ch, set()) <= np_.get(ch, set()) or \
                not set(old["channels"][ch].get("header_names", [])) <= \
                set(new["channels"][ch].get("header_names", [])):
            codes.add("R_CONFIRMED_ID_CHANGED")
    # confirmed_on of a repeat whose confirmation is not reopened (section 10: members, units and
    # channel lists unchanged; a changed image_count alone does not reopen it)
    def layout(rep):
        return {c: {k: v for k, v in _strip(m).items() if k != "image_count"}
                for c, m in rep["members"].items()}

    for rid in set(old["repeats"]) & set(new["repeats"]):
        o, n = old["repeats"][rid], new["repeats"][rid]
        if layout(o) == layout(n) and o["confirmed_on"] != n["confirmed_on"]:
            codes.add("R_CONFIRMED_ON_CHANGED")
    return codes


# ------------------------------------------------------------------ fixture catalogue
CODE = re.compile(r"\b([EWR]_[A-Z_]+)\b")


def run_fixtures(root: Path) -> int:
    fx = root / "fixtures"
    readme = (fx / "README.md").read_text(encoding="utf-8")
    rows = {}
    for line in readme.splitlines():
        m = re.match(r"\| \[`([vit]\d+[^`]*)`\]\([^)]+\) \|(.*)", line)
        if m:
            rows[m.group(1)] = [c.strip() for c in m.group(2).split("|")]
    bad = 0
    docs = {}
    counts = {}
    for sub in ("valid", "invalid", "transitions"):
        for f in sorted((fx / sub).glob("*.toml"), key=lambda p: int(re.match(r"\D(\d+)", p.stem)[1])):
            data = f.read_bytes()
            r, doc = validate(data)
            docs[f.stem.split("_")[0]] = doc
            row = rows.get(f.stem)
            if row is None:
                print(f"NO-CATALOGUE-ROW {sub}/{f.stem}")
                bad += 1
                continue
            if sub == "invalid":
                exp = set(CODE.findall(row[1]))
                ok = r.codes == exp and not r.warnings and str(r.phase) == row[2]
                got = f"phase={r.phase} {sorted(r.codes)}"
                want = f"phase={row[2]} {sorted(exp)}"
            else:
                cell = row[1] if sub == "valid" else row[2]
                expw = {c for c in CODE.findall(cell) if c.startswith("W_")}
                ok = not r.errors and r.warning_codes == expw
                got = f"errors={sorted(r.codes)} warnings={sorted(r.warning_codes)}"
                want = f"warnings={sorted(expw)}"
            counts[sub] = counts.get(sub, 0) + 1
            if not ok:
                bad += 1
                print(f"MISMATCH {sub}/{f.stem}: got {got}; expected {want}")
                for e in r.errors[:5]:
                    print("    ", e)
    for f in sorted((fx / "transitions").glob("*.toml")):
        row = rows[f.stem]
        pred = re.search(r"`valid/(v\d+)`|valid/(v\d+)", row[0] + " " + row[1] + " " + row[2])
        pred = pred.group(1) or pred.group(2)
        exp = {c for c in CODE.findall(row[1]) if c.startswith("R_")}
        got = transition(docs[pred], docs[f.stem.split("_")[0]])
        if got != exp:
            bad += 1
            print(f"MISMATCH transition {f.stem} vs {pred}: got {sorted(got)} expected {sorted(exp)}")
        counts["transition-checks"] = counts.get("transition-checks", 0) + 1
    # derived CSV golden files (section 11): each is listed in the README and equals the derivation
    for f in sorted((fx / "expected").glob("*.csv")):
        key = f.stem.split("_")[0]
        if f"`{f.name}`" not in readme:
            bad += 1
            print(f"NO-CATALOGUE-ROW expected/{f.name}")
        if key not in docs or derive_csv(docs[key]) != f.read_bytes():
            bad += 1
            print(f"MISMATCH expected/{f.name}: not the derivation of {key}")
        counts["expected"] = counts.get("expected", 0) + 1
    # round trip and digests
    for k, d in docs.items():
        if k.startswith("v") or k.startswith("t"):
            once = write(d)
            if model(tomllib.loads(once.decode())) != model(d) or \
                    write(tomllib.loads(once.decode())) != once:
                bad += 1
                print(f"MISMATCH round trip {k}")
    if docs["v18"] != docs["v01"]:
        bad += 1
        print("MISMATCH v18 model != v01")
    for k in ("v16", "v21"):
        if section_digests(docs[k]) != section_digests(docs["v01"]):
            bad += 1
            print(f"MISMATCH {k} digests != v01")
    n_records, bad_records = refrecords.run_fixtures(root)  # the records goldens (section 4.6)
    bad += bad_records
    counts["records"] = n_records
    print(f"checked {counts}; round trip on {sum(1 for k in docs if k[0] in 'vt')} files; "
          f"mismatches: {bad}")
    return bad


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "check":
        r, _ = validate(Path(argv[1]).read_bytes())
        if "--json" in argv:
            print(json.dumps(r.as_dict(), ensure_ascii=False))
        else:
            for e in r.errors:
                print("ERROR", *e)
            for w in r.warnings:
                print("WARN ", *w)
            print("phase", r.phase, "OK" if not r.errors else "REFUSED")
        return 1 if r.errors else 0
    if len(argv) == 2 and argv[0] == "fixtures":
        return 1 if run_fixtures(Path(argv[1])) else 0
    if len(argv) == 3 and argv[0] == "transition":
        old = tomllib.loads(Path(argv[1]).read_text(encoding="utf-8"))
        new = tomllib.loads(Path(argv[2]).read_text(encoding="utf-8"))
        print(sorted(transition(old, new)))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
