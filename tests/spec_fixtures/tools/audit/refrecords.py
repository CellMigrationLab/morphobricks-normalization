"""AUDIT TOOL ONLY: reference validator and canonical encoder of `morphobricks.records.jsonl`.

This is not a runtime package and no brick may import it. It implements, and only, the records
rules of docs/experiment-v1.md section 4.6: the three typed record forms (`reconfirmation`,
`image_count_changed`, `removed`, with RR3) and the byte rules of the file. It sits beside
`refval.py` (which stays the validator of `morphobricks.toml`) and is run by `refval.py fixtures`.

A refused file is `R_RECORDS_FILE_INVALID` (section 8, owner decision RF3); the `E_RECORDS_*`
names below are its reasons, named by this tool (owner decisions RF1-RF5 of 9 November 2026).

Usage:
    python -I refrecords.py check FILE [--json]
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")  # matched with fullmatch
SHA256 = re.compile(r"[0-9a-f]{64}")  # lower-case hex (RF4); matched with fullmatch, not `$`
ID_KINDS = ("conditions", "channels", "repeats", "technical_repeats")
ENTITIES = ("unit", "technical_repeat", "repeat", "condition")
RECONFIRMATION = ("type", "id_kind", "old_id", "new_id", "confirmed_on", "old_sha256", "new_sha256")
IMAGE_COUNT = (
    "type",
    "condition_id",
    "condition_source_folder",
    "repeat_id",
    "source_repeat",
    "technical_repeats",
    "old_image_count",
    "new_image_count",
    "recorded_on",
)
REMOVED_NAMES = (
    "condition_id",
    "condition_source_folder",
    "repeat_id",
    "source_repeat",
    "technical_repeat_id",
    "technical_repeat_source_folder",
)
REMOVED = ("type", "entity", *REMOVED_NAMES, "confirmed_on")
KEY_SETS = {
    "reconfirmation": frozenset(RECONFIRMATION),
    "image_count_changed": frozenset(IMAGE_COUNT),
    "removed": frozenset(REMOVED),
}
# the keys a `removed` line names, per entity; every other name key is null (section 4.6)
NAMED = {
    "unit": REMOVED_NAMES,
    "technical_repeat": ("technical_repeat_id", "technical_repeat_source_folder"),
    "repeat": ("repeat_id",),
    "condition": ("condition_id", "condition_source_folder"),
}
UNIT_KEYS = ("id", "source_folder")
ESCAPE = re.compile(r'\\(?:u([0-9a-fA-F]{4})|.)', re.DOTALL)


class Report:
    """Errors of one records file: (reason, line number, detail)."""

    def __init__(self) -> None:
        self.errors: list[tuple[str, int, str]] = []
        self.records: list[dict] = []

    def e(self, code: str, line: int, detail: str = "") -> None:
        self.errors.append((code, line, detail))

    @property
    def codes(self) -> set[str]:
        return {c for c, _, _ in self.errors}

    def as_dict(self) -> dict:
        return {
            "records": len(self.records),
            "errors": [list(map(str, x)) for x in self.errors],
            "error_codes": sorted(self.codes),
        }


# canonical encoder --------------------------------------------------------------------------


def encode_record(record: dict) -> bytes:
    """Return the canonical bytes of one record (section 4.6, RF1): keys in code point order,
    non-ASCII unescaped, the separators `, ` and `: `, the minimal string escapes, one LF. This is
    Python's `json.dumps(ensure_ascii=False, sort_keys=True)`, which wrote every line so far.
    """
    return (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def encode_records(records: list[dict]) -> bytes:
    """Return the bytes of a records file holding exactly these records, in order."""
    return b"".join(encode_record(r) for r in records)


# parsing ------------------------------------------------------------------------------------


class _Duplicate(ValueError):
    pass


def _pairs(items: list[tuple[str, object]]) -> dict:
    out: dict = {}
    for key, value in items:
        if key in out:
            raise _Duplicate(key)
        out[key] = value
    return out


def _constant(name: str) -> object:
    raise ValueError(f"{name} is not JSON")


def _sorted_keys(value: object) -> bool:
    """True when every object nested in ``value`` lists its keys in code point order."""
    if isinstance(value, dict):
        keys = list(value)
        return keys == sorted(keys) and all(_sorted_keys(v) for v in value.values())
    if isinstance(value, list):
        return all(_sorted_keys(v) for v in value)
    return True


def _escaped_non_ascii(line: str) -> bool:
    """True when the line writes a non-ASCII character as a `\\uXXXX` escape."""
    return any(m.group(1) and int(m.group(1), 16) >= 0x80 for m in ESCAPE.finditer(line))


# values -------------------------------------------------------------------------------------


def _text(v: object) -> bool:
    return type(v) is str and not any(0xD800 <= ord(c) <= 0xDFFF for c in v)


def _date(v: object) -> bool:
    if type(v) is not str or not DATE.fullmatch(v):
        return False
    try:
        date.fromisoformat(v)
    except ValueError:
        return False
    return True


def _count(v: object) -> bool:
    return type(v) is int and v >= 1


def _bad_reconfirmation(rec: dict) -> list[str]:
    bad = []
    if rec["id_kind"] not in ID_KINDS:
        bad.append("id_kind")
    if not _text(rec["old_id"]):
        bad.append("old_id")
    if rec["new_id"] is not None and not _text(rec["new_id"]):
        bad.append("new_id")
    if not _date(rec["confirmed_on"]):
        bad.append("confirmed_on")
    old = rec["old_sha256"]
    if not (old == "absent" or (type(old) is str and SHA256.fullmatch(old))):
        bad.append("old_sha256")
    if not (type(rec["new_sha256"]) is str and SHA256.fullmatch(rec["new_sha256"])):
        bad.append("new_sha256")
    return bad


def _bad_removed(rec: dict) -> list[str]:
    if rec["entity"] not in ENTITIES:
        return ["entity"]
    bad = []
    named = NAMED[rec["entity"]]
    for key in REMOVED_NAMES:
        if key in named and not _text(rec[key]) or key not in named and rec[key] is not None:
            bad.append(key)
    if not _date(rec["confirmed_on"]):
        bad.append("confirmed_on")
    return bad


def _bad_units(units: object) -> bool:
    if type(units) is not list:
        return True
    ids = []
    for unit in units:
        if type(unit) is not dict or set(unit) != set(UNIT_KEYS):
            return True
        if not (_text(unit["id"]) and _text(unit["source_folder"])):
            return True
        ids.append(unit["id"])
    return ids != sorted(ids)  # sorted by id in code point order (ties cannot occur in a member)


def _bad_image_count(rec: dict) -> list[str]:
    names = ("condition_id", "condition_source_folder", "repeat_id", "source_repeat")
    bad = [k for k in names if not _text(rec[k])]
    if _bad_units(rec["technical_repeats"]) or not rec["technical_repeats"]:
        bad.append("technical_repeats")  # an array of {id, source_folder}, sorted, not empty
    bad += [k for k in ("old_image_count", "new_image_count") if not _count(rec[k])]
    if not bad and rec["old_image_count"] == rec["new_image_count"]:
        bad.append("new_image_count")  # the record states a change (RF5)
    if not _date(rec["recorded_on"]):
        bad.append("recorded_on")
    return bad


CHECKS = {
    "reconfirmation": _bad_reconfirmation,
    "removed": _bad_removed,
    "image_count_changed": _bad_image_count,
}


def _check_line(raw: bytes, number: int, r: Report) -> None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        r.e("E_RECORDS_ENCODING", number, "not UTF-8")
        return
    try:
        rec = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    except _Duplicate as err:
        r.e("E_RECORDS_NOT_JSON_OBJECT", number, f"key {err} twice")
        return
    except ValueError as err:
        r.e("E_RECORDS_NOT_JSON_OBJECT", number, str(err))
        return
    if type(rec) is not dict:
        r.e("E_RECORDS_NOT_JSON_OBJECT", number, type(rec).__name__)
        return
    kind = rec.get("type")
    if type(kind) is not str or kind not in KEY_SETS:
        r.e("E_RECORDS_UNKNOWN_TYPE", number, repr(kind))
        return
    if set(rec) != KEY_SETS[kind]:
        odd = sorted(set(rec) ^ KEY_SETS[kind])
        r.e("E_RECORDS_KEY_SET", number, f"{kind}: differs in {odd}")
        return
    before = len(r.errors)
    bad = CHECKS[kind](rec)
    if bad:
        r.e("E_RECORDS_VALUE", number, f"{kind}: {bad}")
    if not _sorted_keys(rec):
        r.e("E_RECORDS_KEY_ORDER", number)
    if _escaped_non_ascii(text):
        r.e("E_RECORDS_NON_ASCII_ESCAPED", number)
    if len(r.errors) > before:
        return
    r.records.append(rec)
    if raw + b"\n" != encode_record(rec):
        r.e("E_RECORDS_FORM", number, "not the canonical bytes (whitespace, CR, optional escape)")
        return


def validate_records(data: bytes) -> Report:
    """Validate the bytes of a `morphobricks.records.jsonl`; every line is checked."""
    r = Report()
    if not data:
        return r  # an empty file holds no record and is accepted (RF4)
    if data.startswith(b"\xef\xbb\xbf"):
        r.e("E_RECORDS_ENCODING", 1, "byte-order mark")  # RF4
    parts = data.split(b"\n")  # LF only: U+2028, U+0085 and CR do not end a line
    if parts[-1] != b"":
        r.e("E_RECORDS_NO_FINAL_LF", len(parts), "last line lacks its LF")
    else:
        parts = parts[:-1]
    for number, raw in enumerate(parts, start=1):
        if number == 1 and data.startswith(b"\xef\xbb\xbf"):
            continue
        _check_line(raw, number, r)
    return r


CODE = re.compile(r"\b(E_RECORDS_[A-Z_]+)\b")
ROW = re.compile(r"\| \[`(r[vi]\d+[^`]*)`\]\([^)]+\) \|([^|]*)\|")


def run_fixtures(root: Path) -> tuple[int, int]:
    """Check every file of fixtures/records against its row of fixtures/README.md.

    Returns (files checked, mismatches). A file with no row, or a row with no file, is a mismatch.
    """
    fx = root / "fixtures"
    readme = (fx / "README.md").read_text(encoding="utf-8")
    rows = {m.group(1): m.group(2).strip() for m in map(ROW.match, readme.splitlines()) if m}
    files = {f.stem: f for f in sorted((fx / "records").glob("*.jsonl"))}
    bad = 0
    for name in sorted(set(rows) ^ set(files)):
        bad += 1
        print(f"NO-CATALOGUE-ROW-OR-FILE records/{name}")
    for name in sorted(set(rows) & set(files)):
        data = files[name].read_bytes()
        r = validate_records(data)
        cell = rows[name]
        codes = set(CODE.findall(cell))
        if name.startswith("rv"):
            # an accepted file is the canonical encoding of the records it holds
            ok = not r.errors and encode_records(r.records) == data
            want = "accepted"
        else:
            line = re.search(r"\(line (\d+)\)", cell)
            ok = r.codes == codes and bool(line) and r.errors[0][1] == int(line.group(1))
            want = f"refused {sorted(codes)} first at line {line and line.group(1)}"
        if not ok:
            bad += 1
            print(f"MISMATCH records/{name}: got errors={r.errors[:3]}; expected {want}")
    return len(files), bad


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "check":
        r = validate_records(Path(argv[1]).read_bytes())
        if "--json" in argv:
            print(json.dumps(r.as_dict(), ensure_ascii=False))
        else:
            for e in r.errors:
                print("ERROR", *e)
            print(len(r.records), "records", "OK" if not r.errors else "REFUSED")
        return 1 if r.errors else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
