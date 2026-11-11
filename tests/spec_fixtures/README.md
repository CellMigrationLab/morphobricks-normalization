# Vendored MorphoBricks-Spec files

**Do not edit anything in this folder.** It is a byte-exact, pinned copy of
[MorphoBricks-Spec](https://github.com/CellMigrationLab/MorphoBricks-Spec) made by the Spec's shared sync tool. Regenerate it with
`vendor.py`, never by hand:

    python <Spec checkout>/tools/vendor.py --into <this repository's root> --commit <full sha>

`PIN.toml` names the Spec commit and the git hash of each pinned path. The pin test hashes
this folder against it offline, and a changed, added or missing byte fails it. The copy is a
test fixture, not a runtime dependency. A fixture that a test fails is a finding for the Spec
repository, not something to change here. How to re-pin: `docs/vendoring.md` in the Spec.

| Here | In the Spec |
|---|---|
| `fixtures/` | `fixtures/` |
| `docs/experiment-v1.md` | `docs/experiment-v1.md` |
| `tools/audit/refval.py` | `tools/audit/refval.py` |
| `tools/audit/refrecords.py` | `tools/audit/refrecords.py` |
