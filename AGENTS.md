# AGENTS.md — MorphoBricks normalization brick

Guidance for AI coding agents (and humans) working on
[CellMigrationLab/morphobricks-normalization](https://github.com/CellMigrationLab/morphobricks-normalization).
Read it, and the documents below, before changing anything.

| File | What it fixes |
|---|---|
| `docs/plan.md` | Goals, method families, milestones, the owner's decisions N1–N28 (§12) and what is still to be measured (§13) |
| `docs/contract.md` | Rules other bricks rely on: methods and formulas, interface, scopes, references, modality profiles, purpose rules, refusal messages, statistics files, parameter columns, defaults, validation |
| `docs/followup_preprocessing.md`, `docs/followup_cellpatcher.md` | What the two calling bricks change once this brick is released |

## 1. What the brick is

One importable library (`morphobricks_normalization`) with every normalization method,
the statistics they need, the modality profiles, and the readers and writers of the
statistics files (N2). It is called by:

- **morphobricks-preprocessing**, which walks the experiment folder, asks the modality and
  references, and writes the statistics files with this library's functions;
- **CellPatcher**, which normalizes patches with this library.

Repository map (one module per job):

```
src/morphobricks_normalization/
  dtypes.py         supported pixel types, default and allowed bin counts
  histogram.py      per-plane histogram (PlaneHistogram, compute_plane_histogram,
                    compute_block_histogram for planes read in blocks), memory estimate
  records.py        PlaneRecord (one plane_histograms.csv row), plane_id, group_records
  csv_io.py         write/read plane_histograms.csv (atomic write, labels always text)
  pooling.py        PooledHistogram: pooled counts, never mean-of-means
  normalization.py  NormalizationParameters (offset, scale) and the four linear methods
tests/              pytest with simulated planes (tests/simulation.py)
```

These six modules were moved from morphobricks-preprocessing commit `fd923c9` in
Milestone 1 with only the package name changed; keep their behaviour and messages
unchanged (contract §1).

## 2. Rules of the repository

1. **Pure library.** No widgets, notebooks, command line or folder walking. Functions
   receive pixel arrays, statistics and settings, and return statistics, parameters and
   arrays. Previews are functions that return thumbnails and tables (Milestone 5); the
   calling bricks display them.
2. **Never modify source data.** `apply` returns a new array. Never apply a transformation
   the user did not choose.
3. **Every algorithm once.** A modality is configuration (a profile), not code. Reuse one
   implementation across modalities, scopes and statistics sources (files or pixels), so
   that the same inputs always give the same numbers. Do not copy methods from TIAToolbox
   or StainTools; implement them here in NumPy (N5).
4. **Only meaningful choices.** A method, scope or purpose outside the modality profile is
   refused, with a message naming an allowed alternative (contract §5–§7).
5. **Plain-language errors.** Every error and warning says what happened, which images,
   groups or patches are affected, and what to do, in the style of the other MorphoBricks
   bricks. The library never prints: errors are exceptions, warnings are returned to the
   caller. Technical details belong in the caller's log.
6. **Contract first.** File formats, column names, method names and defaults are fixed in
   `docs/contract.md`. Changing a file format needs a new `schema_version`; changing any
   other rule needs the owner's approval, recorded in plan §12. Provisional defaults (plan
   §13) are marked as such and change only after measurement.
7. **Parameters as numbers.** Parameters are numeric columns, never JSON; only the
   method's columns are written; targets are stored once per run (N21, N23).
   `from_columns(to_columns(p))` must equal `p` exactly.
8. **Dependencies.** Required: `numpy`, `scikit-image`, `scikit-learn` (N5). No GPU,
   deep-learning, widget or file-format dependency beyond what the statistics files need.

## 3. How we work

1. **Read first:** the plan, the contract and the modules you touch.
2. **Ask before deciding.** Decisions N1–N28 are final; do not re-open them. A choice the
   plan and the contract leave open, and any change of product behaviour or file format,
   is the owner's: ask with 2–4 options, the recommended one first.
3. **Milestone 1 moves code unchanged.** The histogram, bin-count, record, CSV, pooling
   and normalization modules come from morphobricks-preprocessing with only the package
   name changed, with their tests, and give bit-identical outputs (contract §1). New
   behaviour starts in Milestone 2.
4. **Tests** use simulated data generated in the tests; real data only as an optional
   check. Each formula is tested against a direct implementation (plan §11).
5. **Style** as in morphobricks-preprocessing: Python 3.10+, `ruff` (line length 100,
   NumPy docstrings), small modules with one job each, comments that explain why.
6. **Done means** `ruff check .`, `ruff format --check .` and `pytest` pass and the docs
   are updated. Report honestly what was verified and what was not.
7. **One milestone per branch and PR** against `main`. Do the task asked; list unrelated
   problems as follow-ups.
