# AGENTS.md — MorphoBricks normalization brick

Guidance for AI coding agents (and humans) working on
[CellMigrationLab/morphobricks-normalization](https://github.com/CellMigrationLab/morphobricks-normalization).
Read it, and the documents below, before changing anything.

| File | What it fixes |
|---|---|
| `docs/plan.md` | Goals, methods, scopes, milestones, the owner's decisions N1–N53 (§12) and what is still to be measured (§13) |
| `docs/contract.md` | Rules other bricks rely on: methods and formulas, interface, scopes and repeat groups, image types, the statistics file, intensity checks, parameter columns, validation |
| `docs/followup_preprocessing.md`, `docs/followup_cellpatcher.md` | What the two calling bricks change once this brick is released |

## 1. What the brick is

One importable library (`morphobricks_normalization`) with the normalization methods for
fluorescence and RGB images, the statistics they need, the intensity checks, and the reader
and writer of the statistics file (N2, N33). It is called by:

- **morphobricks-preprocessing**, which walks the experiment folder, asks the image type,
  the biological-repeat pairing and confirmation, writes the statistics file with this
  library's functions and shows its intensity checks;
- **CellPatcher**, which normalizes patches with this library.

Repository map (one module per job):

```
src/morphobricks_normalization/
  dtypes.py         supported pixel types, default and allowed bin counts
  histogram.py      per-plane histogram (PlaneHistogram, compute_plane_histogram,
                    compute_block_histogram for planes read in blocks), memory estimate
  records.py        PlaneRecord (one plane_histograms.csv row: an XYC plane, one
                    histogram per channel), plane_id, group_records
  csv_io.py         write/read plane_histograms.csv schema version 3, channels in
                    CellProfiler-style columns (atomic write, labels always text)
  pooling.py        PooledHistogram: pooled counts, never mean-of-means;
                    WeightedHistogram: histograms added with a weight each (Milestone 2);
                    linear_percentile: NumPy's linear interpolation (Milestone 3)
  normalization.py  NormalizationParameters (offset, scale) and the four linear methods
  settings.py       NormalizationSettings, check_settings, MethodNotAllowedError
  scopes.py         group_key, group_name, scope_groups: groups at each scope, repeat groups
  rgb.py            StarDist's percentile normalization: rgb_percentile, apply_rgb
                    (CSBDeep's float32 arithmetic), check_rgb
  fitting.py        fit, Parameters (apply, select, to_columns, from_columns with
                    Normalization_Offset_C<i> and Normalization_Scale_C<i>)
  summaries.py      ImageSummary, image_summaries, summary_columns: one row per image,
                    Z and T pooled, from plane_histograms.csv rows (Milestone 4)
  checks.py         IntensityCheckSettings, intensity_checks: the four intensity warnings
  issues.py         Issue: warnings returned to the calling brick (channel for the checks)
tests/              pytest with simulated planes (tests/simulation.py); tests/data holds
                    a schema-version-2 fixture written by the Milestone 2 code
```

The six modules `dtypes` to `normalization` were moved from morphobricks-preprocessing
commit `fd923c9` in Milestone 1 with only the package name changed; keep their behaviour
and messages unchanged (contract §1), apart from the owner-approved move of the channels
into columns (contract §1.3). Milestone 2 added `settings`, `scopes`, `fitting`, `issues`
and `WeightedHistogram`; Milestone 2b put the channels across columns (`plane_histograms.csv`
schema version 3, per-channel parameter columns); Milestone 3 added `rgb` and
`linear_percentile` for RGB images; Milestone 4 added `summaries` and `checks`, the
intensity checks, and `Issue.channel`.

## 2. Rules of the repository

1. **Pure library.** No widgets, notebooks, command line or folder walking. Functions
   receive pixel arrays, statistics and settings, and return statistics, parameters and
   arrays. Intensity checks return tables and warnings; the calling bricks display them.
2. **Never modify source data.** `apply` returns a new array. Never apply a transformation
   the user did not choose.
3. **Every algorithm once.** Reuse one implementation across image types, scopes and
   statistics sources (files or pixels), so that the same inputs always give the same
   numbers. StarDist's RGB normalization is implemented here in NumPy; CSBDeep is used only
   as an optional check in tests (N33).
4. **Only meaningful choices.** A method the image type does not allow, or a scope the
   statistics do not support, is refused with a message naming an allowed alternative
   (contract §5, §9).
5. **Plain-language errors.** Every error and warning says what happened, which images,
   groups or patches are affected, and what to do, in the style of the other MorphoBricks
   bricks. The library never prints: errors are exceptions, warnings are returned to the
   caller. Technical details belong in the caller's log.
6. **Contract first.** File formats, column names, method names and defaults are fixed in
   `docs/contract.md`. Changing a file format needs a new `schema_version`; changing any
   other rule needs the owner's approval, recorded in plan §12. Provisional defaults (plan
   §13) are marked as such and change only after measurement.
7. **Parameters as numbers.** Parameters are numeric columns, never JSON; only the
   method's columns are written; settings are stored once per run (N21, N23).
   `from_columns(to_columns(p))` must equal `p` exactly.
8. **Dependencies.** Required: `numpy` only. No GPU, deep-learning, widget or file-format
   dependency beyond what the statistics file needs.

## 3. How we work

1. **Read first:** the plan, the contract and the modules you touch.
2. **Ask before deciding.** Decisions N1–N53 are final (those marked withdrawn in plan §12 no longer apply); do not re-open them. A choice the
   plan and the contract leave open, and any change of product behaviour or file format,
   is the owner's: ask with 2–4 options, the recommended one first.
3. **Milestone 1 moves code unchanged.** The histogram, bin-count, record, CSV, pooling
   and normalization modules come from morphobricks-preprocessing with only the package
   name changed, with their tests, and give bit-identical outputs (contract §1). New
   behaviour starts in Milestone 2; Milestone 2b changed only the table layout (contract
   §1.3), and `tests/test_schema_v2_equivalence.py` keeps the numbers bit-identical.
4. **Tests** use simulated data generated in the tests; real data only as an optional
   check. Each formula is tested against a direct implementation (plan §11).
5. **Style** as in morphobricks-preprocessing: Python 3.11+ (MorphoBricks decision D01), `ruff` (line length 100,
   NumPy docstrings), small modules with one job each, comments that explain why.
6. **Done means** `ruff check .`, `ruff format --check .` and `pytest` pass and the docs
   are updated. Report honestly what was verified and what was not.
7. **One milestone per branch and PR** against `main`. Do the task asked; list unrelated
   problems as follow-ups.
