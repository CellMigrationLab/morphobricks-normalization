# AGENTS.md — MorphoBricks normalization brick

Guidance for AI coding agents (and humans) working on
[CellMigrationLab/morphobricks-normalization](https://github.com/CellMigrationLab/morphobricks-normalization).
Read it, and the documents below, before changing anything.

| File | What it fixes |
|---|---|
| `docs/plan.md` | Goals, methods, scopes, milestones, the owner's decisions N1–N69 (§12) and what is still to be measured (§13) |
| `docs/contract.md` | Rules other bricks rely on: methods and formulas, interface, scopes and repeat groups, image types, the statistics file, intensity checks, parameter columns, validation |
| `CHANGELOG.md` | Release notes and migration notes |
| `docs/archive/` | Historical record: the two done follow-ups for the calling bricks |

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
                    compute_block_histogram for planes read in blocks), memory estimate;
                    robust range with HistogramTail underflow/overflow and
                    histogram_with_range_check, quantiles by radix selection (Milestone 5)
  bin_centres.py    float_bin_centres: bin centres that do not overflow near the float64 limits
  bin_reachability.py  float_bin_indices, reachable_float_bins, centres_map_back: which float
                    bins a value can fall in and whether centres map back (#86)
  integer_representatives.py  integer bin representatives (nearest float64 midpoint) and
                    their exact bound for 64-bit values float64 cannot hold (#28)
  ranges.py         HistogramSettings, RangeCheck, range_check_issues (Milestone 5)
  integrity.py      invariants every PlaneHistogram and HistogramTail checks when made
                    (counts, limits, tail moments) and the pixel type of every block
  tail_moments.py   exact centred moments of a tail (count, mean, std; schema version 5)
  channels.py       canonical channel IDs (the Spec's Token rule): check_channel_id(s)
  records.py        PlaneRecord (one plane_histograms.csv row: an XYC plane, one
                    histogram per channel; position_id and channels (IDs) required),
                    plane_id, group_records; ChannelMeasurementFailure/Error (HV3)
  csv_io.py         write/read plane_histograms.csv schema version 5 (ADR 0003): channels
                    in CellProfiler-style columns named by channel ID, presence column
                    `channels`, underflow/overflow columns (atomic write, labels always text)
  pooling.py        PooledHistogram: pooled counts, never mean-of-means, tails by
                    their means for percentiles and by their moments for mean and std;
                    WeightedHistogram: histograms added with a weight each (Milestone 2);
                    linear_percentile: NumPy's linear interpolation (Milestone 3)
  normalization.py  NormalizationParameters (offset, scale) and the four linear methods
  settings.py       NormalizationSettings, check_settings, MethodNotAllowedError
  scopes.py         group_key, group_name, scope_groups: groups at each scope, repeat groups
  rgb.py            StarDist's percentile normalization: rgb_percentile, apply_rgb
                    (CSBDeep's float32 arithmetic), check_rgb
  stain.py          image-to-scalar presets: to_scalar, rgb_to_scalar, PRESETS (catalogue with
                    modality and consumer hints), white_from_region, stain_vectors_from_regions
                    (optional scikit-image extra `histology`; contract §3.4, N67, N68)
  optical_density.py  optical density against a declared white, stain matrices (normalised,
                    cross-product completion, collinearity and conditioning checks), unmixing
                    (NumPy only; the `custom_od` presets)
  scalar_inputs.py  input checks of the presets: RGB with declared axes, single YX planes
  fitting.py        fit, by channel ID
  parameters.py     Parameters (apply, select, to_columns, from_columns with
                    Normalization_Offset_<ID> and Normalization_Scale_<ID>)
  summaries.py      ImageSummary, image_summaries, summary_columns: one row per image,
                    Z and T pooled, from plane_histograms.csv rows (Milestone 4)
  checks.py         IntensityCheckSettings, intensity_checks: the four intensity warnings;
                    unchecked_channels: channels without finite pixels
  parameter_checks.py  finite offsets, positive scales and whole channel indices that
                    NormalizationParameters and Parameters check when made; the scope
                    from_columns checks against the settings
  issues.py         Issue: warnings returned to the calling brick (channel for the checks)
tests/              pytest with simulated planes (tests/simulation.py); tests/data holds
                    the golden pixels and fitted parameters of tests/test_fit_goldens.py
tools/check.py      the release gate: ruff format, pytest and the code standards (design v0.4 §9);
                    `--rules` runs only the standards (CI); tools/code_rules.py is the shared
                    copy (SHA-256 pinned in tests/test_code_rules.py), tools/code_rules_baseline.json
                    the shrink-only baseline taken at v0.3.3 (never add or raise an entry)
tools/dump_outputs.py  bit-exact dump of the outputs for valid inputs and its SHA-256, to
                    show a change leaves them unchanged (run on base and change, compare)
```

The six modules `dtypes` to `normalization` were moved from morphobricks-preprocessing
commit `fd923c9` in Milestone 1 with only the package name changed; keep their behaviour
and messages unchanged (contract §1), apart from the owner-approved move of the channels
into columns (contract §1.3). Milestone 2 added `settings`, `scopes`, `fitting`, `issues`
and `WeightedHistogram`; Milestone 2b put the channels across columns (`plane_histograms.csv`
schema version 3, per-channel parameter columns); Milestone 3 added `rgb` and
`linear_percentile` for RGB images; Milestone 4 added `summaries` and `checks`, the
intensity checks, and `Issue.channel`; Milestone 5 added `ranges`, the robust
histogram range with underflow and overflow (`plane_histograms.csv` schema version 4) and the
range check.

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
8. **Dependencies.** Required: `numpy` only; the optional extra `histology` (scikit-image) is
   imported lazily by `stain.py` alone (N67). No GPU, deep-learning, widget or file-format
   dependency beyond what the statistics file needs.

## 3. How we work

1. **Read first:** the plan, the contract and the modules you touch.
2. **Ask before deciding.** Decisions N1–N69 are final (those marked withdrawn in plan §12 no longer apply); do not re-open them. A choice the
   plan and the contract leave open, and any change of product behaviour or file format,
   is the owner's: ask with 2–4 options, the recommended one first.
3. **Outputs stay bit-identical** unless a change says otherwise. `tests/test_fit_goldens.py` keeps the fitted
   parameters of simulated planes equal to the numbers stored in `tests/data` (contract §1.3).
4. **Tests** use simulated data generated in the tests; real data only as an optional
   check. Each formula is tested against a direct implementation (plan §11).
5. **Style** as in morphobricks-preprocessing: Python 3.11+ (MorphoBricks decision D01), `ruff` (line length 100,
   NumPy docstrings), small modules with one job each, comments that explain why.
6. **Done means** `python tools/check.py` passes (Ruff's findings go through the shrink-only
   baseline, so plain `ruff check .` still lists the baselined debt) and the docs are updated. Report honestly what was verified and what was not.
7. **One milestone per branch and PR** against `main`. Do the task asked; list unrelated
   problems as follow-ups.
