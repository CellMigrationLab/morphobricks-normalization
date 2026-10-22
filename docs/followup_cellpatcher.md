# Follow-up Plan — CellPatcher

> See also: [Normalization brick plan](plan.md) · [Preprocessing follow-up](followup_preprocessing.md)

**Status:** plan; starts after the preprocessing follow-up (N6). Decision numbers N1–N24 refer
to §12 of the [normalization plan](plan.md); D-numbers refer to CellPatcher's own
`docs/plan.md`, and this plan revises some of them (see the last section).
**Repository:** [CellMigrationLab/CellPatcher](https://github.com/CellMigrationLab/CellPatcher)

## Goal

CellPatcher keeps patching as it is and applies every normalization method of the
normalization brick, limited by the modality and purpose, at the experiment, condition,
repeat, image or patch scope. Statistics, parameters and normalization come from
`morphobricks_normalization`; image readers and manifest checks keep coming from
`morphobricks_preprocessing`.

## Today

CellPatcher's `preprocessing` extra pins `morphobricks-preprocessing[zarr]` to a commit and
uses it for two things:

- **Normalization:** `compute_plane_histogram`, `PooledHistogram`, `NormalizationParameters`,
  `min_max`, `percentile_range`, `z_score`, `robust_z_score`, `NoValidPixelsError`,
  `read_plane_histograms`, `make_plane_id` (in `normalization.py` and `statistics.py`).
- **Reading and checks:** `AxisMapping`, `inspect_image`, `read_plane`, `open_ngff`,
  `open_ngff_labels`, `read_manifest`, manifest fingerprints and validation messages (in
  `planes.py`, `io.py`, `statistics.py`).

## C1 — Switch to the normalization brick (no behaviour change)

- The normalization imports come from `morphobricks_normalization`; the reading and manifest
  imports stay on `morphobricks_preprocessing`.
- Extras:
  - `cellpatcher[normalization]` installs `morphobricks-normalization`: enough for the image
    and patch scopes on TIFF images without preprocessing outputs (N9);
  - `cellpatcher[preprocessing]` installs `morphobricks-preprocessing[zarr]`, which brings the
    normalization brick with it: group scopes, manifest checks and OME-Zarr input.
- Both pinned to released versions and raised deliberately.
- Acceptance: identical patches, `patches.csv` and `run_config.json` before and after.

## C2 — Scopes and references

- Add the `experiment` scope (N10) and reference groups as a single parameter source for
  linear methods (§4.2 of the normalization plan).
- Without preprocessing outputs, the `image` scope is computed from the image's own pixels
  (all its Z and T planes, per channel) instead of being refused; group scopes stay
  unavailable with the existing message (N9). This revises CellPatcher's contract §5 and §7.
- References come from the preprocessing manifest; without one, a reference image path can be
  set in the configuration (image and patch scopes only).

## C3 — Modality and purpose

- `NormalizationConfig` gains `modality` and `purpose` (`model_input` by default).
- With a manifest, the modality comes from it; a different modality in the configuration is an
  error. Without one, the modality is required in the configuration whenever normalization is
  enabled (N15).
- The normalization brick's profiles validate method, scope and purpose before any patch is
  read; refusals name an allowed alternative (N16).

## C4 — New method families

- **Shared linear (B), optical density (C), luminance option:** from `plane_histograms.csv`
  at group scopes, or from pixels at image and patch scopes.
- **Histogram matching (D):** source scope histograms and the reference's histograms; the
  mapping is rebuilt, never stored (N22).
- **Colour methods (E):** from `colour_statistics.csv` and `colour_samples.npz` at group and
  image scopes, or from the patch's pixels at patch scope. Patches stay RGB (N20); results are
  clipped (N19) and returned as float32 in 0–1 or uint8 RGB. At patch scope, Macenko and
  Vahadane need about 1,000 tissue pixels; patches below are discarded and reported like the
  D8 patches (N18).
- **Checks** against the statistics files extend to colour statistics: every patched RGB plane
  has a row, and its pixel and tissue counts match.

## C5 — Outputs

- `patches.csv` keeps the common normalization columns (`normalization_method`, `_scope`,
  `_group`, `_exact`) and writes only the parameter columns of the method used (N23), as listed
  in §8 of the normalization plan. The existing linear columns (`normalization_offset`,
  `normalization_scale`) keep their names and meaning.
- `run_config.json` records the target (reference definition, target stain vectors and
  amounts, Reinhard target, Ruifrok stain matrix), the modality and the purpose, once per
  run (N21).
- Raw-file delivery normalizes on load for every family, from the row's columns plus the
  run-level target; histogram matching rebuilds its mapping from the recorded group and
  reference.

## C6 — Notebook

- The normalization step offers only the methods, scopes and purpose choices of the modality,
  shows the references from the manifest, and previews before/after patches.

## Tests and acceptance

- C1: identical outputs with the moved code.
- Every family gives the same arrays in all delivery modes (normalized files, raw files
  normalized on load, eager and lazy in memory).
- Columns written per method match the contract exactly and round-trip through
  `from_columns`.
- Image scope without preprocessing equals image scope computed from the statistics files
  for exact (8- and 16-bit) histograms.
- Modality and purpose refusals happen before any image is read.

## Revisions to CellPatcher's earlier decisions

| CellPatcher decision | Revision |
|---|---|
| D1: import from `morphobricks_preprocessing` | Normalization from `morphobricks_normalization`; reading from preprocessing (C1) |
| D4: channels are never pooled | Unless the modality profile prescribes shared parameters (family B), or the method is a colour method |
| D5: per-group parameters only; scopes condition to patch | Adds `experiment` and reference groups (C2) |
| D10: no clipping | Clipping for colour methods only (N19) |
| D14 and contract §10: fixed parameter columns, always present | Only the columns of the method used; targets in `run_config.json` (C5) |
| Contract §5: image scope needs the CSV | Image scope is computed from pixels when there is no CSV (C2) |

These revisions are owner-approved through decisions N1–N24 and must be recorded in
CellPatcher's `docs/plan.md`, `docs/normalization_contract.md` and `AGENTS.md` §5 when C1–C5
are implemented.
