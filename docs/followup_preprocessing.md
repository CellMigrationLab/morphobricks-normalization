# Follow-up Plan — morphobricks-preprocessing

> See also: [Normalization brick plan](plan.md) · [CellPatcher follow-up](followup_cellpatcher.md)

**Status:** plan; starts once the normalization brick has published a pinned release with the
moved code (normalization plan, Milestone 1). Decision numbers refer to §12 of the
[normalization plan](plan.md).
**Repository:** [CellMigrationLab/morphobricks-preprocessing](https://github.com/CellMigrationLab/morphobricks-preprocessing)

## Goal

The preprocessing brick keeps its job (validate the input, walk the folders, read metadata,
confirm axes, optional OME-Zarr conversion, write statistics files) and gains three things:
the modality question, reference selection, and colour statistics. All statistics and
normalization code comes from `morphobricks_normalization` (N1, N2).

## P1 — Depend on the normalization brick (no behaviour change)

- Add `morphobricks-normalization` as a required dependency, pinned to its first release.
- Remove the moved modules and import them instead: `histogram.py`, the bin-count rules of
  `dtypes.py`, `records.py`, `csv_io.py`, `pooling.py`, `normalization.py`. Their public names
  stay importable from `morphobricks_preprocessing` for one version, with a
  `DeprecationWarning` naming the new location (N6).
- `plane_histograms.csv` stays schema version 2, and every output (CSV, manifest, OME-Zarr
  copies) is bit-identical before and after.
- Acceptance: the existing test suite passes unchanged, apart from tests that move with the
  code to the normalization brick.
- *Done (October 2026), preprocessing PR #16. Agreed with the owner in review: the run
  manifest also records the `morphobricks-normalization` version, the notebook's Step 0
  installs the new dependency into older environments (both with small test updates), and
  the text of preprocessing contract §3–§7 moved to this contract (§1.2, §8.2).*

## P2 — Modality

- A new notebook step and CLI option ask for the imaging modality (`fluorescence`,
  `brightfield_gray`, `brightfield_rgb`, `phase_contrast`, `histology_rgb`,
  `electron_microscopy`). Metadata may suggest one (an RGB samples axis suggests
  `brightfield_rgb` or `histology_rgb`), but the user always confirms (N15).
- The manifest records the modality, and that the channel axis came from RGB samples. These
  are new optional manifest keys, so the manifest stays version 1, as with earlier additions.
- The modality decides which statistics files are written (N7):
  - `plane_histograms.csv` for every modality except `histology_rgb`;
  - `colour_statistics.csv` and `colour_samples.npz` for `histology_rgb`.
- Changing the modality on a rerun recomputes the affected statistics, like a change of bin
  count today.

## P3 — References

- After the modality, the user chooses the references the modality's methods may need
  (N13, N14):
  - a reference image (one image of the experiment);
  - a reference group, chosen by labels (for example every `control` image);
  - for `histology_rgb`, the published H&E or H-DAB standard (default).
- The manifest records each reference by label path or `source_relative_path`, never by
  absolute path.
- A reference that no longer matches any analysed image after a rerun is an error with a
  plain-language message.

## P4 — Colour statistics

For `histology_rgb` experiments, computed with the normalization brick's functions:

- **Reading.** The three colour channels of a plane are read together (memory estimate
  three times one channel). Planes over the memory limit are read in blocks, as for
  histograms today; colour summaries add up across blocks and pixel samples use reservoir
  sampling, so block reading gives identical results.
- **Settings** in the manifest: I₀ mode and values, tissue-mask threshold and exclusions,
  sample size and seed (N15a, N16a, N24). Changing a setting recomputes the colour
  statistics; changing only I₀ or the tissue mask can be served from the stored raw pixel
  samples for sample-based fits, but the exact summaries are recomputed.
- **Incremental reruns, publication and backups** follow the same rules as
  `plane_histograms.csv`, with the colour files staged and published together with the table
  and manifest.

## P5 — Normalized preview

- A notebook step shows before/after previews of a few planes using the normalization brick's
  preview helpers, with the method, scope and purpose choices limited to those the modality
  profile allows (N16).
- The preview never writes normalized images; normalization of patches stays in CellPatcher.

## Tests and acceptance

- P1: bit-identical outputs; deprecated re-exports warn once per name.
- P2: the modality is required before statistics run; the suggested modality never applies
  without confirmation; the right statistics files are written for each modality.
- P3: references survive reruns and are reported clearly when they stop matching.
- P4: blockwise and whole-plane colour statistics are equal; reruns are idempotent and only
  new or changed images are read.
- P5: only allowed methods are offered for each modality and purpose.

## Contract changes

- `docs/contract.md`: modality (§2), references, colour statistics schema and settings, and
  the files written per modality.
- `AGENTS.md`: the dependency on the normalization brick and the moved modules.
