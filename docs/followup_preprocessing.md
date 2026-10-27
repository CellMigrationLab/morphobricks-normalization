# Follow-up Plan — morphobricks-preprocessing

> See also: [Normalization brick plan](plan.md) · [CellPatcher follow-up](followup_cellpatcher.md)

**Status:** P1–P3 done; P3b, P4 and P5 planned (revised 27 October 2026). Decision numbers
refer to §12 of the [normalization plan](plan.md).
**Repository:** [CellMigrationLab/morphobricks-preprocessing](https://github.com/CellMigrationLab/morphobricks-preprocessing)

## Goal

The preprocessing brick keeps its job (validate the input, walk the folders, read metadata,
confirm axes, optional OME-Zarr conversion, write `plane_histograms.csv`) and gains:

- dataset warnings in the folder check (P2);
- the experiment design: pairing and confirming biological repeats, and marking controls
  (P3);
- `plane_histograms.csv` with the channels in columns, schema version 3 (P3b);
- the image type, `fluorescence` or `rgb` (P4);
- intensity checks and plots after the histograms (P5).

Statistics, normalization and intensity-check code comes from `morphobricks_normalization`
(N1, N2). P2 and P3 did not depend on the normalization brick; P3b needs its Milestone 2b,
P4 its Milestone 3 and P5 its Milestone 4. Until the brick's `v0.2.0` release, this brick
pins it by commit.

## P1 — Depend on the normalization brick (no behaviour change)

*Done* in [#16](https://github.com/CellMigrationLab/morphobricks-preprocessing/pull/16) and
[#17](https://github.com/CellMigrationLab/morphobricks-preprocessing/pull/17): the moved
modules are imported from `morphobricks_normalization`, with deprecated re-exports for one
version, and every output is bit-identical.

## P2 — Dataset warnings in the folder check (step 1)

Step 1 reads only folder names and image headers. Its results gain a **Dataset consistency**
panel that gathers every consistency warning in one place (N37). Warnings never block.

- **Balance.** A table with conditions as columns and biological repeats as rows, holding the
  number of images in each cell (technical repeats added up), and a second count of technical
  repeats per biological repeat. A cell with fewer than half or more than twice the median of
  the non-empty cells is highlighted and reported (`unbalanced_images`).
- **Repeats missing from some conditions.** A biological repeat that is not found in every
  condition (after the pairing of P3) is reported (`repeat_missing_conditions`); it is still
  used.
- **Pixel size.** The physical pixel size (X, Y and the Z step) read from the metadata
  differs between images (`different_pixel_sizes`), or is missing for some images and present
  for others (`missing_pixel_size`). Read with the existing `read_physical_sizes`.
- **Already checked today**, now listed in the same panel: mixed pixel types
  (`mixed_dtypes`), inconsistent axes (`inconsistent_mapping`), different image sizes
  (`different_sizes`, `inconsistent_dimensions`), different channel names
  (`channel_names_differ`).
- **New shape checks.** Different numbers of channels, Z planes or time points between images
  (`different_plane_counts`).

Each warning names the affected images or folders and what to check. The command line prints
the same warnings.

## P3 — Experiment design (step 1)

After the labels are confirmed, step 1 asks three things and writes the answers to
`experiment_design.csv` in the output folder. Later steps stay locked until the biological
repeats are confirmed (N35).

1. **Pair the biological repeats across conditions** (N30).
   - By default, repeats with the same folder name in different conditions are one repeat
     group, named after the folder.
   - A table lists every repeat group with its folder in each condition. When a repeat has a
     different name in another condition (acquired a day later, for example), the user moves
     it into the right group with a dropdown. Each folder belongs to exactly one group.
   - Groups not present in every condition are shown in amber and reported (P2), not blocked.
2. **Confirm the biological repeats** (N35).
   - A yellow box asks for each repeat group: "Were these images prepared and acquired under
     the same circumstances, so they are comparable with each other?" with a **Yes, they
     are comparable** button.
   - When there are no biological-repeat folders (every condition has the label `single`),
     the question becomes: "All images of every condition come from one biological repeat:
     were they prepared and acquired under the same circumstances?" (N31).
   - Missing technical-repeat folders need no question: the images of the biological repeat
     are used directly (N31).
3. **Mark the controls** (N36). A checkbox per condition; none, one or several may be marked.
   Recorded only; nothing uses it yet.

**`experiment_design.csv`** (schema version 1), one row per condition and biological-repeat
folder, sorted by condition then biological repeat:

| Column | Meaning |
|---|---|
| `condition`, `biological_repeat` | Folder labels, as in `plane_histograms.csv` (`single` when absent) |
| `repeat_group` | Label of the repeat group the folder belongs to (default: `biological_repeat`) |
| `control` | `true` when the condition is marked as a control |
| `image_count`, `technical_repeat_count` | Counts at the time of confirmation |
| `confirmed` | `true` once the repeat group is confirmed |
| `confirmed_on` | Date of the confirmation (ISO 8601) |
| `schema_version` | `1` |

- **Reruns.** When the file exists and its rows match the folders found, the pairing,
  confirmations and controls are loaded and the questions are skipped, with a green line
  saying when they were confirmed and a **Change** button. A new or renamed condition or
  biological repeat asks again only for the groups it affects. Added or removed images
  within a confirmed folder keep the confirmation.
- **Command line.** `run` reads `experiment_design.csv` from the output folder, or from
  `--design <file>`. Without a file, or with unconfirmed groups, it stops and writes a
  template with `confirmed=false` for the user to review and set to `true`.
- **Manifest.** Records the SHA-256 of the `experiment_design.csv` used, so CellPatcher can
  check it matches (a new optional key; the manifest stays version 1).
- The file format is defined in this brick's `docs/contract.md`; CellPatcher reads it with a
  reader exported by this brick.

## P3b — Channels across columns (step 3)

Follows the normalization brick's Milestone 2b (N41–N44, its contract §1.3 and §6).

- Step 3 writes `plane_histograms.csv` schema version 3 with the brick's writer: one row per
  XYC plane (image, Z, T), each channel's extrema, counts and bins in its own
  CellProfiler-style columns (`Histogram_Count_C0_00000`, …).
- Planes are read with all their channels together, so the memory estimate and the
  block-reading decision cover every channel of a plane; block reading keeps giving the same
  histograms as reading whole.
- **Channel count.** Images with different numbers of channels are an error before any
  image is read (`different_channel_counts`, N43), with the images and their counts and
  "every image of an experiment must have the same channels; split or exclude the
  others". `different_plane_counts` keeps warning about Z, T and other planes only.
- **Reruns.** A table of schema version 1 or 2 is recomputed completely on the next run, as
  for any other version change. The manifest keeps version 1 and records schema version 3.
- **Step 4** (statistics) reads the new layout; its table and plot do not change.
- The histograms in the new table equal those of version 2, channel by channel (test against
  a version-2 table written before the change).

## P4 — Image type (step 2)

- After the axis roles, step 2 asks for the image type: **Fluorescence** or **RGB** (N34).
  It suggests RGB when the metadata has an RGB samples axis (TIFF photometric RGB, three
  samples per pixel) and Fluorescence otherwise; the user confirms with a **Yes** button.
- RGB needs exactly three channels and an integer pixel type; otherwise the RGB option is
  disabled with the reason.
- The manifest records `image_type` and whether the channel axis came from RGB samples (new
  optional keys; manifest version 1). Changing the type on a rerun does not recompute
  histograms: they are the same for both types.
- The statistics stay `plane_histograms.csv` only, one row per channel, for both types.

## P5 — Intensity checks and plots (after step 3)

Run automatically after the histograms are computed, and in step 4 for any table, using the
normalization brick's `image_summaries` and `intensity_checks` (N38):

- **Warnings** (limits adjustable under *Check settings*, default 3 robust deviations):
  images far from the other images of their condition and biological repeat, biological
  repeats shifted from the other repeats of their condition, saturated images, and images
  without intensity spread. Comparisons stay within a condition, so treatment effects are
  not reported.
- **Plots, in the notebook only** (no files written), per channel:
  - box plots of the image medians and 99.8th percentiles, grouped by condition and
    biological repeat, with flagged images highlighted;
  - overlaid intensity distributions per biological repeat within each condition.
- The command line prints the warnings and draws no plots.

## Tests and acceptance

- P1: done.
- P2: each new warning fires on a constructed folder (unbalanced counts, a repeat missing from
  one condition, differing or missing pixel sizes, differing plane counts) and stays silent on
  a balanced one.
- P3: same-named repeats pair by default; manual pairing is saved and reloaded; later steps and
  the command line are blocked until every group is confirmed; a rerun with unchanged folders
  skips the questions; a new repeat asks only for its group; the all-`single` case asks the
  single question.
- P3b: one row per XYC plane with every channel's columns; histograms equal to version 2;
  different channel counts stop the run before reading; a version-2 table is recomputed.
- P4: RGB is suggested only for RGB samples, never applied without confirmation, and refused
  for images that are not three-channel integer images.
- P5: the warnings match the normalization brick's checks on the same table; plots are drawn
  from the table without reading images; nothing is written.

## Contract changes

- `docs/contract.md`: `experiment_design.csv` (schema, reruns, command line), the new
  validation codes of P2, the manifest keys of P3 and P4.
- `docs/user_guide.md` and `README.md`: the new step 1 and step 2 questions, the checks and
  plots; links to the normalization contract point to §6.2 for `plane_histograms.csv`
  (previously §8.2).
- `AGENTS.md`: the experiment design and image type.
