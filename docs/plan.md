# MorphoBricks Normalization Brick — Implementation Plan

> See also: [Preprocessing follow-up](followup_preprocessing.md) · [CellPatcher follow-up](followup_cellpatcher.md)

**Status:** design brief, simplified on 26 October 2026 (decisions N29–N39, §12). The earlier
design with six modalities, two purposes, colour and stain methods and reference targets
(N4–N28) is withdrawn where §12 says so. One choice is still open (§13).
**Scope:** one importable library with the normalization methods for fluorescence and RGB
microscopy images, the statistics they need, the intensity checks run on those statistics,
and the file formats that carry statistics between bricks.
**Out of scope:** image correction (flat-field, dark-field, background, illumination),
deferred (N3); user interfaces, which stay in the bricks that call the library.

## Instructions for implementation

Keep the library free of widgets, notebooks and file-system walking: it receives pixel arrays,
statistics and settings, and returns statistics, parameters, warnings and normalized arrays.
Never modify source images. Never apply a transformation the user did not choose. Implement
every algorithm once and reuse it for every scope and statistics source. Every error explains
in plain language what happened and what to do, in the style of the other MorphoBricks
bricks.

## 1. Role in MorphoBricks

```
morphobricks-preprocessing ──▶ plane_histograms.csv ──▶ CellPatcher ──▶ normalized patches
            │                 + experiment_design.csv        │
            └── calls ──▶ morphobricks-normalization ◀── calls
```

- **morphobricks-preprocessing** validates the input data and its balance, walks the folder
  structure, reads image metadata, asks the user to confirm the axis order, the image type,
  the biological repeats and the controls, optionally converts images to OME-Zarr, writes
  `plane_histograms.csv` and `experiment_design.csv`, and shows the intensity checks and
  plots of this library in its notebook.
- **morphobricks-normalization** (this brick) computes the statistics, owns the format of
  `plane_histograms.csv`, implements the methods and scopes, and computes the intensity
  checks.
- **CellPatcher** reads both files and normalizes patches with this library.

## 2. Concepts

| Term | Meaning |
|---|---|
| **Image type** | `fluorescence` or `rgb`, suggested from the metadata and confirmed by the user (§5). It decides the method. |
| **Method** | One normalization algorithm (§3). |
| **Scope** | Which pixels share one set of parameters: biological repeat, image or patch (§4). |
| **Repeat group** | The images of one biological repeat across every condition, technical repeats merged (§4.1). |
| **Statistics** | Per-plane histograms computed once over the dataset, so parameters for any scope can be derived without rereading images (§7). |
| **Parameters** | An offset and a scale per channel (§8). |

Every method follows one interface:

- `fit(statistics, settings) → Parameters` for one group;
- `Parameters.apply(pixels) → normalized pixels`;
- `Parameters.to_columns() / from_columns()` so parameters are stored as named numeric columns
  and read back exactly (N21).

## 3. Methods

| Image type | Methods | Transform |
|---|---|---|
| `fluorescence` | min-max, percentile range (default 1–99.8), z-score, robust z-score: the four existing methods | `(x − offset) / scale`, each channel separately |
| `rgb` | StarDist percentile normalization only (N33) | `(x − P_low) / (P_high − P_low + 10⁻²⁰)`, with one `P_low` and one `P_high` for R, G and B together |

### 3.1 StarDist percentile normalization

StarDist normalizes its inputs with `csbdeep.utils.normalize`:

```python
mi = np.percentile(x, pmin, axis=axis, keepdims=True)
ma = np.percentile(x, pmax, axis=axis, keepdims=True)
x = (x - mi) / (ma - mi + 1e-20)        # float32, no clipping by default
```

Its examples use `pmin=1`, `pmax=99.8`. For RGB images they normalize "channels jointly" with
`axis=(0, 1, 2)`, so the percentiles are taken over all pixels of all three channels and every
channel gets the same offset and scale; colour ratios are kept.

In this library, `P_low` and `P_high` come from one pooled histogram of the R, G and B planes
of the group (§4), so the method works at every scope from `plane_histograms.csv` without
rereading images. Whether the percentiles use NumPy's default linear interpolation, which
reproduces StarDist exactly, or the library's existing inverse-CDF percentiles is the open
choice of §13.

## 4. Scopes

| Scope | Pixels pooled | Removes differences between |
|---|---|---|
| `biological_repeat` | Every image of the repeat group: one biological repeat in every condition, technical repeats merged | Biological repeats; keeps the differences between conditions within a repeat |
| `image` (default) | All Z and T planes of one image | Images |
| `patch` | The patch's own pixels | Everything outside the patch |

There is no experiment, condition or technical-repeat scope (N29), no batch scope and no
per-stain scope (N12). Parameters are per channel for `fluorescence` and shared by R, G and B
for `rgb`. Time points are pooled by default, with an option to give each time point its own
group.

### 4.1 Repeat groups

- A repeat group is the same biological repeat across conditions. By default, repeats with the
  same folder name form one group (`2024_05_01` in `control` and in `treated`). When a repeat
  was acquired on different days for different conditions, the user pairs the repeats in the
  preprocessing brick (N30).
- A repeat found in only some conditions forms a group from those conditions alone, with a
  warning; the run is not blocked (N30).
- A condition without biological-repeat folders is one repeat labelled `single`. If every
  condition is like this, the user confirms that all images are one biological repeat, and
  the whole experiment is one repeat group (N31).
- Technical repeats are always merged within their biological repeat; a biological repeat
  without technical-repeat folders contributes its images directly (N31).
- The pairing and the user's confirmation are recorded in `experiment_design.csv`, written by
  the preprocessing brick (follow-up P3). The library receives the pairing as a mapping from
  `(condition, biological_repeat)` to the repeat group's label.

### 4.2 Pooling weights

At the `biological_repeat` scope a condition with more or larger images would weigh more.
Both options are kept so they can be compared later (N32):

| `pooling` | Each condition contributes |
|---|---|
| `pixels` (default) | In proportion to its valid pixels: histograms are added, as today |
| `conditions` | Equally: each condition's pooled counts are scaled to the same total before adding |

Minimum and maximum do not depend on the weights. At the `image` and `patch` scopes there is
one condition, so both options give the same result.

## 5. Image types

- `fluorescence` (default): any number of channels; each channel is an independent stain.
- `rgb`: exactly three channels, R, G and B; an integer pixel type (8- or 16-bit RGB).
- The preprocessing brick suggests `rgb` when the image metadata has an RGB samples axis and
  `fluorescence` otherwise; the user confirms (N34). The choice is recorded in its manifest.
  Without preprocessing outputs, the calling brick's configuration states it.

## 6. Intensity checks

From `plane_histograms.csv` alone, without reading images, the library computes per image and
channel (all Z and T planes pooled): median, 1st and 99.8th percentiles, fraction of
saturated pixels, and whether the channel has any spread. It returns warnings (N38):

| Check | Compared with | Default limit |
|---|---|---|
| Image intensity outlier | The other images of the same condition and biological repeat | Median or 99.8th percentile more than 3 robust deviations (MAD × 1.4826) away |
| Biological repeat shifted | The other biological repeats of the same condition | Median of the repeat's image medians more than 3 robust deviations of the other repeats' images away |
| Saturated image | — | More than 1 % of the pixels at the type's maximum (integer images) |
| No intensity spread | — | 1st and 99.8th percentiles equal: the image cannot be normalized at the image scope |

Comparisons stay within one condition, so a treatment effect is never reported as a shift.
Limits are settings; a group with fewer than 5 images is not tested for outliers (provisional).
The preprocessing notebook shows the warnings and draws the plots (follow-up P5).

## 7. Statistics

`plane_histograms.csv`, schema version 2, unchanged: one row per 2D plane and channel, with
exact extrema, pixel counts and binned counts. It serves both image types (an RGB plane has
three rows, `c` = 0, 1, 2), every scope and the intensity checks. No other statistics file is
written (N33).

## 8. Parameters as columns

Parameters are stored as numeric columns, never as JSON (N21); only the columns of the method
used are written (N23). The settings are written once in the run configuration of the
calling brick.

| Column | Meaning |
|---|---|
| `normalization_method` | Written name, e.g. `percentile 1-99.8`, `robust z-score`, `rgb percentile 1-99.8` |
| `normalization_scope` | `biological_repeat`, `image` or `patch` |
| `normalization_group` | Repeat group label, `source_relative_path`, or `<source_relative_path>::<patch id>` |
| `normalization_exact` | Whether every pooled histogram was exact |
| `normalization_offset`, `normalization_scale` | One value per channel, `;`-separated; for `rgb` the shared value three times |

## 9. Dependencies

`numpy` only. No GPU or deep-learning dependency; StarDist and CSBDeep are not dependencies,
their formula is implemented here.

## 10. Milestones

- **Milestone 0 — Contract:** `docs/contract.md`. *Done (October 2026); revised for the
  simplified design together with this plan.*
- **Milestone 1 — Move existing code:** histogram, bin-count, record, CSV, pooling and
  normalization code moved from `morphobricks_preprocessing` commit `fd923c9` unchanged.
  *Done; released as `v0.1.0` (22 October 2026).*
- **Milestone 2 — Settings, scopes and fitting:** `NormalizationSettings`, `fit`, `apply`,
  `to_columns`/`from_columns`; the `biological_repeat` scope with repeat groups and both
  pooling weights; the `image` and `patch` scopes; per-time-point groups.
- **Milestone 3 — RGB:** the StarDist percentile method, the `rgb` image type and its checks.
- **Milestone 4 — Intensity checks:** per-image summaries and the warnings of §6.
- **Release `v0.2.0`**, pinned by the other bricks.
- **Then:** the preprocessing follow-up, then the CellPatcher follow-up (N6). Preprocessing
  steps P2 and P3 do not depend on this brick and may start earlier.

Deferred: before/after preview helpers for notebooks; image correction (N3).

## 11. Tests and acceptance criteria

Use simulated data generated in the tests; real data only as an optional check.

- Moved code keeps giving bit-identical histograms, CSVs and parameters.
- Every method's `apply` matches a direct implementation of its formula; `to_columns` and
  `from_columns` round-trip exactly.
- `biological_repeat` pools exactly the planes of its repeat group across conditions and
  technical repeats; pairing differently named repeats changes only the grouping; a repeat
  missing from some conditions gives its warning.
- `pooling="conditions"` gives each condition equal weight; both weights agree when the
  conditions have equal pixel counts, and at the image and patch scopes.
- The RGB method gives the same offset and scale to all three channels and, with the choice
  of §13, matches `csbdeep.utils.normalize(x, 1, 99.8, axis=(0, 1, 2))` on 8-bit images
  (CSBDeep is used only in the tests, as an optional check).
- Each intensity check fires on simulated shifted, saturated and constant images and stays
  silent on balanced data; a difference between conditions alone never fires.

## 12. Decisions

Made with the project owner, October 2026. N1–N28 are kept as the record; those marked
*withdrawn* or *replaced* no longer apply.

1. **N1** Two bricks: preprocessing (data validation, metadata, axes, image type, optional
   OME-Zarr conversion, statistics files) and normalization (methods, scopes, checks).
2. **N2** The normalization brick defines and computes the statistics and owns their file
   format; preprocessing runs them over the dataset.
3. **N3** Image correction is deferred to a future implementation.
4. **N4** *Replaced by N33.* (Every modality and method for all users.)
5. **N5** *Withdrawn:* no scikit-image or scikit-learn dependency (N33).
6. **N6** Order: normalization brick first, then preprocessing, then CellPatcher.
   Preprocessing keeps deprecated re-exports of the moved code for one version.
7. **N7** Statistics files keep the shared columns. *Colour statistics withdrawn (N33).*
8. **N8** Per-channel histograms are the statistics of every method.
9. **N9** *Replaced:* the image and patch scopes work as CellPatcher implements them today
   (N29).
10. **N10** *Withdrawn:* no experiment scope (N29).
11. **N11** *Withdrawn:* no condition scope (N29).
12. **N12** No batch scope; no per-stain scope.
13. **N13–N14** *Withdrawn:* no reference images, groups or published standards (N33).
14. **N15** *Replaced by N34:* the image type instead of the modality. **N15a** *Withdrawn.*
15. **N16** *Replaced by N33:* the image type decides the method; there is no purpose
    setting. **N16a** *Withdrawn.*
16. **N17–N18** *Withdrawn* (colour methods).
17. **N19** *Replaced:* no clipping for any method, as in StarDist.
18. **N20** RGB stays RGB; *the luminance option is withdrawn.*
19. **N21** Parameters as numeric columns, not JSON; settings stored once per run.
20. **N22** *Withdrawn* (histogram matching).
21. **N23** Only the columns of the method used are written.
22. **N24–N28** *Withdrawn* (colour statistics, modality profiles, optical density,
    references, luminance).

Added with the project owner on 26 October 2026:

23. **N29** Three scopes: `biological_repeat`, `image` and `patch`. The image and patch scopes
    work as before. The condition, technical-repeat and planned experiment scopes are
    removed; CellPatcher removes its condition and technical-repeat scopes straight away,
    without a deprecation period.
24. **N30** The `biological_repeat` scope pools one biological repeat across all conditions,
    technical repeats merged. Repeats with the same name match by default; the user pairs
    differently named repeats. A repeat found in only some conditions is used as it is, with
    a warning.
25. **N31** Without biological-repeat folders, the user confirms that all images of every
    condition are one biological repeat. Without technical-repeat folders, the images of the
    biological repeat are used directly.
26. **N32** Two pooling weights, per pixel (default) and equal per condition, kept so they
    can be compared on real data later.
27. **N33** Two image types only: `fluorescence`, with the four existing per-channel methods,
    and `rgb`, with StarDist's percentile normalization of the three channels together.
    Modalities, purpose, colour and stain methods, optical density, histogram matching,
    luminance, references, colour statistics and the sample-size study are removed from the
    plan.
28. **N34** The image type is detected from the metadata and confirmed by the user.
29. **N35** Preprocessing cannot continue until the user confirms that the biological repeats
    are comparable (prepared and acquired under the same circumstances). The confirmation is
    recorded and not asked again while the repeats are unchanged.
30. **N36** The user may mark one or more conditions as controls; recorded as metadata only
    for now.
31. **N37** The folder check warns, without blocking, about unbalanced data (a condition and
    biological repeat with fewer than half or more than twice the median number of images),
    inconsistent axes, pixel types, pixel sizes, channel counts and plane counts, and
    repeats missing from some conditions.
32. **N38** After the statistics, intensity checks warn about images and biological repeats
    more than 3 robust deviations away (limit adjustable), saturated images and images
    without spread; plots are shown in the notebook only.
33. **N39** RGB uses only the StarDist percentile method (default 1–99.8).

## 13. Still open

- **Percentile definition for `rgb`.** StarDist uses NumPy's default linear interpolation and
  adds 10⁻²⁰ to the scale, so a region without spread becomes 0 instead of being refused. The
  library's existing percentiles use the inverse CDF, and a zero scale is an error.
  Recommended: reproduce StarDist exactly for `rgb` (linear interpolation, which is exact from
  8- and 16-bit histograms, and the 10⁻²⁰ term), and leave the fluorescence methods
  unchanged.
- **Pooling weights** on real data: whether equal weighting per condition changes the
  parameters enough to matter (N32).
- **Minimum images per group** for the outlier check (provisional 5).
