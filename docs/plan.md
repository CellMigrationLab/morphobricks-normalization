# MorphoBricks Normalization Brick — Implementation Plan

> See also: [Preprocessing follow-up](followup_preprocessing.md) · [CellPatcher follow-up](followup_cellpatcher.md)

**Status:** design brief, simplified on 26 October 2026 (decisions N29–N40, §12), with
channels moved into columns on 27 October 2026 (N41–N45). The earlier
design with six modalities, two purposes, colour and stain methods and reference targets
(N4–N28) is withdrawn where §12 says so. What is still to be measured is listed in §13.
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
| **Plane** | One XYC plane: every channel of one image at one Z and T position. One row of a table (N41). |
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
x = (x - mi) / (ma - mi + 1e-20)  # float32, no clipping by default
```

Its examples use `pmin=1`, `pmax=99.8`. For RGB images they normalize "channels jointly" with
`axis=(0, 1, 2)`, so the percentiles are taken over all pixels of all three channels and every
channel gets the same offset and scale; colour ratios are kept.

In this library, `P_low` and `P_high` come from one pooled histogram of the R, G and B planes
of the group (§4), so the method works at every scope from `plane_histograms.csv` without
rereading images. It reproduces StarDist exactly (N40): the percentiles use NumPy's default
linear interpolation, computed from the cumulative counts (exact for 8- and 16-bit
histograms), and the scale includes the 10⁻²⁰ term, so a region without spread becomes 0
instead of being refused. The fluorescence methods keep the library's inverse-CDF percentiles
and zero-scale error.

Applying the parameters repeats CSBDeep's float32 arithmetic, so normalized RGB pixels are
bit-identical to StarDist's (N48). With equal weight per condition (§4.2) the same linear
rule uses the cumulative weights as positions, which gives the pixel rule exactly when every
weight is 1 (N49).

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
| Biological repeat shifted | The other biological repeats of the same condition | Median of the repeat's image medians more than 3 robust deviations of the other repeats' images away (at least 5 images in the other repeats) |
| Saturated image | — | More than 1 % of the pixels at the type's maximum (integer images) |
| No intensity spread | — | 1st and 99.8th percentiles equal: the image cannot be normalized at the image scope |

Comparisons stay within one condition, so a treatment effect is never reported as a shift.
Limits are settings; a group with fewer than 5 images is not tested for outliers (provisional).
The preprocessing notebook shows the warnings and draws the plots (follow-up P5).

## 7. Statistics

`plane_histograms.csv`, schema version 3 (N41): one row per XYC plane, that is per image, Z
and T, with every channel's exact extrema, pixel counts and binned counts side by side in
per-channel columns named as in CellProfiler (`Histogram_Count_C0_00000`, …, N42). The
histograms themselves are those of schema version 2; only the layout changes. It serves both
image types (an RGB plane is one row with channels C0, C1, C2), every scope and the intensity
checks. No other statistics file is written (N33).

### 7.1 Channels across columns (every MorphoBricks table)

Every MorphoBricks table that holds values per channel puts the channels side by side in
columns, never in separate rows (N41):

- One row per plane (or image, or patch), not per channel; such tables have no `c` column.
  One exception: CellPatcher's single-channel patches for deep-learning training, one row
  per patch with its channel in `c`, for the channels the user selects (N45).
- Per-channel columns follow CellProfiler's naming, `<Category>_<Feature>_C<index>` with an
  optional `_<parameter>` after the channel, e.g. `Intensity_MinIntensity_C0`,
  `Histogram_Count_C1_00042`, `Normalization_Offset_C2`. The channel token is the zero-based
  channel index (N42); channel names stay in the metadata.
- The MorphoBricks shared columns (`condition`, `biological_repeat`, `technical_repeat`,
  `position_id`, `image_file`, `z`, `t`, `source_relative_path`, `plane_id`, …) keep their
  names and format (N44).
- Every image of an experiment has the same number of channels; a different count is an error
  (N43).
- The feature-extraction bricks follow the same rule (`Intensity_C0`, `Intensity_C1`, …).

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
| `Normalization_Offset_C<i>`, `Normalization_Scale_C<i>` | One pair per channel (N41); for `rgb` the shared value in C0, C1 and C2 |

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
  pooling weights; the `image` and `patch` scopes; per-time-point groups. *Done (26 October
  2026): modules `settings`, `scopes`, `fitting` and `issues`, and `WeightedHistogram` in
  `pooling`; fluorescence methods only.*
- **Milestone 2b — Channels across columns:** `plane_histograms.csv` schema version 3, with
  one row per XYC plane and CellProfiler-style per-channel columns; `PlaneRecord`, the reader
  and writer, grouping and fitting move to whole planes; the parameter columns
  `Normalization_Offset_C<i>` and `Normalization_Scale_C<i>` (N41–N44). *Done (27 October
  2026): `PlaneRecord.histograms` (one per channel), schema version 3 in `csv_io` (versions
  1 and 2 refused), a channel index in `PooledHistogram` and `fit`, and `Parameters.channels`
  and `select` for single-channel patches (N46); tested against a version-2 fixture written
  by the Milestone 2 code (`tests/data`). Other bricks pin the merge commit.*
- **Milestone 3 — RGB:** the StarDist percentile method, the `rgb` image type and its checks.
  *Done (27 October 2026): module `rgb` (`rgb_percentile`, `apply_rgb` with CSBDeep's
  float32 arithmetic, `check_rgb`) and `linear_percentile` in `pooling`; every scope and both
  pooling weights, from `plane_histograms.csv` rows or pixels. Tested against
  `np.percentile` and a direct NumPy implementation on simulated 8- and 16-bit images, and
  bit for bit against `csbdeep.utils.normalize` when CSBDeep is installed (N48, N49).*
- **Milestone 4 — Intensity checks:** per-image summaries and the warnings of §6.
  *Done (27 October 2026): modules `summaries` (`image_summaries`, `ImageSummary`,
  `summary_columns`) and `checks` (`intensity_checks`, `IntensityCheckSettings`), and
  `Issue.channel` (N50–N53). Summaries tested against NumPy on simulated pixels; each check
  fires on simulated shifted, saturated and constant images, stays silent on balanced data
  with a treatment effect, and matches a direct implementation of contract §7.2.*
- **Release `v0.2.0`**, pinned by the other bricks.
- **Then:** the preprocessing follow-up, then the CellPatcher follow-up (N6). Preprocessing
  steps P2 and P3 do not depend on this brick and may start earlier; its step P3b (writing
  schema version 3) follows Milestone 2b. Until the release, the other bricks pin this brick
  by commit.

Deferred: before/after preview helpers for notebooks; image correction (N3).

## 11. Tests and acceptance criteria

Use simulated data generated in the tests; real data only as an optional check.

- Moved code keeps giving bit-identical histograms and parameters; schema version 3 holds
  exactly the histograms of version 2, one channel group per former row.
- A table round-trips through the writer and reader; images with different channel counts are
  refused; every per-channel column name follows §7.1.
- Every method's `apply` matches a direct implementation of its formula; `to_columns` and
  `from_columns` round-trip exactly.
- `biological_repeat` pools exactly the planes of its repeat group across conditions and
  technical repeats; pairing differently named repeats changes only the grouping; a repeat
  missing from some conditions gives its warning.
- `pooling="conditions"` gives each condition equal weight; both weights agree when the
  conditions have equal pixel counts, and at the image and patch scopes.
- The RGB method gives the same offset and scale to all three channels and matches `csbdeep.utils.normalize(x, 1, 99.8, axis=(0, 1, 2))` bit for bit on 8- and 16-bit images
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
34. **N40** The RGB method reproduces StarDist exactly: NumPy's linear-interpolation
    percentiles and the 10⁻²⁰ term added to the scale. Fluorescence methods are unchanged.

Added with the project owner on 27 October 2026:

35. **N41** Channels go across columns, not rows, in every MorphoBricks table: a plane is XYC,
    one row per plane (image, Z, T), with each channel's values in its own columns.
    `plane_histograms.csv` becomes schema version 3; `normalization_offset` and
    `normalization_scale` become per-channel columns.
36. **N42** Per-channel columns follow CellProfiler's naming,
    `<Category>_<Feature>_C<index>[_<parameter>]`, with the zero-based channel index as the
    channel token.
37. **N43** All images of an experiment must have the same number of channels; a different
    count is an error (it was a warning).
38. **N44** The MorphoBricks shared columns keep their names and format; only per-channel
    columns follow the CellProfiler pattern.
39. **N45** CellPatcher keeps single-channel patches for deep-learning training, where rows
    are patches rather than channel measurements, and lets the user choose the channels: one,
    several or all (all is the earlier `split_channels=True`). Each selected channel gives
    its own single-channel patch and row, with the channel in `c`. Feature extraction keeps
    multichannel patches with the channels in columns.

Added with the project owner on 27 October 2026, for Milestone 2b:

40. **N46** `Parameters` records the image's channel indices it holds (`channels`, by
    default all). `select([i])` gives the parameters of a single-channel patch of channel
    `i`; `to_columns` writes only those channels' pairs, and `from_columns` skips channels
    whose two columns are blank.
41. **N47** `from_columns` reads only the `Normalization_Offset_C<i>` and
    `Normalization_Scale_C<i>` columns. CellPatcher converts the `;`-separated
    `normalization_offset` and `normalization_scale` of its earlier tables itself (follow-up
    C1).

Added with the project owner on 27 October 2026, for Milestone 3:

42. **N48** RGB `apply` repeats CSBDeep's float32 arithmetic: the offset and the rebuilt
    high percentile (`offset + scale − 10⁻²⁰`) are rounded to float32 and
    `(x − mi) / (ma − mi + 10⁻²⁰)` is computed in float32, so normalized pixels are
    bit-identical to `csbdeep.utils.normalize`. Fluorescence keeps its float64 `apply`.
43. **N49** With `pooling="conditions"`, the RGB percentiles use the linear rule with the
    cumulative weights as positions: `h = (W − 1)·q/100` over the total weight `W`, with the
    weights scaled so that each pixel of the condition with the most pixels weighs 1. It
    equals the pixel rule exactly when every weight is 1 (one condition, or equal pixel
    counts).

Added with the project owner on 27 October 2026, for Milestone 4:

44. **N50** `image_summaries` returns `ImageSummary` objects with named per-channel values;
    `to_row()` gives the columns of contract §7.1 as a dictionary and `summary_columns` their
    order. The library writes no summary file, so no new file format is defined.
45. **N51** The limits are an `IntensityCheckSettings(outlier_limit, saturation_limit,
    min_images)` with `to_dict`/`from_dict`, passed as `intensity_checks(summaries,
    settings=None)` (the contract's `check_settings` keyword would shadow the function of that
    name).
46. **N52** One `Issue` per finding (image or repeat, channel and compared value). `Issue`
    gains an optional `channel` (None for other warnings), so plots can mark flagged images
    per channel without parsing messages.
47. **N53** The repeat-shift check also needs at least `min_images` images in the other
    repeats of the condition, like the outlier check.

## 13. Still to measure

- **Pooling weights** on real data: whether equal weighting per condition changes the
  parameters enough to matter (N32).
- **Minimum images per group** for the outlier and repeat-shift checks (provisional 5).
