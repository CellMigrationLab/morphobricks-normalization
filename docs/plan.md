# MorphoBricks Normalization Brick — Implementation Plan

> See also: [Preprocessing follow-up](followup_preprocessing.md) · [CellPatcher follow-up](followup_cellpatcher.md)

**Status:** design brief. Decisions N1–N28 were made with the project owner in October 2026
and are recorded in §12. Points still to be measured are listed in §13.
**Scope:** one importable library with every normalization method for microscopy images, the
statistics those methods need, the modality rules that decide which methods are meaningful,
and the file formats that carry statistics between bricks.
**Out of scope for this milestone:** image correction (flat-field, dark-field, background,
illumination), which is deferred (N3); user interfaces, which stay in the bricks that call the
library.

## Instructions for implementation

Keep the library free of widgets, notebooks and file-system walking: it receives pixel arrays,
statistics and settings, and returns statistics, parameters and normalized arrays. Never
modify source images. Never apply a transformation the user did not choose, and never offer a
method that the chosen modality and purpose make meaningless. Implement every algorithm once
and reuse it across modalities; a modality is configuration, not code. Every error explains
in plain language what happened and what to do, in the style of the other MorphoBricks bricks.

## 1. Role in MorphoBricks

```
morphobricks-preprocessing ──▶ statistics files ──▶ CellPatcher ──▶ normalized patches
            │                        ▲                    │
            └── calls ──▶ morphobricks-normalization ◀── calls
```

- **morphobricks-preprocessing** validates the input data, walks the folder structure, reads
  image metadata, asks the user for the axis order and the imaging modality, optionally
  converts images to OME-Zarr, and writes the statistics files. It computes the statistics
  with this library and uses it to preview normalized images in its notebook.
- **morphobricks-normalization** (this brick) defines and computes the statistics, owns
  their file formats, readers and writers, implements every method, and holds the modality
  profiles.
- **CellPatcher** reads the statistics files and normalizes patches with this library.

The existing statistics and normalization code moves here from the preprocessing package
unchanged before anything new is added (N6, §10).

## 2. Concepts

| Term | Meaning |
|---|---|
| **Modality** | The imaging technique, declared by the user (§5). It selects the allowed methods, scopes and defaults. |
| **Purpose** | `model_input` (deep learning, visualisation, segmentation) or `quantitative` (intensity is the measurement). It further restricts methods and scopes (§6). |
| **Method** | One normalization algorithm (§3). |
| **Source scope** | Which pixels share one set of measured parameters: experiment, condition, biological repeat, technical repeat, image or patch (§4). It decides which differences are removed. |
| **Target** | What every source is mapped onto: a fixed scale (linear methods) or a reference image, reference group or published standard (§4.2). One target per run. |
| **Statistics** | Per-plane summaries computed once over the dataset (§7), so parameters for any scope can be derived without rereading images. |
| **Parameters** | The fitted numbers a method applies: an offset and scale, optical-density levels, a stain matrix, and so on (§8). |

Every method follows one interface:

- `fit(statistics, settings) → Parameters` for a source scope (and, where needed, for the
  target);
- `Parameters.apply(pixels) → normalized pixels`;
- `Parameters.to_columns() / from_columns()` so parameters are stored as named numeric columns
  and read back exactly (N21).

## 3. Method families

| Family | Methods | Transform | Statistics needed |
|---|---|---|---|
| **A. Linear per channel** | min-max, percentile range, z-score, robust z-score | `(x − offset) / scale`, each channel separately | Plane histograms |
| **B. Linear shared across channels** | Same four methods | One offset and scale for all channels, from the pooled channel histograms; keeps colour ratios | Plane histograms |
| **C. Optical density** | OD then a family-A or B method | `OD = −log₁₀(I / I₀)` per channel, then a linear method on OD | Plane histograms (OD is monotonic, so histograms convert exactly) |
| **D. Histogram matching** | Histogram matching | Maps each source value to the target value with the same cumulative frequency | Plane histograms of the source scope and of the reference |
| **E. Colour and stain** | Ruifrok colour deconvolution, Reinhard, Macenko, Vahadane | Joint transforms of R, G and B (§3.2) | Colour statistics (§7.2) |
| **Option** | Luminance (grayscale) conversion for RGB | `Y = 0.2126 R + 0.7152 G + 0.0722 B`, then family A | Plane histograms of the converted planes |

### 3.1 Histogram matching

The binned histograms already in `plane_histograms.csv` carry everything histogram matching
needs. The mapping is built from the cumulative histogram of the source scope and that of the
reference: each source bin maps to the target value at the same cumulative frequency. It is
exact for 8- and 16-bit images with one bin per value, and within half a bin otherwise. The
mapping itself is never stored; a patch records its source group and the reference, and the
mapping is rebuilt from the two histograms (N22).

### 3.2 Colour and stain methods

All start from optical density, `OD = −log₁₀(I / I₀)` per channel, over tissue pixels only
(§7.2).

| Method | Fitted per source | Fitted per target | Output |
|---|---|---|---|
| **Ruifrok** | Nothing (published or user-measured stain matrix, fixed for the run) | Nothing | Per-stain amounts (e.g. hematoxylin, eosin, DAB), each then normalized with a family-A method whose group-scope parameters come from the deconvolved pixel samples; or recombined to RGB |
| **Reinhard** | Mean and SD of L, a, b over tissue pixels | Same | RGB: per Lab channel `(x − μₛ)/σₛ · σₜ + μₜ`, converted back |
| **Macenko** | Stain vectors from the OD plane of the two main eigenvectors and the 1st/99th percentile angles; 99th-percentile stain amounts | Same | RGB rebuilt from the target stain vectors and rescaled amounts |
| **Vahadane** | Stain vectors from sparse non-negative matrix factorization (scikit-learn dictionary learning, seeded); 99th-percentile stain amounts | Same | As Macenko |

- **Stain labels.** Fitted vectors are matched to reference vectors to name them; built-in
  references cover H&E and H-DAB, and users can supply their own (N17).
- **Output.** RGB results are clipped to the valid range, for colour methods only (N19), and
  returned as float32 in 0–1 by default, or uint8 RGB as an option.
- **Patch scope.** Macenko and Vahadane need at least 1,000 tissue pixels in the patch; a
  patch with fewer is discarded and reported, like a patch without intensity spread (N18).

## 4. Scopes and targets

### 4.1 Source scopes

| Scope | Pixels pooled | Statistics | Removes differences between |
|---|---|---|---|
| `experiment` | All images | Statistics files | Nothing; keeps every difference (N10) |
| `condition` | Same condition | Statistics files | Conditions (warning shown, N11) |
| `biological_repeat` | Same condition and biological repeat | Statistics files | Conditions and biological repeats |
| `technical_repeat` | Same condition, biological and technical repeat | Statistics files | Conditions, biological and technical repeats |
| `image` (default) | All Z and T planes of one image | Statistics files, or the image's own pixels when no files exist (N9) | All of the above |
| `patch` | The patch's own pixels | Patch pixels | Everything outside the patch |

Parameters are always per channel unless the modality profile prescribes shared parameters
(family B) or the method is a colour method (family E, where R, G and B are fitted jointly).
Time points are pooled by default, with an option to give each time point its own group.

Groups are identified by their full label path (`bio_rep1` under two conditions is two
groups). There is no batch scope and no per-stain scope (N12).

### 4.2 References

Chosen by the user in the preprocessing brick and recorded in its manifest (N13):

- **Reference image:** one image (or plane) of the experiment.
- **Reference group:** a set of images chosen by labels (for example every `control` image),
  pooled.
- **Published standard** (colour methods only): built-in stain vectors and maximum amounts
  for H&E and H-DAB (N14).

Uses:

- Histogram matching and colour methods map every source onto the reference.
- For linear methods, a reference group may act as the source of one parameter set applied
  to every image (for example controls), which keeps differences between conditions.

Without the preprocessing brick, a reference image can be given as a file path, and only the
image and patch scopes are available (N9).

## 5. Modalities

The modality is chosen by the user in the preprocessing brick, or in CellPatcher when
preprocessing is not used (N15). Metadata may suggest one (an RGB samples axis suggests a
brightfield or histology modality), never decide it. A modality profile lists the allowed
methods, scopes and defaults; only those are offered (N16).

| Modality | Default | Also allowed | Not offered | Notes |
|---|---|---|---|---|
| `fluorescence` (any number of channels) | Percentile 1–99.8 (A), image scope | Min-max, z-score, robust z-score (A); shared parameters (B) only when the user marks channels as ratiometric | Histogram matching across channels, colour methods | Channels are independent stains |
| `brightfield_gray` | Percentile (A) | Optical density (C), histogram matching (D) | Colour methods | OD makes cells bright and is physically meaningful |
| `brightfield_rgb` (unstained) | Shared linear (B) | Optical density with shared I₀ (C), luminance conversion (option) | Per-channel linear (A), per-channel histogram matching | Per-channel scaling would change colours |
| `phase_contrast` (incl. DIC) | Robust z-score (A), image scope | Percentile (A), histogram matching (D) | Min-max, colour methods | Halos dominate min-max; intensity is not quantitative |
| `histology_rgb` (H&E, IHC) | Macenko (E) to the published standard, image scope | Reinhard, Vahadane, Ruifrok (E) | Per-channel linear (A), histogram matching (D) | Needs colour statistics and a tissue mask |
| `electron_microscopy` (SEM, TEM) | Percentile (A), image scope | Histogram matching (D), robust z-score (A) | Colour methods | Contrast varies by session and detector |

## 6. Purpose

`purpose` is chosen together with the method (N16).

- **`model_input`** (default): every method and scope of the modality profile.
- **`quantitative`**: only choices that keep intensity differences between the groups being
  compared:
  - scopes: `experiment`, or a reference group as the single parameter source;
  - families A, B and C, plus Ruifrok with a fixed stain matrix;
  - not offered: image, patch, condition and repeat scopes; histogram matching; Reinhard,
    Macenko, Vahadane.

  For `phase_contrast` and `electron_microscopy`, the profile states that intensities are
  not quantitative, and `quantitative` offers only experiment-scope linear methods.

Choices outside the profile are refused with a message naming an allowed alternative.

## 7. Statistics

The library computes statistics; the preprocessing brick runs that computation over the
dataset and writes the files. All statistics files start with the shared MorphoBricks
columns (`position_id`, `image_file`, `z`, `c`, `t`, `condition`, `biological_repeat`,
`technical_repeat`, `source_relative_path`, `plane_id`) and continue with columns specific to
the statistic (N7). Which files are written depends on the modality and the methods its
profile allows.

### 7.1 Plane histograms (existing)

`plane_histograms.csv`, schema version 2, unchanged: one row per 2D plane and channel, with
exact extrema, pixel counts and binned counts. Used by families A–D and the luminance option.

### 7.2 Colour statistics (new)

For `histology_rgb`. One row per 2D RGB plane (all three channels together, so `c` is blank)
in `colour_statistics.csv`, schema version 1:

| Columns | Meaning |
|---|---|
| `total_pixel_count`, `tissue_pixel_count` | All pixels; pixels passing the tissue mask |
| `od_threshold`, `exclude_saturated`, `exclude_dark` | Tissue-mask settings used (N16a) |
| `i0_r`, `i0_g`, `i0_b`, `i0_source` | Background light per channel and how it was obtained (`estimated`, `fixed`, `user`) (N15a) |
| `lab_count`, `lab_sum_l`, `lab_sum_a`, `lab_sum_b`, `lab_sumsq_l`, `lab_sumsq_a`, `lab_sumsq_b` | Exact Reinhard summaries; pool across any scope by adding |
| `od_count`, `od_sum_r`, `od_sum_g`, `od_sum_b`, `od_cross_rr`, `od_cross_rg`, `od_cross_rb`, `od_cross_gg`, `od_cross_gb`, `od_cross_bb` | Exact OD covariance summaries; pool to the Macenko stain plane at any scope |
| `sample_count`, `sample_seed` | Pixels stored in the sample file for this plane |
| `schema_version` | `1` |

**Pixel samples** (`colour_samples.npz`, next to the CSV): for each `plane_id`, raw RGB
triplets of randomly chosen pixels (seeded, reservoir sampling so planes read in blocks give
the same result). Raw RGB rather than OD, so I₀ and the tissue mask can change later without
rereading images. Group-level Macenko and Vahadane fits use pooled samples, weighted by each
plane's tissue pixel count. The default sample size is provisional and will be set by the
sample-size study (§13).

### 7.3 Defaults

| Setting | Default |
|---|---|
| I₀ | Estimated per channel as the 99th percentile of each plane's pixels; fixed (255 for 8-bit) or user-given as options (N15a) |
| Tissue mask | OD above 0.15 in at least one channel; saturated and near-black pixels excluded; all configurable (N16a) |
| Pixel sample | Provisional; set by the study (§13) |
| Seed | Fixed and recorded |

## 8. Parameters as columns

Parameters are stored as numeric columns, never as JSON (N21). Only the columns of the
method used in a run are written (N23); targets are written once in the run configuration of
the calling brick, not on every row (N21).

| Method | Per-row columns (source) | Run-level (target and fixed settings) |
|---|---|---|
| Families A, B | `normalization_offset`, `normalization_scale` (one value per channel, `;`-separated) | Method, scope, purpose |
| Family C | `normalization_i0` (per channel, `;`-separated), `normalization_offset`, `normalization_scale` | As above |
| Family D | `normalization_reference` (the reference's identifier); the mapping is rebuilt from histograms | Reference definition |
| Ruifrok | `normalization_i0_r`, `_g`, `_b` | Stain matrix (3×3) |
| Reinhard | `normalization_lab_mean_l`, `_a`, `_b`, `normalization_lab_std_l`, `_a`, `_b` | Target means and SDs |
| Macenko, Vahadane | `normalization_stain_<name>_r`, `_g`, `_b` for each stain, `normalization_max_<name>` for each stain, `normalization_i0_r`, `_g`, `_b` | Target stain vectors, maximum amounts, I₀; Vahadane's sparsity and seed |

Common per-row columns for every method: `normalization_method`, `normalization_scope`,
`normalization_group`, `normalization_exact`. Stain names come from the stain labels
(`hematoxylin`, `eosin`, `dab`).

## 9. Dependencies

Required (N5): `numpy`, `scikit-image` (colour space conversions), `scikit-learn`
(Vahadane's dictionary learning). Colour and stain methods are implemented here in NumPy,
not taken from TIAToolbox or StainTools. No GPU or deep-learning dependency.

## 10. Milestones

- **Milestone 0 — Contract:** `docs/contract.md` with the method catalogue, parameter columns,
  statistics schemas, modality profiles, purpose rules, references and validation messages.
  *Done (October 2026).*
- **Milestone 1 — Move existing code:** a new package `morphobricks_normalization` with the
  histogram, bin-count, record, plane-histogram CSV, pooling and normalization code moved from
  `morphobricks_preprocessing` unchanged, with its tests. Outputs are bit-identical; a pinned
  release is published for the other bricks (N6). *Code moved from preprocessing commit
  `fd923c9` (October 2026); release pending.*
- **Milestone 2 — Method framework and families A–D:** the common interface; experiment scope
  and references; shared linear, optical density, histogram matching and luminance conversion.
- **Milestone 3 — Modalities and purpose:** the six profiles, purpose rules, and refusal
  messages naming allowed alternatives.
- **Milestone 4 — Colour:** tissue mask, I₀ estimation, colour statistics and pixel samples
  (computed in blocks for whole-slide images), stain labelling, Ruifrok, Reinhard, Macenko,
  Vahadane, clipping and output types.
- **Milestone 5 — Preview helpers and documentation:** functions that return before/after
  thumbnails and parameter tables for notebooks (no widgets), and user documentation.
- **Then:** the preprocessing follow-up, then the CellPatcher follow-up (N6).

## 11. Tests and acceptance criteria

Use simulated data generated in the tests; real data only as an optional check.

- Moved code gives bit-identical histograms, CSVs and parameters to the preprocessing package
  version it came from.
- Every method's `apply` matches a direct implementation of its formula; `to_columns` and
  `from_columns` round-trip exactly.
- Family B gives the same scale to every channel; family C matches OD computed on pixels;
  histogram matching is exact for 8- and 16-bit images and within half a bin otherwise.
- Pooled colour summaries equal direct calculations on all tissue pixels at every scope.
- Macenko and Vahadane recover known stain vectors from synthetic images mixed from known
  vectors and amounts; stain labels are assigned correctly; results are identical with the
  same seed.
- Blockwise colour statistics equal whole-plane statistics.
- Every modality profile offers exactly its allowed methods and scopes, and `quantitative`
  refuses the choices of §6 with the documented message.

## 12. Decisions

Made with the project owner, October 2026.

1. **N1** Two bricks: preprocessing (data validation, metadata, axes, modality, optional
   OME-Zarr conversion, statistics files) and normalization (all methods). The preprocessing
   notebook calls normalization for previews; CellPatcher calls it to normalize patches.
2. **N2** The normalization brick defines and computes all statistics and owns their file
   formats, readers and writers; preprocessing runs them over the dataset.
3. **N3** Image correction is deferred to a future implementation.
4. **N4** The design covers every modality and method of §5 for all users, not a single lab's
   data.
5. **N5** scikit-image and scikit-learn are required dependencies; colour methods are
   implemented here.
6. **N6** Order: normalization brick first (starting with moving existing code unchanged),
   then the preprocessing update, then CellPatcher. Preprocessing keeps deprecated re-exports
   of the moved code for one version.
7. **N7** Statistics files keep the shared columns; the remaining columns depend on the
   statistic, chosen by the modality's methods. Colour statistics get their own CSV schema.
8. **N8** Per-channel histograms stay the statistics for families A–D; colour methods use
   colour statistics.
9. **N9** Without preprocessing, CellPatcher offers only the image and patch scopes, computed
   from pixels, with a reference image given as a path.
10. **N10** Add the `experiment` scope.
11. **N11** Keep the `condition` scope, with its warning.
12. **N12** No batch scope; no per-stain scope.
13. **N13** References (image or group) are chosen in the preprocessing brick, both options
    offered.
14. **N14** Colour targets: a reference image, a reference group, or published standard
    vectors; the published standard is the default.
15. **N15** The modality is asked in preprocessing, and in CellPatcher when preprocessing is
    not used. **N15a** I₀ is estimated by default, with fixed and user values as options.
16. **N16** Once the modality and purpose are chosen, only meaningful methods are offered.
    **N16a** Tissue mask defaults: OD threshold 0.15, saturated and near-black pixels
    excluded, configurable.
17. **N17** Stain labels by matching against reference vectors, with built-in H&E and H-DAB
    references.
18. **N18** Macenko and Vahadane at patch scope need about 1,000 tissue pixels; patches below
    are discarded and reported.
19. **N19** Clipping only for colour methods.
20. **N20** RGB stays RGB; luminance (grayscale) conversion is an option, never the default.
21. **N21** Parameters as numeric columns, not JSON; targets stored once per run.
22. **N22** Histogram matching uses the existing binned histograms; patches record the source
    group and reference, and the mapping is rebuilt.
23. **N23** Only the columns of the method used are written.
24. **N24** Colour statistics: exact summaries plus pixel samples in a binary file next to
    the CSV.

Added with the project owner during Milestone 0 (contract review), October 2026:

25. **N25** Modality profiles offer only the methods they name; where a family is named, all
    its methods (contract §5).
26. **N26** For optical density, I₀ is estimated per image (all Z and T planes, per time point
    when time points are separate groups), so patches may span planes; colour methods keep
    I₀ per 2D plane (contract §3.4, §9.1).
27. **N27** Every row whose method uses a reference records it in `normalization_reference`,
    as an identifier relative to the experiment root, never an absolute path (contract
    §10.1).
28. **N28** Luminance is computed from pixels: at the image and patch scopes, or with a
    reference image as the single parameter source (also for `quantitative`). Luminance
    histograms for group scopes are deferred (contract §3.6, §12).

## 13. Still to measure

- **Pixel sample size.** How stable the stain vectors, maximum amounts and Reinhard
  statistics are at 5,000, 10,000, 20,000 and 50,000 pixels per plane, on real H&E slides,
  like the binning study of the preprocessing brick. Deferred; the default stays provisional
  until then.
- **Default parameters** of the colour methods (percentile angles, Vahadane sparsity) on the
  same slides.
- **I₀ estimation** on images with little background: whether the 99th percentile still finds
  the glass brightness, or a fixed value should be used when the background fraction is small.
