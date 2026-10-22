# Normalization Brick Contract (Milestone 0)

> See also: [Plan](plan.md) · [Preprocessing follow-up](followup_preprocessing.md) ·
> [CellPatcher follow-up](followup_cellpatcher.md)

This document fixes the rules other bricks can rely on: the method catalogue, scopes and
references, modality profiles and purpose rules, the statistics files, the parameter
columns, the defaults and the validation severities. It implements decisions N1–N28 of the
[plan](plan.md) (§12); it does not change any of them.

- Changing a file format (§8) requires a new `schema_version` of that file.
- Changing any other rule requires the project owner's approval and an entry in plan §12.
- Values marked **Provisional** are left to measurement by plan §13. They are part of the
  contract as defaults, are recorded with every result, and may change once measured
  without a new schema version (§9.4).

| Section | Implemented in |
|---|---|
| 1. Existing code | Milestone 1 (moved unchanged; done) |
| 2. Interface; 3.1–3.6 families A–D and luminance; 4. Scopes and references; 10. Columns | Milestone 2 |
| 5. Modality profiles; 6. Purpose; 7. Refusal messages | Milestone 3 |
| 3.7 Colour and stain methods; 8.3–8.5 Colour statistics; 9. Colour defaults | Milestone 4 |
| 11. Validation | With the milestone of each check |

Notation: `x` is a pixel value in source units; `n` a pixel count; channels are numbered
from 0 in the image's channel order; R, G, B are channels 0, 1, 2 of an RGB image.

---

## 1. Existing code (Milestone 1)

### 1.1 Modules moved unchanged

The following modules move from `morphobricks_preprocessing` (as of commit
[`fd923c9`](https://github.com/CellMigrationLab/morphobricks-preprocessing/commit/fd923c95f4b36064e82ba4d4df8a294b7280d140),
the commit CellPatcher pins) to the package `morphobricks_normalization`, with their tests
(`test_histogram.py`, `test_csv_io.py`, `test_pooling.py`, `test_normalization.py`) and the
test helpers they use (`conftest.py`, `simulation.py`, copied: other preprocessing tests
keep using theirs):

| Module | Public names |
|---|---|
| `dtypes.py` | `MAX_BIN_COUNT`, `DEFAULT_FLOAT_BIN_COUNT`, `MIN_FLOAT_BIN_COUNT`, `INTEGER_DTYPES`, `FLOAT_DTYPES`, `UnsupportedDtypeError`, `check_supported`, `is_integer`, `bit_depth`, `default_bin_count`, `validate_bin_count` |
| `histogram.py` | `PlaneHistogram`, `compute_plane_histogram`, `compute_block_histogram`, `estimate_histogram_memory` |
| `records.py` | `SINGLE`, `LEVELS`, `PlaneRecord`, `make_plane_id`, `group_records` |
| `csv_io.py` | `SCHEMA_VERSION`, `OUTPUT_FILENAME`, `METADATA_COLUMNS`, `BIN_COLUMN_PREFIX`, `CsvFormatError`, `bin_column`, `write_plane_histograms`, `read_plane_histograms` |
| `pooling.py` | `DEFAULT_PERCENTILES`, `NoValidPixelsError`, `PooledStatistics`, `PooledHistogram`, `pooled_statistics` |
| `normalization.py` | `NORMAL_CONSISTENCY`, `NormalizationParameters`, `min_max`, `percentile_range`, `z_score`, `robust_z_score` |

"Unchanged" means:

- The code is the same, apart from the package name in imports and docstring
  cross-references.
- Histograms, CSV files and parameters are bit-identical to those of the preprocessing
  package at that commit (plan §11).
- Exceptions keep their types and messages (later milestones may raise subclasses of the
  same types, so existing `except` clauses keep working).
- The preprocessing brick keeps deprecated re-exports for one version (N6, follow-up P1).

Additions in later milestones extend these modules or add new ones; they never change the
behaviour of the names above.

### 1.2 Pixel types, histograms, group statistics, linear parameters

Until the preprocessing follow-up (P1) moves the text here, these rules are defined in the
[preprocessing contract](https://github.com/CellMigrationLab/morphobricks-preprocessing/blob/main/docs/contract.md)
and are part of this contract by reference:

| Rules | Preprocessing contract | Summary |
|---|---|---|
| Pixel types and bin counts | §3 | Integers int8–uint64 (default one bin per value up to 65,536 bins), float32 and float64 (default 10,000 bins); everything else rejected |
| Histogram semantics | §4 | Each plane binned over its own exact range; exact for 8- and 16-bit integers by default; every pixel within half a bin of its representative; blocks give the same row as the whole plane |
| Group statistics | §6 | Pooled (representative, count) pairs, never averaged per-plane statistics; inverse-CDF percentiles; exact min and max |
| Linear parameters | §7 | `(x − offset) / scale`; zero or non-finite scale is an error |

`normalization_exact` (§10) is `true` for histogram-based parameters exactly when every
pooled histogram has `max_representative_error == 0`.

---

## 2. Interface

### 2.1 Objects

| Object | Holds | Stored as |
|---|---|---|
| `NormalizationSettings` | Modality, purpose, method and its options, scope, parameter source, reference (§10.4) | Run configuration of the calling brick, once per run |
| Source statistics | Pooled plane histograms per channel (§1.2), pooled colour statistics (§8.3–8.5), or pixel arrays from which the library computes the same statistics | Not stored by the library |
| `Target` | What every source is mapped onto: nothing (fixed scale), a reference's fitted parameters, or a published standard (§4.4) | Run configuration, once per run (N21) |
| `Parameters` | The fitted numbers of one source group (§10) | Per-row columns (N21, N23) |

### 2.2 Functions

```python
check_settings(settings, *, statistics_files: bool) -> list[Issue]
target = fit_target(reference_statistics, settings)     # or published_standard("he")
parameters = fit(source_statistics, settings, target=target)
normalized = parameters.apply(pixels, channel_axis=...)
columns = parameters.to_columns()                       # dict[str, str]
parameters = Parameters.from_columns(columns, settings, target=target, statistics=None)
settings.to_dict(); NormalizationSettings.from_dict(...)
target.to_dict(); Target.from_dict(..., settings)
```

| Function | Rule |
|---|---|
| `check_settings` | Checks the modality profile (§5), the purpose (§6), the scope against the available statistics (§4.5) and the target (§4.4) using the settings alone, so a calling brick can refuse a choice before reading any image. Raises `MethodNotAllowedError` (a `ValueError`) with a message of §7; returns the warnings of §11.2 that follow from the settings alone. |
| `fit_target` | Fits a reference with the same algorithm as a source (every algorithm once). Methods without a target (families A–C, luminance) take `target=None`; Ruifrok's target is its fixed stain matrix, built from the settings without a reference (`ruifrok_target(settings)`). |
| `fit` | One entry point for every method; dispatches on `settings.method`. Never reads files and never walks folders. |
| `apply` | Returns a new array; the input is never modified. `channel_axis` gives the channel axis of `pixels`; it may be left out when the parameters have one channel. Output types: §3.8. |
| `to_columns` | Only the columns of the method used (N23), as text in the format of §10.3, in the order of §10.2. |
| `from_columns` | Rebuilds parameters equal to the fitted ones: `from_columns(to_columns(p)) == p` exactly. Needs the run-level `settings` and `target`; histogram matching also needs the statistics of the source group and the reference (N22, §3.5). |
| `to_dict` / `from_dict` | JSON-compatible dictionaries (§10.4) that round-trip exactly. |

The library never prints. Errors are raised as exceptions with plain-language messages;
warnings are returned as `Issue` objects (severity, code, message, affected items,
suggested action) for the calling brick to report (§11).

---

## 3. Method catalogue

### 3.1 Identifiers

The method identifier is a setting (`settings.method`); the written name is what
`normalization_method` holds (§10.1). Family A keeps the names written today.

| Family | `method` | Modifiers | Written name |
|---|---|---|---|
| A. Linear per channel | `minmax`, `percentile`, `zscore`, `robust_zscore` | — | `min-max`, `percentile <low>-<high>` (e.g. `percentile 1-99.8`), `z-score`, `robust z-score` |
| B. Linear shared across channels | as A | `shared_channels` | `shared ` + the A name, e.g. `shared percentile 1-99.8` |
| C. Optical density | as A | `optical_density=true` (with or without `shared_channels`) | `optical density + ` + the A or B name |
| D. Histogram matching | `histogram_matching` | — | `histogram matching` |
| E. Colour and stain | `ruifrok`, `reinhard`, `macenko`, `vahadane` | §3.7 | `ruifrok`, `ruifrok + ` + the A name of the amounts (§3.7.3), `reinhard`, `macenko`, `vahadane` |
| Luminance option | as A | `luminance=true` | `luminance + ` + the A name |

Percentile values in names are formatted with `:g` (as `percentile_range` does today).

### 3.2 Family A — linear per channel

Each channel `c` gets its own offset and scale from the pooled statistics of that channel
in the source group:

| `method` | offset | scale | Settings |
|---|---|---|---|
| `minmax` | exact minimum | maximum − minimum | — |
| `percentile` | `P(low)` | `P(high) − P(low)` | `percentile = [low, high]`, `0 ≤ low < high ≤ 100`, default `[1.0, 99.8]` |
| `zscore` | mean | population standard deviation | — |
| `robust_zscore` | median | MAD × `consistency` | `consistency`, default `NORMAL_CONSISTENCY` = 1.482602218505602 |

`apply`: `y_c = (x_c − offset_c) / scale_c`, computed in float64, returned as float32. No
clipping (N19). Non-finite pixels stay non-finite.

### 3.3 Family B — linear shared across channels

As family A, but one offset and one scale are computed from the pooled histograms of all
channels in `shared_channels` together (one `PooledHistogram` over those channels' planes),
and applied to each of them. Colour ratios between those channels are kept.

| `shared_channels` | Meaning |
|---|---|
| `null` | Family A |
| `"all"` | Every channel shares one offset and scale |
| list of at least two channel indices | Those channels share one offset and scale; every other channel keeps its own (family A, same linear method). For `fluorescence`, these are the channels the user marked as ratiometric (§5). |

### 3.4 Family C — optical density

Optical density of a value `x` with background light `I₀` (per channel, or shared):

```
OD(x) = −log₁₀( max(x, x_floor) / I₀ )
x_floor = 1 for integer images; 10⁻⁶ · I₀ for float images
```

Then a family-A or family-B method is fitted on OD values and applied to
`OD(x)`: `y = (OD(x) − offset) / scale`. Values above `I₀` give negative OD; they are kept
(no clipping, N19).

- **I₀:** per image (all its Z and T planes; per image and time point with
  `per_time_point=true`) and per channel, or one value per image for the channels in
  `shared_channels` (`brightfield_rgb` "shared I₀", §5) (N26). Modes (N15a): §9.1. Patches
  may span several Z or T planes: all planes of an image share its I₀, so a row's one value
  per channel describes every plane of the patch. A patch uses the I₀ of its image.
- **From histograms:** OD is decreasing, so each occupied bin of each plane becomes the pair
  (`OD(representative)` with its image's I₀, count) and the pairs are pooled with the
  definitions of §1.2. The OD minimum and maximum come from the exact extrema
  (`OD(pixel_max)`, `OD(pixel_min)`). Parameters are exact when every histogram is exact.
  Milestone 2 adds a constructor of `PooledHistogram` from (value, count) pairs; existing
  behaviour is unchanged.

### 3.5 Family D — histogram matching

Per channel. Let `S` be the pooled histogram of the source group and `R` that of the
reference (both §1.2), with values `s₀ < s₁ < …` and `r₀ < r₁ < …`, cumulative counts
`Cs(i)`, `Cr(j)` and totals `Ns`, `Nr`.

```
T(sᵢ) = r_j  for the smallest j with  Cr(j) · Ns ≥ Cs(i) · Nr
```

(integer arithmetic, so the comparison is exact). Each source value maps to the reference
value with the same cumulative frequency, with the inverse-CDF definition of the
percentiles.

- `apply` maps a pixel through `T` of the source value nearest to it (ties to the lower
  value); pixels below `s₀` or above the largest `sᵢ` map to `T(s₀)` or `T(s_last)`. The
  result is float32 in the reference's intensity units.
- Exact for 8- and 16-bit images with one bin per value; within half a bin otherwise.
- The mapping is never stored (N22): a row records its source group and the reference, and
  `from_columns` rebuilds `T` from the two histograms. At the patch scope the source
  histogram is that of the patch's own pixels, so raw patches are rehistogrammed on load.

### 3.6 Luminance option (RGB only)

```
Y = 0.2126 R + 0.7152 G + 0.0722 B
```

computed in float64 on source values, giving one channel; then a family-A method on `Y`.
`Y` planes are histogrammed with the float rules of §1.2 (default 10,000 bins), so their
parameters are approximate. Never the default (N20).

The statistics files hold no histograms of `Y`, so luminance is computed from pixels (N28):

- at the `image` and `patch` scopes, from the image's or patch's own pixels;
- with `parameter_source="reference"` and a reference image, plane or file: one parameter
  set fitted on the luminance of the reference's pixels (read by the calling brick) and
  applied to every image. This keeps the differences between images, so it is also offered
  for `quantitative` (§6).

Other scopes are refused (`luminance_needs_pixels`, §7). Group-scope luminance histograms
are deferred (§12).

### 3.7 Family E — colour and stain methods

#### 3.7.1 Common rules

| Rule | Detail |
|---|---|
| Input | 2D RGB planes: exactly three channels in R, G, B order. A patch spanning several Z or T planes is refused; split planes first (e.g. CellPatcher `z_policy`/`t_policy`). |
| RGB in 0–1 | Integer images divided by the type's maximum; float images used as they are and must lie in 0–1. |
| OD | §3.4 per channel, with the plane's `I₀` (§9.1). |
| Tissue pixel | Finite in all three channels; OD above `od_threshold` in at least one channel; not saturated (when `exclude_saturated`); not near-black (when `exclude_dark`). Defaults: §9.2. |
| Saturated | Any channel at the type's maximum (integer) or at or above 1.0 (float). |
| Near-black | Every channel below `dark_limit` × its `I₀` (default 0.05, i.e. OD above log₁₀ 20 ≈ 1.30 in all three; **Provisional**). |
| Fitting | Only tissue pixels are used for fitting; every pixel is transformed by `apply`. |
| Sample weights | At group and image scopes, sample-based fits (§8.4) use the pooled tissue samples of the group's planes. A tissue sample of plane `p` has weight `tissue_pixel_count_p / tissue_samples_p`, so each plane weighs exactly in proportion to its tissue pixels (plan §7.2). |
| Weighted statistics | Same definitions as §1.2 with weights instead of counts: the `q`-th percentile is the smallest value whose cumulative weight reaches `q/100` of the total weight (at least the first value). |
| Settings match the statistics | When colour statistics are used, the tissue-mask and `I₀` settings must equal those recorded in `colour_statistics.csv`; otherwise the run is refused (§11) and the preprocessing brick recomputes the statistics (follow-up P4). |

#### 3.7.2 Stain sets and labels (N17)

A stain set names the stains and gives their reference OD vectors (unit length, R, G, B):

| `stain_set` | Stains | Reference vectors (before normalization to unit length) |
|---|---|---|
| `he` (default) | `hematoxylin`, `eosin` | (0.650, 0.704, 0.286), (0.072, 0.990, 0.105) |
| `hdab` | `hematoxylin`, `dab` | (0.650, 0.704, 0.286), (0.268, 0.570, 0.776) |
| user | 2 or 3 names matching `[a-z][a-z0-9_]*` | One non-negative vector per stain |

Values from Ruifrok & Johnston (2001) (scikit-image's `rgb_from_hed` rounds them to two
decimals). Fitted vectors
(Macenko, Vahadane) are labelled by choosing the assignment to the set's reference vectors
with the largest sum of cosine similarities. A label with cosine similarity below 0.9
(**Provisional**) is a warning (§11.2).

#### 3.7.3 Ruifrok colour deconvolution

- **Stain matrix `M`:** rows are the unit OD vectors of the stains, fixed for the run:
  the stain set's vectors (default), or user-measured ones. With two stains, the third row
  is their normalized cross product, named `residual`. `M` must be invertible.
- **Amounts:** `A = OD · M⁻¹` per pixel (OD as a row vector of R, G, B), one amount per row
  of `M`.
- **Output `ruifrok_output`:**
  - `amounts` (default): the named stains' amounts (not `residual`), in matrix order, each
    normalized with a family-A method (`amount_method`, default `percentile` 1–99.8) whose
    parameters come from the tissue pixels' amounts of the source group: weighted samples
    at group and image scopes, all tissue pixels at the patch scope. Float32, no clipping.
  - `rgb`: RGB rebuilt from the amounts of the stains in `ruifrok_stains` only, the others
    set to zero: `RGB = 10^(−A_sel · M_sel)`, clipped to 0–1 (N19), relative to `I₀`.
- Nothing is fitted for the stains themselves, so Ruifrok is allowed for `quantitative`
  (§6).

#### 3.7.4 Reinhard

- Lab: `skimage.color.rgb2lab` of RGB in 0–1 (D65, 2° observer).
- **Fitted** per source and per target: mean `μ` and population standard deviation `σ` of
  L, a, b over tissue pixels. At group and image scopes these come exactly from the
  summaries `lab_*` of §8.3: `μ = lab_sum / lab_count`,
  `σ = sqrt(lab_sumsq / lab_count − μ²)`.
- **`apply`:** per Lab channel `(x − μₛ) / σₛ · σₜ + μₜ`, converted back with
  `skimage.color.lab2rgb`, clipped to 0–1.
- **Target:** a reference image or reference group. There is no published Reinhard standard,
  so a Reinhard run must choose a reference (§4.4).

#### 3.7.5 Macenko

1. **Stain plane.** Covariance of tissue OD: `C = Σ(od odᵀ) / n − μ μᵀ` with
   `μ = Σ od / n`. At group and image scopes from the exact summaries `od_*` of §8.3; at the
   patch scope from the patch's tissue pixels. `e₁`, `e₂` are the eigenvectors of the two
   largest eigenvalues (`numpy.linalg.eigh`), each signed so that its components sum to a
   positive value.
2. **Angles.** For each tissue pixel, `φ = atan2(od · e₂, od · e₁)`; `φ_min` and `φ_max`
   are the `angle_percentile`-th and `(100 − angle_percentile)`-th weighted percentiles
   (default 1, i.e. 1st/99th; **Provisional**). Samples at group and image scopes; all
   tissue pixels at the patch scope.
3. **Stain vectors.** `e₁ cos φ_min + e₂ sin φ_min` and `e₁ cos φ_max + e₂ sin φ_max`,
   normalized to unit length and labelled (§3.7.2). `V` has one row per stain.
4. **Amounts.** Least squares `A = OD · V⁺` (`V⁺` the pseudo-inverse); the maximum amount
   of each stain is its `max_percentile`-th weighted percentile over tissue pixels (default
   99; **Provisional**).
5. **`apply`:** `A = OD · Vₛ⁺`, scaled per stain by `maxₜ / maxₛ`, then
   `OD_out = A · Vₜ` and `RGB = 10^(−OD_out)`, clipped to 0–1. The result is relative to the
   target's `I₀`.

#### 3.7.6 Vahadane

As Macenko, except step 1–3: the stain vectors are the rows of a non-negative dictionary
of two atoms learned on tissue OD with sparse non-negative codes (scikit-learn dictionary
learning, `positive_dict=True`, `positive_code=True`, `alpha = sparsity`, default 0.1
**Provisional**, `random_state = seed`), normalized to unit length and labelled. When the
sample weights of a group differ, the dictionary is learned on a seeded resample, with
replacement, of the pooled tissue samples, drawn with probability proportional to their
weights, of the same size; when they are equal, on the pooled tissue samples themselves.
Amounts and `apply` as Macenko steps 4–5. Results are identical with the same seed.

#### 3.7.7 Minimum tissue

| Method | Needs | Below it, at the patch scope | Below it, at other scopes |
|---|---|---|---|
| Macenko, Vahadane | `min_tissue_pixels` tissue pixels (default 1,000, N18) | Patch discarded and reported (§11.2) | Error (§11.1) when the group's `tissue_pixel_count` is below it; a warning (`few_tissue_samples`, §11.2) when only its tissue samples are fewer |
| Reinhard | At least one tissue pixel and `σ > 0` in L, a and b | Patch discarded and reported | Error |
| Ruifrok `amounts` | Non-zero spread of every amount (family-A rule) | Patch discarded and reported | Error |

### 3.8 Output types and exactness

| Methods | `apply` returns | Clipping |
|---|---|---|
| A, B, C, D, luminance, Ruifrok `amounts` | float32 | None (N19) |
| Reinhard, Macenko, Vahadane, Ruifrok `rgb` | RGB float32 in 0–1 (default), or uint8 `rint(255 · value)` with `output_dtype="uint8"` | 0–1 (N19) |

`normalization_exact`:

- Histogram-based methods (A–D, luminance), at every scope including `patch`: `true`
  exactly when every pooled histogram has `max_representative_error == 0` (§1.2, as
  CellPatcher writes it today). Luminance is therefore always `false` (float histograms).
- Colour methods: `true` when every parameter was computed from all tissue pixels without
  sampling: Reinhard at any scope (exact Lab summaries) and every colour method at the
  `patch` scope. Sample-based fits (Macenko, Vahadane, Ruifrok amounts at group and image
  scopes) are `false`.

---

## 4. Scopes, groups and references

### 4.1 Source scopes

| `scope` | Pooled pixels | Group key | `normalization_group` |
|---|---|---|---|
| `experiment` | All planes | `()` | `experiment` |
| `condition` | Same condition | `(condition,)` | `<condition>` |
| `biological_repeat` | Same condition and biological repeat | `(condition, biological_repeat)` | `<condition>/<biological_repeat>` |
| `technical_repeat` | Same condition, biological and technical repeat | all three labels | `<condition>/<biological_repeat>/<technical_repeat>` |
| `image` (default) | All Z and T planes of one image | three labels + `source_relative_path` | `<source_relative_path>` |
| `patch` | The patch's own pixels | calling brick's patch identifier | `<source_relative_path>::<patch id>` |

- Groups are identified by their full label path (`bio_rep1` under two conditions is two
  groups), as `PlaneRecord.group_key` does today; Milestone 2 adds the `experiment` level.
- Every pooled plane of the group is used, including planes the calling brick does not
  normalize (for example Z slices without cells).
- There is no batch scope and no per-stain scope (N12).

### 4.2 Channels and time

- Parameters are per channel, except for family B (channels in `shared_channels`), family E
  (R, G and B fitted jointly) and luminance (one converted channel).
- Time points are pooled by default. `per_time_point=true` gives each time point its own
  group; `::t=<t>` is appended to `normalization_group`.

### 4.3 Parameter source

| `parameter_source` | Meaning | Allowed for |
|---|---|---|
| `scope` (default) | One parameter set per group of `scope` | Every method |
| `reference` | One parameter set fitted on the reference, applied to every image (for example the controls), so differences between conditions are kept | Families A, B and C, and the amount normalization of Ruifrok (§3.7.3), with a reference group; luminance with a reference image, plane or file (§3.6, N28) |

With `reference`, `normalization_scope` is `reference`, and `normalization_group` and
`normalization_reference` are the reference identifier (§4.4).

### 4.4 References and targets

One target per run. References are chosen in the preprocessing brick and recorded in its
manifest (N13, follow-up P3); without the preprocessing brick only a reference file is
possible (N9).

| Kind | Identifier | Chosen by | Statistics |
|---|---|---|---|
| Reference image | `image:<source_relative_path>` | Preprocessing manifest | That image's planes in the statistics files |
| Reference plane | `plane:<plane_id>`, with the `plane_id` format of the statistics file the method reads (§8.2 or §8.3) | Preprocessing manifest | That plane's row |
| Reference group | `group:<label path>`, e.g. `group:control` or `group:control/bio_rep1` | Preprocessing manifest | Every plane whose labels start with the label path, pooled |
| Published standard | `standard:he`, `standard:hdab` | Built in (N14) | §9.3 |
| Reference file | `file:<file name>` (with its SHA-256 at run level, never its path) | Calling brick's configuration, when there is no preprocessing output | The file's pixels; only with the image and patch scopes (N9), or as luminance's parameter source (§3.6) |

| Method | Allowed targets | Default |
|---|---|---|
| A, B, C, luminance | Fixed scale (no target); as parameter source (§4.3) a reference group for A–C, a reference image, plane or file for luminance | Fixed scale |
| D | Reference image, plane, group or file | None: a reference must be chosen |
| Ruifrok | Its fixed stain matrix (no reference); for its amounts a reference group as parameter source (§4.3) | The stain set's matrix |
| Reinhard | Reference image, plane, group or file | None: a reference must be chosen, since there is no published Reinhard standard (N14 makes the published standard the default only where one exists) |
| Macenko, Vahadane | Reference image, plane, group, file, or published standard | `standard:he` (`standard:hdab` with `stain_set="hdab"`) (N14) |

A reference that matches no planes in the statistics files is an error (§11.1).

### 4.5 Availability

| Statistics given to the library | Scopes |
|---|---|
| Statistics files of the preprocessing brick (§8) | All six |
| None (pixels only) | `image` (all Z and T planes of the image's own pixels) and `patch` (N9) |

Without files, image-scope parameters equal those computed from the files whenever the
histograms are exact, and colour statistics computed from pixels follow §8.3–8.5 exactly.

---

## 5. Modality profiles

The modality is declared by the user (N15): in the preprocessing brick, or in the calling
brick when preprocessing is not used. Metadata may suggest one (an RGB samples axis suggests
`brightfield_rgb` or `histology_rgb`); it never decides.

**Rule:** a profile offers exactly its default and its "also allowed" choices (N16, N25). Where a
family is named, all its methods are offered; where methods are named, only those. Family C
is OD followed by a linear method, so it offers only the linear methods the profile offers
(for `brightfield_gray`, `percentile`). Every other choice is refused with a message of §7.

| `modality` | Channels | Default | Also allowed | Not offered (reason in §7) |
|---|---|---|---|---|
| `fluorescence` | any | `percentile` 1–99.8, per channel (A), `image` scope | `minmax`, `zscore`, `robust_zscore` (A); shared parameters (B, all four methods) only for channels the user marks as ratiometric | Histogram matching (D), colour methods (E), OD (C), luminance |
| `brightfield_gray` | 1 | `percentile` (A), `image` scope | OD + `percentile` (C); histogram matching (D) | Other linear methods, colour methods (E), luminance |
| `brightfield_rgb` | 3 | Shared `percentile` 1–99.8 (B, `shared_channels="all"`), `image` scope | Other B methods; OD with shared I₀ followed by a B method (C); luminance + an A method | Per-channel linear (A), histogram matching (D), colour methods (E) |
| `phase_contrast` (incl. DIC) | any | `robust_zscore` (A), `image` scope | `percentile` (A); histogram matching (D) | `minmax`, `zscore`, OD (C), colour methods (E), luminance |
| `histology_rgb` (H&E, IHC) | 3 | `macenko` (E) to `standard:he`, `image` scope | `reinhard`, `vahadane`, `ruifrok` (E) | Linear methods (A, B), OD (C), histogram matching (D), luminance |
| `electron_microscopy` (SEM, TEM) | any | `percentile` (A), `image` scope | Histogram matching (D); `robust_zscore` (A) | `minmax`, `zscore`, OD (C), colour methods (E), luminance |

- Every profile allows every scope of §4.1, subject to the purpose (§6) and to §4.5.
- `phase_contrast` and `electron_microscopy` state that their intensities are not
  quantitative (§6).
- `histology_rgb` needs colour statistics (§8.3) at group and image scopes, and a tissue
  mask (§9.2); its statistics file is `colour_statistics.csv`, not `plane_histograms.csv`
  (follow-up P2).
- A wrong channel count for the modality is an error (§11.1).

---

## 6. Purpose

`purpose` is chosen together with the method (N16).

- **`model_input`** (default): every method and scope of the modality profile.
- **`quantitative`**: only choices that keep intensity differences between the groups being
  compared:
  - scope `experiment`, or `parameter_source="reference"` with a reference group;
  - families A, B and C, and Ruifrok with a fixed stain matrix, as far as the profile
    allows them; luminance only with a reference image, plane or file as the single
    parameter source (§3.6, N28);
  - refused: the `condition`, `biological_repeat`, `technical_repeat`, `image` and `patch`
    scopes; histogram matching; Reinhard, Macenko, Vahadane.
  - For `phase_contrast` and `electron_microscopy`, only the profile's linear methods at the
    `experiment` scope (no reference source).

| `modality` | `quantitative` offers | Default |
|---|---|---|
| `fluorescence` | A; B for ratiometric channels; experiment or reference group | `percentile` 1–99.8, `experiment` |
| `brightfield_gray` | `percentile`; OD + `percentile`; experiment or reference group | `percentile` 1–99.8, `experiment` |
| `brightfield_rgb` | B; OD + B; experiment or reference group. Luminance + A with a reference image, plane or file as source | Shared `percentile` 1–99.8, `experiment` |
| `phase_contrast` | `robust_zscore`, `percentile`; experiment only | `robust_zscore`, `experiment` |
| `histology_rgb` | `ruifrok`; experiment or reference group | `ruifrok`, `stain_set="he"`, `amounts` with `percentile` 1–99.8, `experiment` |
| `electron_microscopy` | `percentile`, `robust_zscore`; experiment only | `percentile` 1–99.8, `experiment` |

---

## 7. Refusal messages

`check_settings` and `fit` refuse a choice outside the profile or the purpose with
`MethodNotAllowedError`, whose message names an allowed alternative. Placeholders are in
braces:

| Placeholder | Text |
|---|---|
| `{modality}` | The modality in words: `fluorescence`, `grayscale brightfield`, `RGB brightfield`, `phase-contrast`, `histology`, `electron microscopy` |
| `{default}`, `{allowed}` | The profile's default for the purpose, and its other choices |
| `{Choice}`, `{Method}` | The refused method or modifier by its written name (§3.1), capitalized |
| `{scope}`, `{unit}` | The scope, and what it gives its own parameters: `condition`, `biological repeat`, `technical repeat`, `image`, `patch` |
| `{file}` | `plane_histograms.csv` or `colour_statistics.csv` (§8.6) |
| `{kinds}` | The allowed reference kinds of §4.4, e.g. "reference image or reference group" |
| `{standard}` | `H&E` or `H-DAB` |
| `{expected}`, `{n}` | "RGB images with exactly three channels" or "one channel"; the channel count found |

| Code | Situation | Message |
|---|---|---|
| `method_not_in_profile` | Method or modifier outside the profile (§5) | "{Choice} is not available for {modality} images: {reason}. Use {default} instead, or one of: {allowed}." |
| `shared_needs_ratiometric` | `fluorescence` with `shared_channels` but no ratiometric channels | "Shared parameters are not available for fluorescence channels unless they are ratiometric: each channel is an independent stain and gets its own parameters. Mark the ratiometric channels, or use {default}." |
| `scope_not_quantitative` | `quantitative` with a refused scope | "The {scope} scope cannot be used with purpose 'quantitative': it gives each {unit} its own parameters, which removes the intensity differences between them. Use the experiment scope, or a reference group as the single parameter source." |
| `method_not_quantitative` | `quantitative` with histogram matching | "Histogram matching cannot be used with purpose 'quantitative': it replaces the intensities of every image with those of the reference. Use {default} instead, or one of: {allowed}." |
| `stain_fit_not_quantitative` | `quantitative` with Reinhard, Macenko or Vahadane | "{Method} cannot be used with purpose 'quantitative': it fits the stain colours of each image or group, so stain amounts are no longer comparable between groups. Use Ruifrok colour deconvolution with a fixed stain matrix." |
| `modality_not_quantitative` | `quantitative` with `phase_contrast` or `electron_microscopy` and anything but an experiment-scope linear method | "Intensities in {modality} images are not quantitative, so purpose 'quantitative' only offers {allowed} at the experiment scope. Choose one of these, or use purpose 'model_input'." |
| `scope_needs_statistics` | Group scope without statistics files (N9) | "The {scope} scope needs the statistics files of the preprocessing brick ({file}), which were not given. Run the preprocessing brick on this experiment, or use the image or patch scope." |
| `reference_needed` | D, Reinhard, or `parameter_source="reference"` without a reference | "{Method} needs a {kinds} as its reference, but none was chosen. Choose one in the preprocessing brick, or give a reference image file." |
| `target_not_allowed` | A published standard for D or Reinhard | "{Method} cannot use the published {standard} standard: published standards exist only for stain vectors (Macenko and Vahadane). Choose a reference image or a reference group." |
| `reference_source_not_linear` | `parameter_source="reference"` with D, Reinhard, Macenko, Vahadane or Ruifrok `rgb` output | "A reference can be the single parameter source only for linear methods and Ruifrok stain amounts. {Method} maps every image onto its reference instead: choose the reference as its target." |
| `luminance_needs_pixels` | Luminance at a group scope, or with a reference group as source (§3.6) | "Luminance conversion is available only at the image and patch scopes, or with a reference image as the single parameter source, because the statistics files hold no histograms of converted images. Choose one of these, or use shared parameters without luminance conversion." |
| `plane_stack_not_allowed` | A colour method on a patch spanning several Z or T planes | "{Method} normalizes 2D RGB planes, and its background light and stain fit are per plane, so each patch must come from a single Z and T plane. Split the image into planes before normalizing." |
| `wrong_channel_count` | Colour methods, luminance, `brightfield_rgb` or `histology_rgb` without three channels; `brightfield_gray` with more than one | "{Choice} needs {expected}, but these images have {n} channels. Check the axis mapping (for RGB images the samples axis must be the channel axis), or choose another modality." |

Reasons for `method_not_in_profile`:

| Modality | Refused | `{reason}` |
|---|---|---|
| `fluorescence` | Histogram matching | "each channel is an independent stain, and matching intensity distributions would remove real differences in staining" |
| `fluorescence`, `brightfield_gray`, `phase_contrast`, `electron_microscopy` | Colour methods | "colour and stain methods need RGB brightfield or histology images" |
| any but `brightfield_rgb` | Luminance | "luminance conversion is only for RGB brightfield images" |
| `brightfield_rgb` | Per-channel linear, histogram matching | "scaling or matching each colour channel separately would change the colours" |
| `phase_contrast` | `minmax` | "the bright and dark halos around cells dominate the minimum and maximum" |
| `histology_rgb` | Linear methods, OD, histogram matching | "scaling or matching each colour channel separately would change the stain colours; histology images are normalized with a colour method" |
| any | Anything else | "it is not among the methods offered for {modality} images" |

---

## 8. Statistics files

The library defines and computes the statistics and owns the readers and writers (N2); the
preprocessing brick runs the computation over the dataset and writes the files next to its
`run_manifest.json`. Which files are written depends on the modality (N7, follow-up P2).

### 8.1 Shared columns

Every statistics file starts with these columns, in this order (as `plane_histograms.csv`
does today): `position_id`, `image_file`, `z`, `c`, `t`, `condition`, `biological_repeat`,
`technical_repeat`, `source_relative_path`, `plane_id`. Their meaning is that of the
preprocessing contract §5. Labels are always read as text.

Common format: UTF-8, comma-separated, one header row, rows in a deterministic order, floats
written at full precision (Python `repr`), blanks for missing values, booleans `true` or
`false`, written to a temporary file and replaced only once writing succeeded.

### 8.2 `plane_histograms.csv` (existing, schema version 2)

Unchanged (plan §7.1): one row per 2D plane and channel, defined by the preprocessing
contract §4–§5 and the moved `csv_io.py` (§1.1). Used by families A–D and luminance (from
pixels, §3.6).

> Version 2 is the version the preprocessing brick writes since 21 October 2026 (`z`, `c`,
> `t` always written), and the one CellPatcher already pins; "unchanged" means version 2,
> as the code moves in Milestone 1.

### 8.3 `colour_statistics.csv` (new, schema version 1)

For `histology_rgb`. One row per 2D RGB plane (all three channels together), sorted by
`source_relative_path`, then `z`, then `t`.

| Column | Type | Meaning |
|---|---|---|
| Shared columns (§8.1) | | `c` is blank. `plane_id` is `<source_relative_path>::z=<z>,t=<t>` (no `c`, since the row covers all three channels), built by a new `make_colour_plane_id(source_relative_path, *, z, t)` next to `make_plane_id`. |
| `dtype` | text | NumPy dtype name of the source plane |
| `bit_depth` | integer | Storage bits |
| `total_pixel_count` | integer | All pixels of the plane |
| `tissue_pixel_count` | integer | Pixels passing the tissue mask (§3.7.1) |
| `od_threshold` | number | Tissue-mask OD threshold used (N16a) |
| `exclude_saturated`, `exclude_dark` | boolean | Tissue-mask exclusions used |
| `dark_limit` | number or blank | Near-black limit used (fraction of `I₀`, §3.7.1); blank when `exclude_dark` is `false` |
| `i0_r`, `i0_g`, `i0_b` | number | Background light per channel (§9.1) |
| `i0_source` | text | `estimated`, `fixed` or `user` (N15a) |
| `lab_count` | integer | Tissue pixels in the Lab sums (equals `tissue_pixel_count`) |
| `lab_sum_l`, `lab_sum_a`, `lab_sum_b` | number | Σ L, Σ a, Σ b over tissue pixels |
| `lab_sumsq_l`, `lab_sumsq_a`, `lab_sumsq_b` | number | Σ L², Σ a², Σ b² over tissue pixels |
| `od_count` | integer | Tissue pixels in the OD sums (equals `tissue_pixel_count`) |
| `od_sum_r`, `od_sum_g`, `od_sum_b` | number | Σ OD per channel over tissue pixels |
| `od_cross_rr`, `od_cross_rg`, `od_cross_rb`, `od_cross_gg`, `od_cross_gb`, `od_cross_bb` | number | Σ ODᵢ·ODⱼ over tissue pixels |
| `sample_count` | integer | Pixels of this plane stored in `colour_samples.npz` |
| `sample_seed` | integer | Plane seed `s` of this plane's sample, derived from the run seed (§8.5) |
| `schema_version` | text | `1` |

- Sums are computed in float64 and pool across any scope by adding them (N24); Reinhard
  and Macenko statistics of a group are exact.
- A plane read in blocks gives exactly the same row as the whole plane (block sums added
  in a fixed order; samples by §8.5).
- Reading checks: header and schema version; `lab_count = od_count = tissue_pixel_count ≤
  total_pixel_count`; `sample_count ≤ total_pixel_count`; `i0_source` one of the three
  values.

### 8.4 `colour_samples.npz` (new, schema version 1)

Next to `colour_statistics.csv`, written by `numpy.savez` (uncompressed) and read with
`allow_pickle=False`. Raw RGB values rather than OD, so `I₀` and the tissue mask can change
later without rereading images (follow-up P4).

| Array | Shape and type | Meaning |
|---|---|---|
| `schema_version` | 0-d `<U1` | `"1"` |
| `seed` | 0-d uint64 | Run seed (§9.4) |
| `sample_size` | 0-d int64 | Requested pixels per plane |
| `plane_id` | (P,) unicode | `plane_id` of each plane, in CSV row order |
| `offset` | (P + 1,) int64 | Samples of plane `p` are rows `offset[p]` to `offset[p + 1] − 1`; `offset[0] = 0` |
| `pixel_index` | (N,) int64 | Row-major index `y · width + x` of each sampled pixel in its plane, increasing within a plane |
| `rgb` | (N, 3) source dtype | R, G, B values of each sampled pixel |

Reading checks: the `plane_id` array equals the CSV's `plane_id` column in order, and
`offset[p + 1] − offset[p]` equals each row's `sample_count`. The arrays are identical
between runs with the same inputs and settings; the file bytes may differ (zip timestamps).

### 8.5 Pixel sampling

Reservoir sampling by smallest keys, so the result does not depend on whether, or in which
blocks, a plane is read (all arithmetic on uint64, modulo 2⁶⁴):

```
plane seed  s = first 8 bytes (big-endian) of SHA-256 of the UTF-8 text
              "<run seed in decimal>:<plane_id>", as uint64
key(i)      = splitmix64 output i:  z = s + (i + 1) · 0x9E3779B97F4A7C15  (mod 2⁶⁴)
              z = (z ⊕ (z ≫ 30)) · 0xBF58476D1CE4E5B9
              z = (z ⊕ (z ≫ 27)) · 0x94D049BB133111EB
              key = z ⊕ (z ≫ 31)
sample      = the sample_size pixels with the smallest keys (ties by smaller i),
              stored in increasing i; every pixel when the plane has fewer
```

`i` is the pixel's row-major index. Every pixel is eligible, including background and
non-finite ones; the tissue mask is applied when the samples are used. A plane's sample
depends only on its own pixels, the run seed and its `plane_id`, so adding images never
changes other planes' samples.

### 8.6 Files per modality

| Modality | Files |
|---|---|
| `histology_rgb` | `colour_statistics.csv`, `colour_samples.npz` |
| every other | `plane_histograms.csv` |

The library reads the files it is given and checks them (§8.2–8.4); fingerprints
(SHA-256) of the files used are recorded at run level by the calling brick (§10.4).

---

## 9. Defaults

### 9.1 I₀ (N15a)

| Mode | Value |
|---|---|
| `estimated` (default) | Per channel: the 99th percentile (§1.2 definition) of the pooled histogram of the pixels it describes (pooled over the shared channels for a shared I₀), from histograms computed with `compute_plane_histogram` or `compute_block_histogram` at the default bin count, so exact for 8- and 16-bit images. Family C: all Z and T planes of the image (per time point with `per_time_point=true`), from the image's rows of `plane_histograms.csv` or its pixels (N26). Colour methods: one 2D RGB plane. **Provisional** on images with little background (plan §13). |
| `fixed` | The type's maximum for integer images (255 for 8-bit, 65,535 for 16-bit); 1.0 for float images |
| `user` | One value per channel, finite and positive |

For colour methods `I₀` is recorded per plane in `colour_statistics.csv` (histology images
are 2D, so a plane is the image). A patch always uses the `I₀` of its image (family C) or
plane (colour methods), never one estimated from the patch, which may hold no background:
from the statistics files, or from the image's pixels when there are none.

### 9.2 Tissue mask (N16a)

| Setting | Default |
|---|---|
| `od_threshold` | 0.15 |
| `exclude_saturated` | `true` |
| `exclude_dark` | `true` |
| `dark_limit` | 0.05 (fraction of `I₀`, **Provisional**) |

All configurable; the values used are recorded in every `colour_statistics.csv` row.

### 9.3 Published standards (N14)

| Standard | Stain vectors (R, G, B) | Maximum amounts | `I₀` |
|---|---|---|---|
| `standard:he` | hematoxylin (0.5626, 0.7201, 0.4062), eosin (0.2159, 0.8012, 0.5581) | 0.8558, 0.4477 (1.9705 and 1.0308 ÷ ln 10) | 240 |
| `standard:hdab` | hematoxylin (0.650, 0.704, 0.286), dab (0.268, 0.570, 0.776), unit length | **Provisional**, set by the study of plan §13 | 255 |

The H&E values come from the reference implementation of Macenko et al. (2009). That code
uses natural-log OD, so its maximum amounts (1.9705, 1.0308) are divided by ln 10 for the
log₁₀ OD of §3.4; the unit stain vectors do not change. All values are checked against
their sources before Milestone 4 is released. The 0–1 output of §3.8 is relative to the
target's `I₀`, so `I₀` does not change it.

### 9.4 Samples, seeds and colour-method parameters

| Setting | Default | Status |
|---|---|---|
| `sample_size` (pixels per plane) | 10,000 | **Provisional** (study at 5,000–50,000, plan §13) |
| Run seed of the samples | 0, recorded in `colour_samples.npz` and the manifest | Fixed |
| `seed` (Vahadane) | 0, recorded at run level | Fixed |
| `angle_percentile` (Macenko) | 1 | **Provisional** |
| `max_percentile` (Macenko, Vahadane) | 99 | **Provisional** |
| `sparsity` (Vahadane) | 0.1 | **Provisional** |
| `min_tissue_pixels` | 1,000 (N18) | Fixed |
| Stain-label warning | cosine similarity below 0.9 | **Provisional** |
| `output_dtype` | `float32` | Fixed |

A provisional default may change once measured. Results stay reproducible because every
value that affects the statistics or the parameters is recorded at run level (§10.4) or in
the statistics files; changing a provisional default changes no file schema. (The
stain-label limit only decides a warning.)

---

## 10. Parameter columns

Parameters are written as numeric columns, never as JSON (N21). Only the columns of the
method used are written (N23); targets are written once in the calling brick's run
configuration (N21).

### 10.1 Common columns (every method)

| Column | Type | Meaning |
|---|---|---|
| `normalization_method` | text | Written name (§3.1) |
| `normalization_scope` | text | §4.1 scope, or `reference` (§4.3) |
| `normalization_group` | text | §4.1 group text, or the reference identifier (§4.3) |
| `normalization_exact` | `true`/`false` | §3.8 |
| `normalization_reference` | text | Only when the method uses a reference (N27): histogram matching, Reinhard, Macenko, Vahadane (including `standard:he`/`standard:hdab`), and `parameter_source="reference"`. The reference identifier of §4.4, which says where the reference is: `image:<source_relative_path>` and `plane:<plane_id>` relative to the experiment root, `group:<label path>`, or `file:<file name>`. Never an absolute path. The reference's fitted values stay in the run configuration (§10.4), so normalizing on load never needs the reference itself. |

### 10.2 Per-method columns

| Method | Per-row columns (source) | Run level (target and fixed settings) |
|---|---|---|
| A, B, luminance | `normalization_offset`, `normalization_scale` | Settings |
| C | `normalization_i0`, `normalization_offset`, `normalization_scale` | Settings |
| D | None beyond the common columns: the mapping is rebuilt from the histograms of `normalization_group` and `normalization_reference` (N22) | Settings; reference definition |
| Ruifrok | `normalization_i0_r`, `normalization_i0_g`, `normalization_i0_b`; with `amounts` also `normalization_offset`, `normalization_scale` (one value per named stain) | Stain names and stain matrix (3 × 3) |
| Reinhard | `normalization_lab_mean_l`, `normalization_lab_mean_a`, `normalization_lab_mean_b`, `normalization_lab_std_l`, `normalization_lab_std_a`, `normalization_lab_std_b` | Target means and standard deviations |
| Macenko, Vahadane | For each stain `<name>`: `normalization_stain_<name>_r`, `normalization_stain_<name>_g`, `normalization_stain_<name>_b`; then for each stain `normalization_max_<name>`; then `normalization_i0_r`, `normalization_i0_g`, `normalization_i0_b` | Target stain names, vectors, maximum amounts and `I₀`; `angle_percentile`, `max_percentile`; Vahadane's `sparsity` and `seed` |

Stain names come from the stain set (§3.7.2): `hematoxylin`, `eosin`, `dab`, or the user's.

### 10.3 Formatting

- Numbers at full precision (Python `repr` of the float), read back unchanged.
- Lists (`normalization_offset`, `normalization_scale`, `normalization_i0`): one value per
  channel of the patch (per named stain for Ruifrok amounts), in channel order, separated
  by `;`; a single value for one channel. Shared channels repeat the shared value.
- The existing linear columns keep their names and meaning (CellPatcher follow-up C5).
- Column order within a file is the calling brick's; readers select columns by name.

### 10.4 Run level

The calling brick stores, once per run, in its own run configuration:

1. **Settings** — `NormalizationSettings.to_dict()`:

   | Key | Default |
   |---|---|
   | `library_version` | Version of `morphobricks-normalization` |
   | `modality` | Required |
   | `purpose` | `model_input` |
   | `method` | Profile default (§5, §6) |
   | `percentile` | `[1.0, 99.8]` |
   | `consistency` | 1.482602218505602 |
   | `shared_channels` | `null` (`"all"` for `brightfield_rgb`) |
   | `optical_density`, `luminance` | `false` |
   | `scope` | Profile default |
   | `per_time_point` | `false` |
   | `parameter_source` | `scope` |
   | `reference` | Identifier (§4.4), or `null` |
   | `i0` | `{"mode": "estimated", "values": null}` |
   | `tissue_mask` | `{"od_threshold": 0.15, "exclude_saturated": true, "exclude_dark": true, "dark_limit": 0.05}` |
   | `stain_set` | `"he"`, `"hdab"`, or `{"names": [...], "vectors": [[r, g, b], ...]}` |
   | `ruifrok_output`, `ruifrok_stains`, `amount_method` | `"amounts"`, all named stains, `{"method": "percentile", "percentile": [1.0, 99.8]}` |
   | `angle_percentile`, `max_percentile`, `sparsity`, `seed`, `min_tissue_pixels` | §9.4 |
   | `output_dtype` | `"float32"` |

   Keys that do not apply to the method are omitted.

2. **Target** — `Target.to_dict()`, or `null` for methods without one:

   | Method | Target keys |
   |---|---|
   | D | `reference` |
   | Ruifrok | `stains` (three names, `residual` last when completed), `stain_matrix` (3 × 3, rows) |
   | Reinhard | `reference`, `lab_mean` `[l, a, b]`, `lab_std` `[l, a, b]` |
   | Macenko, Vahadane | `reference`, `stains`, `stain_vectors` (rows), `max_amounts`, `i0` `[r, g, b]` |

   For a reference file, `reference` is `{"id": "file:<name>", "sha256": "..."}`; paths are
   never recorded.

3. **Statistics inputs** — for each statistics file used: its name, `schema_version` and
   SHA-256.

For CellPatcher these go into `run_config.json` (follow-up C5); the key names under which
they are stored there are CellPatcher's. Raw patches are normalized on load from the row's
columns plus these three objects; histogram matching rebuilds its mapping from the recorded
group and reference (N22).

---

## 11. Validation

Every message states in plain language what happened, names the affected images, groups or
patches, and suggests an action, in the style of the other MorphoBricks bricks. Technical
details go to the calling brick's log.

### 11.1 Errors

Raised as exceptions before any pixel is normalized; the calling brick blocks the run.

| Situation | Exception |
|---|---|
| Refusals of §7 | `MethodNotAllowedError` |
| Unsupported pixel type; bin count not allowed (§1.2) | `UnsupportedDtypeError`, `ValueError` (existing) |
| Group without finite pixels | `NoValidPixelsError` (existing message) |
| Zero or non-finite scale for a group (existing message); zero `σ` (Reinhard) or maximum amount (Macenko, Vahadane) for a group | `ValueError` |
| Percentiles outside `0 ≤ low < high ≤ 100` (existing message); other invalid settings (non-positive `od_threshold`, `sample_size` below 1, invalid stain names or vectors, singular stain matrix, `shared_channels` naming missing channels) | `ValueError` |
| Statistics file with unexpected columns, schema version or inconsistent counts (§8.2–8.4); samples not matching the CSV rows | `CsvFormatError` (existing) and a new `ColourStatisticsFormatError` (a `ValueError`) |
| Colour method at a group or image scope without colour statistics; tissue-mask or `I₀` settings differing from those of the statistics files (§3.7.1) | `ValueError` |
| Reference matching no planes in the statistics files | `ValueError` |
| `I₀` not finite or not positive; user `I₀` with the wrong number of values | `ValueError` |
| Float RGB outside 0–1 for a colour method | `ValueError` |
| A colour method on a patch spanning several Z or T planes (`plane_stack_not_allowed`, §7) | `MethodNotAllowedError` |
| Group with fewer tissue pixels than `min_tissue_pixels` for Macenko or Vahadane at a group or image scope (§3.7.7) | `ValueError` |
| `from_columns` with a missing or unreadable column, a list of the wrong length, or (histogram matching) without the statistics needed to rebuild the mapping | `ValueError` |

### 11.2 Warnings

Returned as `Issue` objects; processing continues and the calling brick reports them.

| Code | Situation |
|---|---|
| `group_scope_removes_differences` | Scope `condition` (N11), `biological_repeat` or `technical_repeat`: intensity differences between those groups are removed |
| `approximate_parameters` | Parameters from inexact histograms or samples (`normalization_exact = false`) |
| `planes_without_finite_pixels` | Planes without finite pixels excluded from a group that still has others |
| `patches_discarded` | Patches without spread, finite pixels or enough tissue (§3.7.7, N18), listed in one warning with a suggested method, region or scope |
| `few_tissue_samples` | A group with enough tissue pixels but fewer than `min_tissue_pixels` tissue samples for Macenko or Vahadane; suggests a larger `sample_size` in the preprocessing brick |
| `stain_label_uncertain` | A fitted stain vector's cosine similarity to its label's reference below the limit of §9.4; suggests checking the stain set |

---

## 12. Open points

These are not settled by plan §12 and need the owner's decision before the milestone that
needs them:

1. **Luminance statistics at group scopes (deferred, N28).** Group scopes need histograms of
   the converted planes, which follow-up P2 does not write. Until they are added, luminance
   is computed from pixels (§3.6). A likely future form: the preprocessing brick also writes
   the luminance planes' histograms in the `plane_histograms.csv` format, in a second file,
   when the modality is `brightfield_rgb`.

Values left to measurement (plan §13) are marked **Provisional** above: the sample size, the
colour-method parameters, `I₀` estimation on tissue-dense images, the near-black limit, the
stain-label warning limit and the H-DAB maximum amounts.
