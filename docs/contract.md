# Normalization Brick Contract

> See also: [Plan](plan.md) · [Preprocessing follow-up](followup_preprocessing.md) ·
> [CellPatcher follow-up](followup_cellpatcher.md)

This document fixes the rules other bricks can rely on: the methods, the scopes and repeat
groups, the image types, the statistics file, the intensity checks, the parameter columns and
the validation severities. It implements the decisions of the [plan](plan.md) (§12) as
simplified on 26 October 2026 (N29–N40); it does not change any of them. The earlier contract
with modality profiles, purpose rules, colour statistics and references is withdrawn.

- Changing a file format (§6) requires a new `schema_version` of that file.
- Changing any other rule requires the project owner's approval and an entry in plan §12.
- Values marked **Provisional** are defaults left to measurement by plan §13; they are
  recorded with every result and may change once measured.

| Section | Implemented in |
|---|---|
| 1. Existing code | Milestone 1 (moved unchanged; done) |
| 2. Interface; 3.1–3.2 fluorescence methods; 4. Scopes and repeat groups; 8. Columns | Milestone 2 (done; `image_summaries` and `intensity_checks` of §2.2 follow in Milestone 4) |
| 3.3 RGB method; 5. Image types | Milestone 3 |
| 7. Intensity checks | Milestone 4 |
| 9. Validation | With the milestone of each check |

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

These rules were defined in the
[preprocessing contract](https://github.com/CellMigrationLab/morphobricks-preprocessing/blob/main/docs/contract.md)
(§3, §4, §6 and §7) until the preprocessing follow-up P1 moved them here unchanged. The
moved code (§1.1) implements them.

#### 1.2.1 Pixel types and bin counts

| Kind | Types | Default bins | Allowed bins |
|---|---|---|---|
| Integer | int8/16/32/64, uint8/16/32/64 | min(65,536, 2^bits) | 1 to min(65,536, 2^bits) |
| Float | float32, float64 | 10,000 | 2 to 65,536 |

Everything else (bool, complex, float16, ...) is rejected (`UnsupportedDtypeError`).
The preprocessing brick adds two rules for a whole experiment: one dtype per experiment,
and the same bin count for all its planes (preprocessing contract §3).

#### 1.2.2 Histogram semantics

Notation in this section: `n` = `bin_count`, `lo` = `binning_min`, `hi` = `binning_max`
(columns of §6.2).

##### Integer planes

`lo` and `hi` are the plane's exact minimum and maximum. `W = hi − lo + 1` values.

- If `W ≤ n`: bin `i` holds exactly the value `lo + i` for `i < W`; bins `W .. n−1` are
  empty. The histogram is exact.
- If `W > n`: bin `i` holds the values `lo + ceil(i·W/n)` to `lo + ceil((i+1)·W/n) − 1`.
  Limits are whole numbers, consecutive bins touch without gaps or overlaps, and widths
  are `floor(W/n)` or `ceil(W/n)` values.
- Representative value of a bin: `(first value + last value) / 2`.
- Bound: every pixel is within `(widest bin − 1) / 2` of its representative.

##### Float planes

`lo` and `hi` are the plane's finite minimum and maximum; NaN and ±infinity are ignored.

- Bin width `(hi − lo) / n`. Bin `i` is `[lo + i·width, lo + (i+1)·width)`; the last bin
  also includes `hi`. A pixel `x` goes to bin `floor((x − lo) / width)`, clipped to
  `n − 1` (edge assignment is subject to floating-point rounding).
- Constant plane (`lo == hi`): all pixels in bin 0; representative `lo`.
- No finite pixels: `lo`, `hi` blank; all counts 0.
- Representative value: the bin midpoint `lo + (i + 0.5)·width`.
- Bound: every pixel is within `width / 2` of its representative.

##### Planes read in blocks

A plane too large for the memory limit is read in blocks (rectangles of its stored strips,
tiles or chunks). Its extrema, pixel counts and bin counts are exactly those of the whole
plane, so a row never depends on whether, or in which blocks, its plane was read.

#### 1.2.3 Group statistics

Groups pool their planes' (representative value, count) pairs; per-plane statistics are
never averaged.

| Statistic | Definition | Accuracy |
|---|---|---|
| min, max | Smallest `pixel_min`, largest `pixel_max` | Exact |
| mean, std | Count-weighted mean and population std (÷N) of representatives | Exact for exact histograms; otherwise very close (errors cancel) |
| percentile q | Smallest representative with cumulative count ≥ ceil(q/100 · N) (numpy `inverted_cdf`) | Within the group's largest per-plane bound |
| median | 50th percentile | As percentiles |
| MAD | Median of \|representative − median\| | Within twice the bound |

Planes without finite pixels are excluded and reported; a group with no finite pixels is
an error.

#### 1.2.4 Linear parameters

Every method is an offset and a scale; applying it computes `(pixels − offset) / scale`.

| Method | Offset | Scale |
|---|---|---|
| min-max | min | max − min |
| percentile range (default 1–99.8) | low percentile | high − low percentile |
| z-score | mean | std |
| robust z-score | median | MAD × 1.4826 |

A zero or non-finite scale is an error with a plain-language message.

`normalization_exact` (§8) is `true` for histogram-based parameters exactly when every
pooled histogram has `max_representative_error == 0`.

---

## 2. Interface

### 2.1 Objects

| Object | Holds | Stored as |
|---|---|---|
| `NormalizationSettings` | Image type, method and its options, scope, pooling weight, per-time-point option (§8.3) | Run configuration of the calling brick, once per run |
| Source statistics | Pooled plane histograms of a group (§1.2), or pixel arrays from which the library computes the same statistics | Not stored by the library |
| `RepeatGroups` | Mapping from `(condition, biological_repeat)` to the repeat group's label (§4.2), built by the calling brick from `experiment_design.csv` | Not stored by the library |
| `Parameters` | Offset and scale per channel of one group (§8) | Per-row columns (N21, N23) |
| `Issue` | Severity, code, message, affected items, suggested action | Reported by the calling brick |

### 2.2 Functions

```python
issues = check_settings(settings, *, statistics_file: bool)
key = group_key(record, settings, repeat_groups=None)     # group of one plane_histograms row
parameters = fit(statistics, settings)                   # one group
normalized = parameters.apply(pixels, channel_axis=...)
columns = parameters.to_columns()                        # dict[str, str]
parameters = Parameters.from_columns(columns, settings)
settings.to_dict(); NormalizationSettings.from_dict(...)
summaries = image_summaries(records)                     # §7
issues = intensity_checks(summaries, check_settings=None)
```

| Function | Rule |
|---|---|
| `check_settings` | Checks the method against the image type (§5) and the scope against the available statistics (§4.5) using the settings alone, so a calling brick can refuse a choice before reading any image. Raises `MethodNotAllowedError` (a `ValueError`) with a message of §9.1. |
| `group_key` | The group of a row at the settings' scope (§4.1). `biological_repeat` needs `repeat_groups`; without it, repeats match by name (§4.2). |
| `fit` | One entry point for every method; dispatches on `settings.image_type` and `settings.method`. Statistics are pooled histograms per channel, or pixels; with `pooling="conditions"` the histograms carry their condition (§4.3). Never reads files and never walks folders. |
| `apply` | Returns a new float32 array; the input is never modified. `channel_axis` gives the channel axis of `pixels`; it may be left out for one channel. No clipping (N19). Non-finite pixels stay non-finite. |
| `to_columns` / `from_columns` | The columns of §8.1 as text in the format of §8.2; `from_columns(to_columns(p)) == p` exactly. |
| `to_dict` / `from_dict` | JSON-compatible dictionaries (§8.3) that round-trip exactly. |
| `image_summaries`, `intensity_checks` | §7. |

The library never prints. Errors are raised as exceptions with plain-language messages;
warnings are returned as `Issue` objects for the calling brick to report (§9).

---

## 3. Methods

### 3.1 Identifiers

The method identifier is a setting (`settings.method`); the written name is what
`normalization_method` holds (§8.1). The fluorescence names are those written today.

| Image type | `method` | Written name |
|---|---|---|
| `fluorescence` | `minmax`, `percentile` (default), `zscore`, `robust_zscore` | `min-max`, `percentile <low>-<high>` (e.g. `percentile 1-99.8`), `z-score`, `robust z-score` |
| `rgb` | `percentile` (the only one) | `rgb percentile <low>-<high>` (e.g. `rgb percentile 1-99.8`) |

Percentile values in names are formatted with `:g` (as `percentile_range` does today).

### 3.2 Fluorescence

Each channel `c` gets its own offset and scale from the pooled statistics of that channel in
the group, with the definitions of §1.2:

| `method` | offset | scale | Settings |
|---|---|---|---|
| `minmax` | exact minimum | maximum − minimum | — |
| `percentile` | `P(low)` | `P(high) − P(low)` | `percentile = [low, high]`, `0 ≤ low < high ≤ 100`, default `[1.0, 99.8]` |
| `zscore` | mean | population standard deviation | — |
| `robust_zscore` | median | MAD × `consistency` | `consistency`, default `NORMAL_CONSISTENCY` = 1.482602218505602 |

`apply`: `y_c = (x_c − offset_c) / scale_c`, computed in float64, returned as float32. A zero
or non-finite scale is an error (§1.2.4).

### 3.3 RGB — StarDist percentile normalization

The normalization StarDist applies to its inputs (`csbdeep.utils.normalize` with
`axis=(0, 1, 2)`, "normalize channels jointly"):

```
offset = P(low)   over the R, G and B pixels of the group together
scale  = P(high) − P(low) + 1e-20
y_c    = (x_c − offset) / scale          for c = R, G, B
```

- `percentile = [low, high]`, default `[1.0, 99.8]` (StarDist's examples), same rule as §3.2.
- One pooled histogram of the R, G and B planes of the group (§4): the three channels' rows
  of `plane_histograms.csv` pooled together, or the group's pixels of all three channels.
- Every channel gets the same offset and scale, so colour ratios are kept. Output float32,
  no clipping (as `normalize(..., clip=False)`).
- **Percentile definition (N40):** NumPy's default linear interpolation, as StarDist, `P(q) = v[k] + (h − k)·(v[k+1] − v[k])` with
  `h = (N − 1)·q/100`, `k = floor(h)` and `v` the sorted pooled values, computed from the
  cumulative counts; exact for histograms with one bin per value, within the histogram bound
  otherwise. Because of the `1e-20` term, a group without spread gives zeros instead of an
  error, and such patches are not discarded.

---

## 4. Scopes, groups and weights

### 4.1 Scopes

| `scope` | Pooled pixels | Group key | `normalization_group` |
|---|---|---|---|
| `biological_repeat` | Every plane of the repeat group: one biological repeat in every condition, all technical repeats | `(repeat group label,)` | `<repeat group label>` |
| `image` (default) | All Z and T planes of one image | `source_relative_path` | `<source_relative_path>` |
| `patch` | The patch's own pixels | calling brick's patch identifier | `<source_relative_path>::<patch id>` |

- There is no experiment, condition or technical-repeat scope (N29). `PlaneRecord.group_key`
  and `group_records` keep their existing behaviour (§1.1); the scopes above use the new
  `group_key` function.
- Every pooled plane of the group is used, including planes the calling brick does not
  normalize (for example Z slices without cells).

### 4.2 Repeat groups

- **Default:** the repeat group of a row is its `biological_repeat` label. Rows of different
  conditions with the same label are one group.
- **Pairing:** `repeat_groups` maps `(condition, biological_repeat)` to a label chosen in the
  preprocessing brick (`experiment_design.csv`, follow-up P3). Every `(condition,
  biological_repeat)` of the statistics must be in the mapping when one is given; a missing
  one is an error (§9.1).
- **`single`:** a condition without biological-repeat folders has the label `single`, which
  forms a group with the other conditions' `single` rows, or with the repeat it is paired
  with.
- **Incomplete groups:** a repeat group without rows from every condition of the statistics
  gives the warning `repeat_missing_conditions` (§9.2) and is still used.

### 4.3 Pooling weights

| `pooling` | Rule |
|---|---|
| `pixels` (default) | The group's histograms are added: each pixel counts once (§1.2.3) |
| `conditions` | Each condition's histograms in the group are added, then scaled by `1 / n_condition` (its valid pixel count), so every condition has total weight 1; percentiles, mean, standard deviation, median and MAD use these weights (percentile: smallest representative whose cumulative weight reaches `q/100` of the total) |

Minimum and maximum do not depend on the weights. At the `image` and `patch` scopes, and in a
group with a single condition, both options give identical parameters. `normalization_exact`
keeps its meaning (§1.2.4): weights do not make a histogram inexact.

### 4.4 Channels and time

- Parameters are per channel for `fluorescence` and shared by R, G and B for `rgb` (§3.3).
- Time points are pooled by default. `per_time_point=true` gives each time point its own
  group; `::t=<t>` is appended to `normalization_group`.

### 4.5 Availability

| Statistics given to the library | Scopes |
|---|---|
| `plane_histograms.csv` | All three |
| None (pixels only) | `image` and `patch`, computed from the image's or patch's own pixels |

Image-scope parameters from pixels equal those from the file whenever the histograms are
exact. Which scopes a calling brick offers without the file is the calling brick's choice
(CellPatcher keeps its current behaviour, N29).

---

## 5. Image types

| `image_type` | Channels | Pixel types | Methods |
|---|---|---|---|
| `fluorescence` (default) | any | every type of §1.2.1 | §3.2 |
| `rgb` | exactly 3 (R, G, B) | integer types | §3.3 only |

- The type is chosen by the user: in the preprocessing brick, which suggests `rgb` when the
  metadata has an RGB samples axis (TIFF photometric RGB, three samples per pixel) and
  `fluorescence` otherwise (N34), and records it in its manifest; or in the calling brick's
  configuration when there are no preprocessing outputs.
- An `rgb` image must keep its three channels together: a calling brick that splits channels
  into separate patches cannot use `rgb`.

---

## 6. Statistics file

The library defines the statistics and owns the reader and writer (N2); the preprocessing
brick runs the computation over the dataset and writes the file next to its
`run_manifest.json`.

### 6.1 Shared columns

The file starts with these columns, in this order: `position_id`, `image_file`, `z`, `c`, `t`,
`condition`, `biological_repeat`, `technical_repeat`, `source_relative_path`, `plane_id`.
Their meaning is that of §6.2. Labels are always read as text.

Common format: UTF-8, comma-separated, one header row, rows in a deterministic order, floats
written at full precision (Python `repr`), blanks for missing values, written to a temporary
file and replaced only once writing succeeded.

### 6.2 `plane_histograms.csv` (schema version 2)

Unchanged: one row per 2D plane and channel (an RGB plane has three rows, `c` = 0, 1, 2),
with the histogram of §1.2.2, written and read by the moved `csv_io.py` (§1.1). The
preprocessing contract §5 defined it until follow-up P1 moved the text here:

| Column | Type | Meaning |
|---|---|---|
| `position_id` | text or blank | The XY position within its technical repeat (MorphoBricks decision D02): from reliable metadata, or the image file name without extension when one file holds exactly one position, as morphobricks-preprocessing writes it. Blank only in tables written before it was filled. |
| `image_file` | text | File name. |
| `z`, `c`, `t` | integer | Zero-based plane coordinates, always written; 0 for an axis the image does not have (preprocessing contract §2). |
| `condition`, `biological_repeat`, `technical_repeat` | text | Folder labels; `single` when absent. Always read as text. |
| `source_relative_path` | text | Path relative to the experiment root, `/`-separated. |
| `plane_id` | text | `source_relative_path::z=…,c=…,t=…`, always with all three coordinates (`control/img.tif::z=0,c=0,t=0` for a 2D image). Row key. |
| `dtype` | text | NumPy dtype name, e.g. `uint16`. |
| `bit_depth` | integer | Storage bits (8, 16, 32, 64). |
| `pixel_min`, `pixel_max` | number or blank | Exact finite extrema. Floats written at full precision. |
| `total_pixel_count` | integer | All pixels in the plane. |
| `valid_pixel_count` | integer | Finite pixels; equals the sum of the bin columns. |
| `binning_min`, `binning_max` | number or blank | Range the bins cover (equal to the extrema). |
| `bin_count` | integer | Number of bin columns. |
| `schema_version` | text | `2`. Version 1 left `z`, `c`, `t` blank for absent axes and out of `plane_id`; a table of another version is recomputed on the next run. |
| `histogram_bin_00000` … | integer | Pixel count per bin. |

The biological-repeat pairing, its confirmation and the controls are not part of this file;
they are in the preprocessing brick's `experiment_design.csv` (follow-up P3).

---

## 7. Intensity checks

Computed from `plane_histograms.csv` rows alone (N38).

### 7.1 Image summaries

`image_summaries(records)` returns one row per image and channel, pooling all Z and T planes
of the image (§1.2.3): `source_relative_path`, `condition`, `biological_repeat`, `c`,
`median`, `p_low`, `p_high` (default the 1st and 99.8th percentiles), `saturated_fraction`
and `exact`.

- `saturated_fraction`: pixels equal to the type's maximum over valid pixels, for integer
  types whose histogram holds the maximum value in a bin of its own; blank otherwise (float
  images, or inexact top bins).

### 7.2 Checks

`intensity_checks` compares images only within one condition, so differences between
conditions are never reported. A robust deviation is `MAD × 1.482602218505602` of the
compared values.

| Code | Fires when | Setting, default |
|---|---|---|
| `image_intensity_outlier` | An image's `median` or `p_high` in a channel is more than `outlier_limit` robust deviations from the median of the other images of the same condition and biological repeat | `outlier_limit` 3.0; groups with fewer than `min_images` images are skipped (`min_images` 5, **Provisional**) |
| `repeat_intensity_shift` | In one condition and channel, the median of a biological repeat's image `median` values is more than `outlier_limit` robust deviations of the other repeats' image medians away from their median | `outlier_limit` 3.0; needs at least two biological repeats in the condition |
| `saturated_image` | `saturated_fraction` above `saturation_limit` in a channel | `saturation_limit` 0.01 |
| `no_intensity_spread` | `p_low == p_high` in a channel: the image cannot be normalized at the image scope with a percentile method | — |

A zero robust deviation (identical values) skips the comparison rather than firing. Each
warning names the image or repeat, the channel, the value and the group value it was
compared with.

---

## 8. Parameter columns

Parameters are written as numeric columns, never as JSON (N21). Only these columns are
written (N23); the settings are written once in the calling brick's run configuration.

### 8.1 Columns

| Column | Type | Meaning |
|---|---|---|
| `normalization_method` | text | Written name (§3.1) |
| `normalization_scope` | text | `biological_repeat`, `image` or `patch` |
| `normalization_group` | text | §4.1 group text |
| `normalization_exact` | `true`/`false` | `true` exactly when every pooled histogram has `max_representative_error == 0` (§1.2.4), at every scope |
| `normalization_offset`, `normalization_scale` | text | One value per channel of the patch, in channel order, separated by `;`; a single value for one channel; for `rgb`, the shared value three times |

### 8.2 Formatting

- Numbers at full precision (Python `repr` of the float), read back unchanged.
- The existing linear columns keep their names and meaning (CellPatcher follow-up C1).
- Column order within a file is the calling brick's; readers select columns by name.

### 8.3 Run level

The calling brick stores `NormalizationSettings.to_dict()` once per run:

| Key | Default |
|---|---|
| `library_version` | Version of `morphobricks-normalization` |
| `image_type` | `fluorescence` |
| `method` | `percentile` |
| `percentile` | `[1.0, 99.8]` |
| `consistency` | 1.482602218505602 (`robust_zscore` only) |
| `scope` | `image` |
| `pooling` | `pixels` (`biological_repeat` only) |
| `per_time_point` | `false` |

Keys that do not apply to the method or scope are omitted. With `biological_repeat`, the
calling brick also records which `experiment_design.csv` it used (name and SHA-256), and the
statistics file's name, `schema_version` and SHA-256. Raw patches are normalized on load from
the row's columns alone.

---

## 9. Validation

Every message states in plain language what happened, names the affected images, groups or
patches, and suggests an action, in the style of the other MorphoBricks bricks.

### 9.1 Errors

Raised as exceptions before any pixel is normalized; the calling brick blocks the run.

| Situation | Exception | Message |
|---|---|---|
| A method other than `percentile` with `rgb` | `MethodNotAllowedError` | "RGB images are normalized with StarDist's percentile method only, which scales the three colour channels together. Use method 'percentile'." |
| An unknown scope, including the removed `condition`, `technical_repeat` and `experiment` | `ValueError` | "Scope '{scope}' does not exist. Choose 'biological_repeat' (one biological repeat across all conditions), 'image' or 'patch'." |
| `biological_repeat` without the statistics file | `MethodNotAllowedError` | "The biological-repeat scope needs the preprocessing brick's plane_histograms.csv, which was not given. Run the preprocessing brick on this experiment, or use the image or patch scope." |
| `rgb` without exactly three channels, or with a float pixel type | `ValueError` | "RGB normalization needs images with exactly three 8- or 16-bit colour channels, but these images have {n} channels of type {dtype}. Check the axis mapping (the samples axis must be the channel axis), or choose image type 'fluorescence'." |
| A `(condition, biological_repeat)` missing from `repeat_groups` | `ValueError` | Names the missing pairs and asks to update the experiment design in the preprocessing brick. |
| Unsupported pixel type; bin count not allowed (§1.2) | `UnsupportedDtypeError`, `ValueError` | Existing messages |
| Group without finite pixels | `NoValidPixelsError` | Existing message |
| Zero or non-finite scale (fluorescence only; `rgb` adds `1e-20`, §3.3) | `ValueError` | Existing message |
| Percentiles outside `0 ≤ low < high ≤ 100`; unknown `pooling` or `image_type` | `ValueError` | Existing message for percentiles; the allowed values otherwise |
| Statistics file with unexpected columns, schema version or inconsistent counts | `CsvFormatError` | Existing messages |
| `from_columns` with a missing or unreadable column, or a list of the wrong length | `ValueError` | Names the column |

### 9.2 Warnings

Returned as `Issue` objects; processing continues and the calling brick reports them.

| Code | Situation |
|---|---|
| `biological_repeat_scope` | Scope `biological_repeat`: intensity differences between biological repeats are removed, differences between conditions within a repeat are kept |
| `repeat_missing_conditions` | A repeat group without images from every condition (§4.2) |
| `approximate_parameters` | Parameters from inexact histograms (`normalization_exact = false`) |
| `planes_without_finite_pixels` | Planes without finite pixels excluded from a group that still has others |
| `patches_discarded` | Patches without spread or finite pixels, listed in one warning with a suggested scope |
| The four codes of §7.2 | Intensity checks |

---

## 10. Provisional values

Values left to measurement (plan §13) are marked **Provisional** above: `min_images`.
