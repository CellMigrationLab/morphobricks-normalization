# Changelog

Release notes and migration notes of `morphobricks-normalization`. Numbers are pull requests and
issues of [CellMigrationLab/morphobricks-normalization](https://github.com/CellMigrationLab/morphobricks-normalization).
The decisions are in [docs/plan.md](docs/plan.md) §12.

## Unreleased (after v0.3.4)

Obsolete-code sweep (SWEEP-NM; inventory rows NM-01, NM-02, NM-04). No fitted number, histogram
or written table changes: `tools/dump_outputs.py` (and `--corpus`) give identical SHA-256 digests on
`main` 1c9ef7c and on this change, and the 400 golden fits are bit-identical.

- **Breaking, NM-04:** `PlaneRecord.position_id` is required and keyword-only. `None`, blank or
  non-text values are refused with a `ValueError`, and `read_plane_histograms` refuses a row whose
  `position_id` is blank with a `CsvFormatError` (it used to read it as `None`). Nothing is filled in
  or guessed. Callers that build records (preprocessing, tests) must pass it; a table written
  before the field was filled must be recomputed.
- **NM-01:** the readers behind the old-version refusals are removed (`_EARLIER_LAYOUT`,
  `_VERSION_3_FEATURES`, the second layout pass). A table of schema version 1, 2 or 3 is recognised
  by the `schema_version` value of its rows alone and refused with the same messages as before,
  which name the version and the fix. A version 1 to 3 header with no rows has no value to read and
  is refused as a table with unexpected columns.
- **NM-02:** `tests/test_schema_v2_equivalence.py` is now `tests/test_fit_goldens.py`, a science
  golden of the 400 fitted parameters of simulated planes against the numbers stored in
  `tests/data/golden_*.{npz,json}`. The checks that only compared tables and one-channel rows of the
  old layout, the two version-2 CSV fixtures and `make_v2_fixture.py` are deleted.

Merged to `main` since the `v0.3.4` tag; from the commit history, not yet released. Each is a
new refusal or a numerical-range fix (see the pull requests for the exact messages):

- Named refusals for a non-finite z-score scale and for opposing infinite tail sums in the
  pooled mean (#72).
- Percentiles are interpolated between finite extremes without overflowing the gap (#74).
- Float bin centres are computed without overflowing near the float64 limits (#76).
- Float histograms over a very narrow range (down to the smallest subnormal) bin pixels and place bin centres correctly: the position is a bounded fraction times the bin count where the halved arithmetic would round to zero or overflow. Histograms of ordinary ranges are bit-identical (#84).
- Ultra-narrow float ranges (fewer representable floats than bins, e.g. `[0, 5e-324]` with 4 bins): documented and tested, no output changes. Bins no float64 value can fall in may share a representative centre; the centre of every bin holding two or more representable values maps back to it, and a bin holding exactly one can have a centre that rounds into the adjacent bin. New `bin_reachability` helpers state this; `tools/dump_outputs.py --corpus` dumps a generated corpus (all pixel types, extremes, subnormals, one-ulp to full-range float ranges, 2 to 1024 bins) that is bit-identical against `main` (#86).
- Image summaries whose channels disagree, pooled summaries with different percentile
  definitions, and images whose Z/T planes have different pixel types are refused (#77).
- RGB: channel count and pixel type are checked before histograms are built, RGB groups with
  different pixel types are refused before pooling, `Parameters.select` of a strict subset of
  R, G and B is refused, and `apply_rgb` uses one float32 copy (#78).
- `range_check_issues` refuses malformed channels and mixed settings (#79).
- The non-finite scale refusal names no method-specific cause: it says finite pixel values overflowed float64 arithmetic, which is true for min-max, percentile range and robust z-score as well as z-score (#82).
- New `iter_image_summaries(records)`: streaming, bounded-memory per-image summaries for rows
  grouped by image (contiguity required, otherwise refused); `image_summaries` is unchanged (#58).

## Release notes moved from the README

The sections below were the README's "Status" section; they are kept as they were written.

Pre-alpha. Milestone 0 (the contract) is done. Milestone 1 moved the existing code here
from morphobricks-preprocessing unchanged, with identical outputs:

- Supported pixel types and bin counts (`dtypes`)
- Per-plane histograms, also of planes read in blocks (`histogram`)
- Plane records, plane identifiers and grouping by labels (`records`)
- Reading and writing `plane_histograms.csv` (`csv_io`)
- Group statistics pooled from histograms (`pooling`)
- Min-max, percentile, z-score and robust z-score parameters (`normalization`)

Milestone 2 added the settings, the scopes and the fitting interface for fluorescence
images:

- `NormalizationSettings` (image type, method, percentiles, consistency, scope, pooling
  weight, per-time-point groups), stored with `to_dict`/`from_dict`, and `check_settings`
  (`settings`)
- The `biological_repeat`, `image` and `patch` scopes: `group_key`, `group_name` and
  `scope_groups`, with repeat groups matched by name or paired by the calling brick
  (`scopes`)
- Pooling weighted per pixel or equally per condition (`WeightedHistogram` in `pooling`)
- `fit` from `plane_histograms.csv` rows or from pixels, and `Parameters` with `apply`,
  `to_columns` and `from_columns` (`fitting`)
- Warnings returned as `Issue` objects (`issues`)

```python
from morphobricks_normalization import NormalizationSettings, check_settings, fit, scope_groups
from morphobricks_normalization import group_name, read_plane_histograms

settings = NormalizationSettings(scope="biological_repeat", pooling="pixels")
warnings = check_settings(settings, statistics_file=True)
records = read_plane_histograms("plane_histograms.csv")
groups, group_warnings = scope_groups(records, settings, repeat_groups=None)
for key, rows in groups.items():
    parameters = fit(rows, settings, group=group_name(key))
    columns = parameters.to_columns()  # normalization_* and Normalization_*_C<i> columns
```

Milestone 2b put the channels across columns (decisions N41–N47):

- `plane_histograms.csv` schema version 3: one row per XYC plane (image, Z and T), with
  each channel's extrema, pixel counts and bins in CellProfiler-style columns
  (`Intensity_MinIntensity_C0`, `Histogram_Count_C1_00042`, ...). Tables of versions 1 and
  2 are refused with a message asking to run the preprocessing brick again; images with
  different numbers of channels are an error.
- `PlaneRecord.histograms` holds one histogram per channel; pooling and fitting take the
  channel index. Histograms, pooled statistics and parameters are bit-identical to those of
  Milestone 2, checked against a version-2 fixture written by that code (`tests/data`).
- Parameters are written as `Normalization_Offset_C<i>` and `Normalization_Scale_C<i>`, one
  pair per channel; `Parameters.select([i])` gives the pair of a single-channel patch.

Milestone 3 added RGB images, with StarDist's percentile normalization (decisions N33,
N39, N40, N48, N49):

- `NormalizationSettings(image_type="rgb")` allows only `method="percentile"` (default
  1–99.8, written `rgb percentile 1-99.8`); another method raises `MethodNotAllowedError`.
- `fit` pools the R, G and B values of the group together, at every scope, from
  `plane_histograms.csv` rows or from pixels: `offset = P(low)`, `scale = P(high) − P(low) +
  1e-20`, with NumPy's linear-interpolation percentiles (`linear_percentile` in `pooling`,
  exact for 8- and 16-bit histograms). The three channels share the values, so colour
  ratios are kept; a region without spread becomes zeros instead of an error.
- `Parameters.apply` repeats CSBDeep's float32 arithmetic, so the result equals
  `csbdeep.utils.normalize(x, 1, 99.8, axis=(0, 1, 2))` bit for bit (`rgb`). CSBDeep is
  not a dependency; the tests compare with it when it is installed.
- RGB needs exactly three channels of an integer type; anything else is refused with a
  message that suggests checking the axis mapping or choosing `fluorescence`.
- Fluorescence methods are unchanged, bit for bit.

```python
settings = NormalizationSettings(image_type="rgb", scope="image")
parameters = fit(rgb_pixels, settings, group="img.tif", channel_axis=-1)  # (Y, X, 3)
normalized = parameters.apply(rgb_pixels, channel_axis=-1)
```

Milestone 4 added the intensity checks, computed from `plane_histograms.csv` rows alone
(decisions N38, N50–N53):

- `image_summaries(records)` gives one `ImageSummary` per image, all Z and T planes pooled per
  channel: median, 1st and 99.8th percentiles, the fraction of pixels at the pixel type's
  maximum and whether the histograms were exact. `to_row()` gives the CellProfiler-style
  columns (`Intensity_MedianIntensity_C0`, ...) for a table; the library writes no file.
- `intensity_checks(summaries, settings)` returns warnings: images far from the other images
  of their condition and biological repeat (`image_intensity_outlier`), biological repeats
  shifted from the other repeats of their condition (`repeat_intensity_shift`), saturated
  images and images without spread. Images are compared only within a condition, so a
  treatment effect is never reported. Each `Issue` names the image or repeat and the channel
  (`Issue.channel`).
- The limits are an `IntensityCheckSettings` (default 3 robust deviations, 1 % saturated
  pixels, at least 5 images per comparison, provisional).
- `unchecked_channels(summaries)` names each image channel without finite pixels
  (`channel_without_finite_pixels`): its summaries are blank, so the checks cannot see it.

```python
from morphobricks_normalization import IntensityCheckSettings, image_summaries, intensity_checks

summaries = image_summaries(records)
table = [summary.to_row() for summary in summaries]  # e.g. pandas.DataFrame(table)
for issue in intensity_checks(summaries, IntensityCheckSettings(outlier_limit=3.0)):
    print(issue.code, issue.items, issue.channel, issue.message)
```

Milestones 2–4 are released as `v0.2.0` (27 October 2026) and Milestone 5 as `v0.3.1`
(28 October 2026); the other bricks pin the latest tag. The tag `v0.3.0` holds the same
code but reports version 0.2.0, as its `pyproject.toml` was not updated; use `v0.3.1`.
`v0.3.2` (30 October 2026) refuses a `plane_histograms.csv` row whose `plane_id`, `image_file`
or `bit_depth` disagrees with its other columns (#18).
`v0.3.3` (3 November 2026) refuses corrupt or ambiguous statistics and parameters instead
of using them (issues #25, #26, #31, #36–#38, #42–#48); valid inputs give bit-identical
results. See [Migrating to v0.3.3](#migrating-to-v033).
`v0.3.4` (5 November 2026) refuses a histogram whose missing `pixel_min` or `pixel_max` used
to be filled in (#47, PR #65), makes means, standard deviations and z-score fits independent
of the CPU thread count, changing their last bits (#67, PR #69), and adds the section 9
code-standards gate (PR #66). See [Migrating to v0.3.4](#migrating-to-v034).

Milestone 5 added a robust histogram range for histograms that are not exact (float planes,
and integer planes with more values than bins), for issue #12 (decisions N54–N60):

- By default nothing changes: bins span each plane's minimum to maximum.
- `HistogramSettings(histogram_range="robust")` bins between the plane's 0.01 % and 99.99 %
  quantiles (adjustable) and keeps the pixels outside as underflow and overflow counts and
  sums, so one hot pixel no longer squeezes the other pixels into a few bins. Exact 8- and
  16-bit histograms never change.
- `plane_histograms.csv` schema version 4 adds `Histogram_UnderflowCount_C<i>`,
  `Histogram_UnderflowSum_C<i>`, `Histogram_UnderflowSumOfSquares_C<i>` and the same for
  `Overflow`; tables of version 3 are refused with a message asking to run the
  preprocessing brick again.
- `histogram_with_range_check` computes a plane's histogram together with a `RangeCheck`: how
  many bins its central range (1st to 99.8th percentile) covers, and how many it would cover
  with the robust range. `range_check_issues` turns the checks into one warning per channel
  for planes below 100 bins, for the preprocessing brick to show.

```python
from morphobricks_normalization import HistogramSettings, histogram_with_range_check
from morphobricks_normalization import range_check_issues

settings = HistogramSettings(histogram_range="minmax")  # or "robust"
histogram, check = histogram_with_range_check(lambda: (plane,), plane.dtype, settings=settings)
warnings = range_check_issues([("img.tif::z=0,t=0", 0, check)])
```

### v0.3.4 (5 November 2026)

- **`PlaneHistogram` refuses a missing extremum (#65, issue #47)**: it no longer fills a
  missing `pixel_min` or `pixel_max` from `binning_min` or `binning_max`. A caller that
  builds a histogram of a plane with valid pixels passes both extrema explicitly, e.g.
  `pixel_min=lo, pixel_max=hi` for a min-max histogram. A `plane_histograms.csv` row with
  one blank intensity limit and valid pixels is refused with a `CsvFormatError` naming the
  column. This change alone leaves every output for valid inputs bitwise unchanged.
- **Means and standard deviations no longer depend on CPU threads (#69, issue #67).** They
  summed with `np.dot`, which the BLAS library splits across its threads, so a z-score fit
  changed in its last bits with `OMP_NUM_THREADS` or `OPENBLAS_NUM_THREADS`. They now add the
  same products with `math.fsum` (the correctly rounded sum): the same bits for any thread
  count, BLAS or platform, never less accurate than before. A sum that overflows float64 is
  still `nan` or `inf`, refused as before. The last bits change (see the migration note).
- **Section 9 code-standards gate (#66)**: `python tools/check.py` runs the shared code
  rules, Ruff and pytest against a shrink-only baseline (`tools/code_rules_baseline.json`,
  taken at `v0.3.3`); CI runs its `--rules` part. No library behaviour changes.

### Migrating to v0.3.4

Results are **not** all bit-identical to v0.3.3:

- **Mean, standard deviation and z-score change in their last bits (#67).** The pooled
  `mean` and `std` (`PooledHistogram`, `WeightedHistogram`, `pooled_statistics`) and the
  `zscore` offset and scale, at every scope, are now the correctly rounded sums of the same
  products, so they can differ from v0.3.3 by a few units in the last place: at most
  3.7e-14 relative on `tools/dump_outputs.py` (134 numbers on 86 of its 1,788 lines; 89 of
  them by 1 ulp, the largest by 317 ulps, for int64 planes with values near 2**53). Cached
  z-score parameters or features computed from them with v0.3.3 are not bitwise
  reproducible with v0.3.4; recompute them rather than mixing the two. Every other output
  (histograms, `plane_histograms.csv`, minimum, maximum, percentiles, median, MAD, the
  min-max, percentile, robust z-score and RGB methods) is bitwise unchanged.

One input that was accepted before is now refused:

- **Histograms with a missing extremum:** a `PlaneHistogram` of a plane with valid pixels
  whose `pixel_min` or `pixel_max` is `None` raises `ValueError` (`CsvFormatError` when read
  from `plane_histograms.csv`, naming the blank column). Earlier versions filled it from
  `binning_min` or `binning_max`. Pass the plane's actual extrema, e.g.
  `PlaneHistogram(..., pixel_min=lo, pixel_max=hi)`. A channel without valid pixels may still
  leave both blank.

### Migrating to v0.3.3

No valid statistics change: on the same inputs, histograms, `plane_histograms.csv` bytes,
pooled statistics, fitted parameters and normalized pixels are bit-identical to `v0.3.2`.
These inputs are now refused with `ValueError` (`CsvFormatError` when read from the file):

- **Pixel blocks:** a non-empty block of another pixel type than the plane's. A block in the
  other byte order (big-endian TIFF) is now converted by value; 16-bit ones were misread.
- **Histograms:** negative or wrapping bin counts, `valid > total`, missing or disordered
  intensity limits on a plane with valid pixels, NaN or impossible underflow/overflow sums.
  `PlaneHistogram.counts` must be a one-dimensional int64 array.
- **Tables:** a `plane_histograms.csv` with the header but no rows; two rows with the same
  `plane_id` (also refused by `write_plane_histograms`, `scope_groups`, `fit` and
  `image_summaries`).
- **Plane coordinates:** `PlaneRecord` `z` and `t` must be whole numbers 0 or above (`None`
  is still 0); booleans, floats (even `1.0`), text and negative values are refused.
- **Repeat groups:** `repeat_groups` that put two biological repeats of one condition in one
  repeat group, or give a present pair an empty or non-text label.
- **Pooling:** a group of more than 2**63 − 1 finite pixels. Counts above 2**53 are now
  exact.
- **Parameters:** a non-finite offset or a scale that is not finite and above 0
  (`Parameters`, `NormalizationParameters`); a channel index that is not a whole number 0 or
  above (booleans, `1.0`, `"1"` are no longer truncated).
- **`Parameters.from_columns`:** a `normalization_scope` that is not a scope, or that differs
  from `settings.scope` (#37). Pass the settings of the run that wrote the row, scope included,
  e.g. `NormalizationSettings(method=..., percentile=..., scope=run_scope)`.
- **Summaries:** `ImageSummary.exact` (`Histogram_Exact_C<i>`) is `None`, not `True`, for a
  channel without finite pixels. The new `unchecked_channels(summaries)` returns a
  `channel_without_finite_pixels` warning for each such channel; `intensity_checks` still
  returns its four codes only, so calling bricks add the new code themselves (#57).
