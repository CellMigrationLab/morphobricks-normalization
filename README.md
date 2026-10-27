# morphobricks-normalization

The normalization brick of MorphoBricks: one importable Python library with the
normalization methods for fluorescence and RGB microscopy images, the statistics those methods
need, intensity checks on those statistics, and the file format that carries statistics
between bricks.

- **morphobricks-preprocessing** runs this library over a dataset to write
  `plane_histograms.csv` and to show the intensity checks in its notebook.
- **CellPatcher** reads the statistics and normalizes patches with this library.

The library has no widgets, notebooks or folder walking: it receives pixel arrays,
statistics and settings, and returns statistics, parameters, warnings and normalized arrays.
It never modifies source images. Image correction (flat-field, background, illumination) is
out of scope for now.

## Methods and scopes

| Image type | Methods |
|---|---|
| Fluorescence | min-max, percentile range (default 1–99.8), z-score, robust z-score, each channel separately |
| RGB | StarDist's percentile normalization (1–99.8), the three colour channels together |

| Scope | Parameters from |
|---|---|
| Biological repeat | One biological repeat across all conditions, technical repeats merged |
| Image (default) | All Z and T planes of one image |
| Patch | The patch's own pixels |

## Status

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

```python
from morphobricks_normalization import IntensityCheckSettings, image_summaries, intensity_checks

summaries = image_summaries(records)
table = [summary.to_row() for summary in summaries]  # e.g. pandas.DataFrame(table)
for issue in intensity_checks(summaries, IntensityCheckSettings(outlier_limit=3.0)):
    print(issue.code, issue.items, issue.channel, issue.message)
```

The next step is the `v0.2.0` release, pinned by the other bricks (see the plan).

## Installation

```bash
pip install "git+https://github.com/CellMigrationLab/morphobricks-normalization.git"
```

For development, from a clone:

```bash
pip install -e ".[dev]"
python -m pytest
```

## Documents

| Document | Content |
|---|---|
| [docs/plan.md](docs/plan.md) | Goals, milestones and the owner's decisions N1–N53 |
| [docs/contract.md](docs/contract.md) | Rules other bricks rely on: methods, scopes and repeat groups, image types, the statistics file, intensity checks, parameter columns, validation |
| [docs/followup_preprocessing.md](docs/followup_preprocessing.md) | Changes to morphobricks-preprocessing |
| [docs/followup_cellpatcher.md](docs/followup_cellpatcher.md) | Changes to CellPatcher |
| [AGENTS.md](AGENTS.md) | Rules for anyone changing this repository |

## Related repositories

- [morphobricks-preprocessing](https://github.com/CellMigrationLab/morphobricks-preprocessing)
- [CellPatcher](https://github.com/CellMigrationLab/CellPatcher)
