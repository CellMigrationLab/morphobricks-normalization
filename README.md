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

Pre-alpha; the latest release is v0.4.0. The library offers:

- supported pixel types, bin counts and per-plane histograms (`dtypes`, `histogram`), with a robust range and a range check;
- plane records, `plane_histograms.csv` reading and writing (schema version 4) and pooling of group statistics (`records`, `csv_io`, `pooling`);
- settings, scopes, repeat groups and `fit`, with `Parameters` (`apply`, `to_columns`, `from_columns`), for fluorescence and RGB images;
- per-image summaries and intensity checks, returned as `Issue` warnings.

Release notes and migration notes are in [CHANGELOG.md](CHANGELOG.md).

Example:

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

## Installation

```bash
pip install "git+https://github.com/CellMigrationLab/morphobricks-normalization.git@v0.4.0"
```

For development, from a clone:

```bash
pip install -e ".[dev]"
python -m pytest
```

## Documents

| Document | Content |
|---|---|
| [docs/plan.md](docs/plan.md) | Goals, milestones and the owner's decisions N1–N60 |
| [docs/contract.md](docs/contract.md) | Rules other bricks rely on: methods, scopes and repeat groups, image types, the statistics file, intensity checks, parameter columns, validation |
| [CHANGELOG.md](CHANGELOG.md) | Release notes and migration notes |
| [AGENTS.md](AGENTS.md) | Rules for anyone changing this repository |

## Related repositories

- [morphobricks-preprocessing](https://github.com/CellMigrationLab/morphobricks-preprocessing)
- [CellPatcher](https://github.com/CellMigrationLab/CellPatcher)
