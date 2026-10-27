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
    columns = parameters.to_columns()  # normalization_* columns, numbers as text
```

The RGB method and the intensity checks follow in Milestones 3–4 (see the plan).

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
| [docs/plan.md](docs/plan.md) | Goals, milestones and the owner's decisions N1–N40 |
| [docs/contract.md](docs/contract.md) | Rules other bricks rely on: methods, scopes and repeat groups, image types, the statistics file, intensity checks, parameter columns, validation |
| [docs/followup_preprocessing.md](docs/followup_preprocessing.md) | Changes to morphobricks-preprocessing |
| [docs/followup_cellpatcher.md](docs/followup_cellpatcher.md) | Changes to CellPatcher |
| [AGENTS.md](AGENTS.md) | Rules for anyone changing this repository |

## Related repositories

- [morphobricks-preprocessing](https://github.com/CellMigrationLab/morphobricks-preprocessing)
- [CellPatcher](https://github.com/CellMigrationLab/CellPatcher)
