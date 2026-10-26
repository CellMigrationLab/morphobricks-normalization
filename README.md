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

The scopes, the RGB method and the intensity checks follow in Milestones 2–4 (see the
plan).

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
