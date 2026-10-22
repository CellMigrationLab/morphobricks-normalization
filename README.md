# morphobricks-normalization

The normalization brick of MorphoBricks: one importable Python library with every
normalization method for microscopy images, the statistics those methods need, the modality
rules that decide which methods are meaningful, and the file formats that carry statistics
between bricks.

- **morphobricks-preprocessing** runs this library over a dataset to write the statistics
  files and to preview normalized images in its notebook.
- **CellPatcher** reads the statistics files and normalizes patches with this library.

The library has no widgets, notebooks or folder walking: it receives pixel arrays,
statistics and settings, and returns statistics, parameters and normalized arrays. It never
modifies source images. Image correction (flat-field, background, illumination) is out of
scope for now.

## Methods

| Family | Methods |
|---|---|
| A. Linear per channel | min-max, percentile range, z-score, robust z-score |
| B. Linear shared across channels | The same four, with one offset and scale for several channels |
| C. Optical density | OD, then a family-A or B method |
| D. Histogram matching | Onto a reference image or group |
| E. Colour and stain | Ruifrok colour deconvolution, Reinhard, Macenko, Vahadane |
| Option | Luminance (grayscale) conversion of RGB images |

Which methods are offered depends on the imaging modality (`fluorescence`,
`brightfield_gray`, `brightfield_rgb`, `phase_contrast`, `histology_rgb`,
`electron_microscopy`) and on the purpose (`model_input` or `quantitative`).

## Status

Pre-alpha. Milestone 0 (the contract) is done. Milestone 1 moved the existing code here
from morphobricks-preprocessing unchanged, with identical outputs:

- Supported pixel types and bin counts (`dtypes`)
- Per-plane histograms, also of planes read in blocks (`histogram`)
- Plane records, plane identifiers and grouping by labels (`records`)
- Reading and writing `plane_histograms.csv` (`csv_io`)
- Group statistics pooled from histograms (`pooling`)
- Min-max, percentile, z-score and robust z-score parameters (`normalization`)

The other methods, modality profiles and colour statistics follow in Milestones 2–5
(see the plan).

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
| [docs/plan.md](docs/plan.md) | Goals, milestones and the owner's decisions N1–N28 |
| [docs/contract.md](docs/contract.md) | Rules other bricks rely on: methods, scopes, references, modality profiles, purpose rules, statistics files, parameter columns, defaults, validation |
| [docs/followup_preprocessing.md](docs/followup_preprocessing.md) | Changes to morphobricks-preprocessing |
| [docs/followup_cellpatcher.md](docs/followup_cellpatcher.md) | Changes to CellPatcher |
| [AGENTS.md](AGENTS.md) | Rules for anyone changing this repository |

## Related repositories

- [morphobricks-preprocessing](https://github.com/CellMigrationLab/morphobricks-preprocessing)
- [CellPatcher](https://github.com/CellMigrationLab/CellPatcher)
