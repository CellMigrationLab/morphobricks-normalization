# Changelog

Release notes and migration notes of `morphobricks-normalization`. Numbers are pull requests and
issues of [CellMigrationLab/morphobricks-normalization](https://github.com/CellMigrationLab/morphobricks-normalization).
The decisions are in [docs/plan.md](docs/plan.md) §12.

## Unreleased

- **Pixel-free check of a conversion request** (issue #97, contract §3.4): new
  `stain.check_options(preset, *, white=None, stain_vectors=None, polarizer=None) -> None` makes,
  without pixels, every check `to_scalar` makes of the preset and these arguments (unknown preset,
  needed and unused arguments, polarizer, `white` range and masked values, stain-vector names,
  shape, finiteness, masked values, collinearity and conditioning), so consumers can refuse a
  request before reading an image. `to_scalar` runs it first (one implementation): exception types,
  messages and outputs are unchanged; an argument error is now reported before a pixel error when
  both are present.

## v0.6.0 (2026-11-13)

Additive release (issue #94, PR #95; plan N67-N69, contract §3.4): a new `stain` module turns one
declared imaging unit into one scalar map with a named preset; scikit-image is the new optional extra
`histology`. Nothing that existed changes: `rgb.py`, `NormalizationParameters`,
`plane_histograms.csv`, the CSBDeep parity path and every public name and output of v0.5.0 are
untouched, and the package still imports with NumPy only. Reviewed independently at `3a7af3e`,
`8cc45da` and `2506e51` (all findings fixed).

**Known limitations** (state them with results):
- RGB inputs are declared 8-bit sRGB only, three samples and no alpha; 16-bit, ICC profiles and
  scanner colour management are not handled.
- `he_custom` is a two-stain model (cross-product residual, declared white), not scikit-image's
  three-stain HED deconvolution; use `he_eosin` for the published conversion.
- The fixed HED/BPX deconvolution keeps scikit-image's floor and zero-clipping; `zero_fraction`
  cannot tell naturally zero components from clipped negatives.
- With a declared white, the luminance and CIELAB projections re-quantise to uint8 (clipping
  reported); `custom_od` floors zero intensity to 1 and keeps negative concentrations.
- The trichrome vectors are one Masson profile, not every protocol; a linear polarizer makes
  orientation unreliable; eosin-derived maps are not collagen-specific.
- Beer–Lambert on scanner sRGB values is an approximation; preset scaling hints are recorded, never
  applied.


- **RGB-to-scalar stain presets** (issue #94, plan N67, contract §3.4): new `stain` module with
  `rgb_to_scalar(pixels, *, axes, preset)` and the preset registry `PRESETS` (`he_eosin`,
  `he_hematoxylin`, `luminance_inverted`, `lab_a`). 8-bit sRGB, `axes` `"YXC"` or `"CYX"`.
  scikit-image is the new optional extra `histology`. `rgb.py`, the parameter columns and
  `plane_histograms.csv` are unchanged.
- **Preset catalogue** (issue #94, plan N68, contract §3.4): general entry point
  `to_scalar(pixels, *, axes, preset, white=None, stain_vectors=None, polarizer=None)`;
  `rgb_to_scalar` delegates to it and the four P0 presets give bitwise identical outputs. New
  presets `fluorescence`, `shg` (passthrough), `psr_brightfield`, `he_custom`,
  `trichrome_custom` (`custom_od`: optical density against a declared white, user stain vectors,
  NumPy only), `psr_polarized` (needs `polarizer`), `trichrome_blue`, `trichrome_green`.
  `StainPreset` gains `modality`, read-only consumer `hints`, `needs`, `inverted`,
  `target_stain` and `exclusion_stain`; the provenance gains `modality` and `hints`. New
  helpers `white_from_region`, `stain_vectors_from_regions`; new pure modules `optical_density`
  and `scalar_inputs`; `PRESETS_BY_MODALITY`.
- **`he_custom` stated as a two-stain model** (review of `8cc45da`): contract §3.4 says plainly that it is not the
  three-stain HED conversion (cross-product residual instead of DAB, log10 OD against the declared white, no
  clipping); the wrong "completion of `rgb_from_hed`" sentence is corrected; the uint8 re-quantisation of the
  declared-white path is documented (at most half a grey level per channel). New tests: the literal published
  vectors independent of scikit-image, `he_custom` on a reference pixel computed by hand, and its difference from
  scikit-image's HED eosin.
- **Review fixes (ChatGPT review of `3a7af3e`)**: a masked `white`, a masked stain-vector component (or a masked
  value inside a list) and a masked region are refused (`TypeError`), never unmasked by `np.asarray`;
  `stain_vectors_from_regions` refuses a reference region holding a pixel with a channel at 0 (its optical density
  would be taken at the floor), while zeros outside the regions are allowed; the condition-number limit is
  documented as a heuristic norm-wise threshold; the `psr_polarized` green-weighted greyscale bias is documented;
  the median-255 refusal of `white_from_region` says a declared white of 255 is still accepted.
- **CellProfiler-style image names** (issue #94, contract §3.4): each stain preset names its image
  (`StainPreset.image_name`, e.g. `he_eosin` -> `Eosin`, `trichrome_blue` -> `MethylBlue`), recorded in the
  provenance; consumers use it as the derived channel's name in feature columns.
- **Default stain vectors and declared white** (issue #94, plan N69, contract §3.4):
  `he_custom` (Ruifrok & Johnston H&E) and `trichrome_custom` (Methyl Blue / Ponceau Fuchsin of
  `rgb_from_bpx`) use published default vectors when `stain_vectors` is not given;
  `psr_brightfield` still needs them. New `StainPreset` fields `accepts`,
  `default_stain_vectors`, `default_stain_vectors_citation`; provenance `stain_vectors_source`,
  `stain_vectors_citation`. `trichrome_custom`'s exclusion stain is renamed `nuclei` ->
  `ponceau_fuchsin`. `luminance_inverted`, `lab_a`, `psr_polarized`, `trichrome_green` accept an
  optional `white` (rescale to `clip(rint(I * 255 / white), 0, 255)`, `white_clipped_fraction`
  reported); without it outputs are bitwise unchanged. The registry moves to `stain_presets.py`
  (re-exported by `stain.py`).

## v0.5.0 (2026-11-11)

Breaking release. Stage 3 S3.2 (MorphoBricks-Spec roadmap; ADR 0003, plan N61–N66, PR #90):
`plane_histograms.csv` schema version 5 and channels named by canonical channel ID everywhere,
plus the numerical-correctness and validation fixes of #91 and #92 merged since `v0.4.0`. See
[Migrating to v0.5.0](#migrating-to-v050).

- **`plane_histograms.csv` schema version 5** (HV1, HV2): one row per plane; the presence column
  `channels` (the plane's IDs in source order, `;`-separated) after the 15 plane columns; the
  per-channel columns of every channel ID of the experiment, sorted, named
  `<Category>_<Feature>_<ID>` (`Histogram_Count_DNA_00042`). An absent channel is blank and a value
  there is refused. `write_plane_histograms(records, path, *, channel_ids, failures=())` needs the
  experiment's channel IDs; new `read_plane_histogram_table` (header IDs and records),
  `table_header`, `header_channels`.
- **Failed measurements** (HV3): `ChannelMeasurementFailure`, `ChannelMeasurementError`,
  `check_measurements`; the writer writes nothing when given a failure, and removes a table an
  earlier run left at the path (`ChannelMeasurementError.removed` names it), so no out-of-date
  table looks current.
- **Version 4 is refused** (HV4) with a message naming version 5 and how to recompute; no converter.
- **Channels by ID**: `PlaneRecord(..., *, position_id, channels)` (required), `channel_histogram(id)`,
  `has_channel`; `PooledHistogram(..., channel=id)`; `fit(..., channels=)` (rows: which IDs, default
  all; pixels: the ID of each entry, required except RGB); `Parameters.channels` are sorted IDs
  (required), `select(ids)`, `apply(..., channels=)`, columns `Normalization_Offset_<ID>` /
  `Normalization_Scale_<ID>`; `ImageSummary.channels`, `summary_columns(channel_ids)`;
  `range_check_issues` takes channel IDs; `Issue.channel` is an ID. RGB uses the IDs `R`, `G`, `B`.
  An index where an ID is expected is refused.
- **Heterogeneous images** (N43 replaced): rows may have different channels, and each channel is
  pooled by its ID. A fitted group in which only some planes have a channel is refused with the
  new `ChannelScopeIncompleteError` (the Spec's `A_SCOPE_INCOMPLETE`, "present in k of n
  planes"), unless the user selects those planes with the new setting
  `NormalizationSettings(partial_channels="planes_with_channel")`, stored by `to_dict` when
  chosen and reported per fit by the new warning `channel_partially_present`. Planes of one image
  with different channel lists are refused.
- **Centred tail moments** (owner decision of 11 November 2026, plan N65; #91's question): the
  underflow and overflow columns are `Histogram_UnderflowCount_<ID>`,
  `Histogram_UnderflowMean_<ID>`, `Histogram_UnderflowStd_<ID>` (and `Overflow`), replacing the
  raw `...Sum_<ID>` and `...SumOfSquares_<ID>`. `HistogramTail(count, mean, std)` replaces
  `HistogramTail(count, sum, sum_of_squares)`. The new module `tail_moments.py` accumulates every
  tail pixel exactly (significand times a power of two, or the integer itself), so the mean and
  the standard deviation are each the float64 nearest the exact value: no cancellation for large
  values with a small spread, no overflow for finite pixels (no scale column needed), and the
  same bits whole or in blocks (they could differ in the last bits before). Pooling adds
  `count · std²` and `count · (mean − pooled mean)²`, and a pooled mean whose tail products
  overflow is redone with exact power-of-two scaling. This supersedes #91's refusal of an
  infinite tail sum of squares and of opposite infinite tail sums below: those tails now give
  their finite mean and standard deviation. Numbers without tails (the default min-max range)
  do not change; with tails, means, standard deviations and percentiles move by at most
  about `1.4e-14` relative in the `tools/dump_outputs.py` cases, and every tail mean of the
  `--corpus` is unchanged.
- **Parameter columns are versioned**: `to_columns` adds `normalization_parameters_version` =
  `2` (appended to `COLUMNS`), and `from_columns` refuses a row without it (the
  `Normalization_Offset_C<i>` columns of 0.4, which would otherwise read as channel `C0`) or of
  another version: fit again.
- `Parameters` moved to the new module `parameters.py` (re-exported by `fitting` and the package).
- Tests: the Spec's test material is vendored at `cc9820f` (`tests/spec_fixtures`, pin test and CI
  check); described-tree tests for the owner's extra-stain case and fixtures v04, v05, v06, v13,
  v17, v22.

Numbers: for planes whose channels are all present, histograms and fits are bit-identical to
schema version 4 (v0.4.0 and the #91/#92 fixes below) for the same pixels: a one-off comparison
of 3600 fitted channels on `v13` trees, the 400 golden fits, and `tools/dump_outputs.py
--corpus` unchanged. `tools/dump_outputs.py` changes only in the table digests (new format), the
added `normalization_parameters_version` column, and `Issue.channel` and messages naming channel
`C0` instead of index `0`.

Numerical-correctness fixes. Each says which outputs change; every other output is
bit-identical (`tools/dump_outputs.py` and `--corpus` lines, and the 400 golden fits).

- **64-bit integers float64 cannot hold (#28).** A bin's representative is the float64 nearest
  its midpoint, rounded once (it was rounded twice, `binning_min` first, which could miss a
  representable value by 2 above `2**53`). `max_representative_error` of an integer plane
  with a value of magnitude `2**51` or more is the exact largest distance between its
  integers and the float64 standing in for them, rounded up, so a histogram is exact only
  when every representative equals its value: `[2**53 + 1]` is no longer reported exact with
  error 0 (its representative is `2**53`; error 1), and `normalization_exact` and
  `Histogram_Exact_<ID>` are false for it. Only int64 and uint64 planes with such values
  change; bin counts and limits never do.

  **Who is affected, and by how much.** Only int64 and uint64 images with a pixel of
  magnitude `2**51` (about `2.25e15`) or more. Every other pixel type (8-, 16- and 32-bit
  integers, float32, float64), and 64-bit images whose values all stay below `2**51`, give
  bit-identical numbers. For the affected images:
  - representatives move by at most one float64 step (at most 1536 for int64 and 2048 for
    uint64 values near `9e18` and `1.8e19`, about `1e-16` relative);
  - `max_representative_error` is the true largest distance: for bins of one value float64
    cannot hold it grows from 0 to at most half a float64 step (for example 1 for
    `[2**53 + 1]`, 170 and 341 for the constant planes `3074457345618258602` and
    `6148914691236517205`), so `is_exact`, `normalization_exact` and
    `Histogram_Exact_<ID>` change from true to false; for wider bins it changes in its last
    bits only;
  - pooled means, standard deviations and percentiles, and fitted offsets and scales, move by
    at most about `3e-14` relative in the `tools/dump_outputs.py` cases (a zscore offset
    `9618469625711740` becomes `9618469625711942`, nearer the exact model mean
    `9618469625711938`).

  Statistics files need not be recomputed: they store counts and limits, which do not change,
  and representatives and bounds are derived when the file is read, so an existing
  `plane_histograms.csv` gives the new numbers. Parameters fitted with v0.4.0 from such
  images differ from a refit by the amounts above.
- **Standard deviations of large finite values (#33).** `PooledHistogram.std` and
  `WeightedHistogram.std` summed squared deviations that overflow for values beyond about
  `1e154` (`[-1e200, 1e200]` gave `inf`, so `zscore` was refused, although its standard
  deviation is `1e200`; a tail could raise a bare `OverflowError`). When the plain sum is not
  finite but the mean and every value are, it is redone with the values divided by a power of
  two, which is exact; every finite result is unchanged bit for bit. A tail whose stored sum
  of squares is infinite (schema version 4 keeps raw sums of squares) is refused by name
  instead of giving `inf`: its spread is not known. No dump output changes. Not changed: the
  raw sums of squares of the file lose precision by cancellation for tails of large values
  with a small spread; a centred representation needs a new schema version. (Done in this
  release by schema version 5's centred tail moments, N65, above; the named refusals of
  infinite tail sums are superseded by it.)
- **Missing conditions per time point (#41).** With `scope="biological_repeat"` and
  `per_time_point=True`, `repeat_missing_conditions` checked each repeat group over all its
  time points, not the groups that are fitted, so a group such as `r1::t=0` with control
  images only gave no warning when treated images of `r1` existed at another time point.
  The groups that are fitted are now checked, and the warning's items and message name them
  (`r1::t=0`). Without `per_time_point` the warning is unchanged. No dump output changes.
- **Exact percentiles in the written method name (#35).** `normalization_method` wrote each
  percentile with `:g`, six significant digits, so different settings shared a name
  (`percentile=(1.000001, 99.8)` and `(1.000002, 99.8)` both wrote `percentile 1-99.8`) and
  `Parameters.from_columns` read parameters fitted with one as the other's. Each percentile is
  now the shortest decimal that reads back as exactly the same float, never in exponent form
  (`NormalizationSettings.method_name`, `percentile_range` and the new `normalization.percentile_name`).
  Names of percentiles with at most six significant digits and at least `0.0001`, the default
  `percentile 1-99.8` among them, are unchanged; others change, for example
  `percentile 1-99.8` becomes `percentile 1.000001-99.8`, and `percentile 1e-05-50` becomes
  `percentile 0.00001-50`. **Not compatible:** a parameter row written before this change with
  such percentiles holds the old short name and is refused by `from_columns` with the existing
  message naming the row's and the settings' names ("Column normalization_method is
  'percentile 1-99.8', but the settings use 'percentile 1.000001-99.8'"); fit the parameters
  again (the owner's ruling of 10 November 2026: no backward compatibility before the first
  pipeline release). Fitted offsets and scales do not change, and no dump output changes.
- **Clip limits and result type of `NormalizationParameters.apply` (#39).** Reversed clip
  limits (`clip=(1, 0)`) gave every pixel the same value, and an integer `dtype` truncated
  and wrapped the normalized values, without a word. `apply` now refuses, with a
  `ValueError` naming the value, clip limits that are not two finite numbers with
  `low < high` (equal limits too), and every result type but float32 and float64. Valid
  calls give bit-identical results; no dump output changes.
- **RGB images of 32- and 64-bit integers are refused (#23).** `check_rgb` accepted every
  integer type although its message, the contract's, asks for 8- or 16-bit colour channels,
  and `apply_rgb` repeats CSBDeep's float32 arithmetic, which cannot hold every 32- or 64-bit
  value (above `2**24`). `check_rgb`, and with it `fit` of RGB pixels or plane rows, now
  refuses `int32`, `uint32`, `int64` and `uint64` with that message ("... but these images
  have 3 channels of type int32 ..."). `uint8`, `uint16`, `int8` and `int16` are accepted as
  before, with bit-identical results; no dump output changes. **Not compatible:** RGB
  images of 32- or 64-bit integers that were fitted before are refused; save them as 8- or
  16-bit, or choose image type `fluorescence`.

### Migrating to v0.5.0

For the consuming bricks (preprocessing, CellPatcher, MorphoFeature) and any other caller. Every
brick should refuse a schema version 5 table explicitly until it is updated. Tables and parameter
rows written by v0.4.0 are refused, never converted: recompute them.

**All callers**

- `PlaneRecord(..., *, position_id, channels)`: `channels` (the plane's canonical channel IDs, in
  the source order of its imaging unit's confirmed list) is required. Use
  `record.channel_histogram(id)` and `record.has_channel(id)`; an index where an ID is expected is
  refused.
- `HistogramTail(count, mean, std)` replaces `HistogramTail(count, sum, sum_of_squares)`;
  `.sum` and `.sum_of_squares` are gone. New module `tail_moments`.
- `fit(..., channels=)`: for rows, the IDs to fit (default every ID some row has); for a pixel
  array, the ID of each entry along `channel_axis` (required, except RGB, whose IDs are `R`, `G`,
  `B`). `PooledHistogram(..., channel=id)` and `pooled_statistics(..., channel=id)`.
- A channel that only some planes of a fitted group have raises the new
  `ChannelScopeIncompleteError` (`A_SCOPE_INCOMPLETE`, "present in k of n planes"; `.channels`
  maps each such ID to the planes that lack it). Fit only the planes with it when the user
  explicitly chooses `NormalizationSettings(partial_channels="planes_with_channel")`; that choice
  is stored by `to_dict`, and each fit then reports the `channel_partially_present` warning listing
  the planes. Offer it only as an explicit user choice.
- `Parameters.channels` are sorted IDs; `select(ids)`, `apply(..., channels=)`. Parameter columns
  are `Normalization_Offset_<ID>` / `Normalization_Scale_<ID>` plus
  `normalization_parameters_version` = `2` (`parameters.COLUMNS` has five entries); `from_columns` refuses a
  row without it or of another version (the `..._C<i>` columns of v0.4.0): fit again.
- `Issue.channel` is a channel ID; `summary_columns(channel_ids)`; `range_check_issues` takes
  channel IDs; `ImageSummary.channels`.
- New: `ChannelScopeIncompleteError`, `NormalizationSettings.partial_channels`,
  `ChannelMeasurementFailure`, `ChannelMeasurementError` (with `.failures` and `.removed`),
  `check_measurements`, `records.check_universe`, `read_plane_histogram_table`,
  `csv_io.table_header`, `csv_io.header_channels`; `Parameters` lives in the new module `parameters` (re-exported).

**preprocessing**

- Build `PlaneRecord(..., channels=)` from the unit's confirmed channel list.
- Call `write_plane_histograms(records, path, channel_ids=<the description's declared [channels]
  IDs>, failures=...)`, giving every parallel shard the same IDs so shards share one header.
- On `ChannelMeasurementError`, report `.failures` (plane and channel) and `.removed` (an
  earlier table removed from the path); nothing is written (HV3).
- In `RunInputs`, read with `read_plane_histogram_table` (header IDs and records). A schema
  version 4 table is refused with a message naming version 5 and saying to run preprocessing
  again (HV4).
- Drop the N43 refusals (images may have different channels); pass channel IDs to
  `range_check_issues`; show `ChannelScopeIncompleteError` as `A_SCOPE_INCOMPLETE`.
- The tail columns (`Histogram_{Underflow,Overflow}{Count,Mean,Std}_<ID>`) are written by the
  library; nothing else changes for them.

**CellPatcher**

- Use `record.channel_histogram(id)`, `fit(..., channels=)`, and `Parameters.apply` / `select` by
  ID. Patch columns are `Normalization_Offset_<ID>` / `Normalization_Scale_<ID>` plus
  `normalization_parameters_version`; old tables are refused. Handle
  `ChannelScopeIncompleteError`.

**MorphoFeature** (`histogram_fit`)

- Map catalogue names to `record.channels`; expect `ChannelScopeIncompleteError` unless the
  settings opt in with `partial_channels`.

**Numbers.** Planes whose channels are all present give the v0.4.0 histograms and fits bit for
bit (3600 fitted channels on `v13` trees, the 400 golden fits, `tools/dump_outputs.py --corpus`
apart from the stored tail fields). With tails (robust range), the stored tail fields change from
sums to centred moments, and pooled means, standard deviations, percentiles and fits move by at
most about `1.4e-14` relative; the #28 64-bit representative fix moves int64/uint64 numbers as
described above.

## v0.4.0 (2026-11-10)

Breaking release: `PlaneRecord.position_id` is required (NM-04) and the readers of old table layouts
are removed (NM-01). See [Migrating to v0.4.0](#migrating-to-v040).

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

Merged to `main` between `v0.3.4` and `v0.4.0`; from the commit history. Each is a
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

### Migrating to v0.4.0

Valid inputs give bit-identical results (`tools/dump_outputs.py` and `--corpus` digests are
unchanged, and the 400 golden fits are bit-identical). Two inputs that were accepted before are
now refused:

- **`PlaneRecord` needs `position_id`.** It is required and keyword-only. The preprocessing brick
  (and any other caller that builds `PlaneRecord`s) must pass the position of the plane's field of
  view; `None`, blank or non-text values raise `ValueError`. Nothing is filled in or guessed.
  `read_plane_histograms` refuses a row whose `position_id` is blank with a `CsvFormatError`
  (it used to read it as `None`). A `plane_histograms.csv` written before the field was filled
  must be recomputed by running preprocessing again.
- **Old-version tables are refused, not read.** A `plane_histograms.csv` of schema version 1, 2 or
  3 is recognised by the `schema_version` of its rows and refused with the same pointed messages as
  before; each names the version and says to run the preprocessing brick again to write schema
  version 4. A version 1 to 3 header with no rows is refused as a table with unexpected columns.
  Only schema version 4 is read.

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
