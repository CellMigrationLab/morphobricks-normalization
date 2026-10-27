# Follow-up Plan — CellPatcher

> See also: [Normalization brick plan](plan.md) · [Preprocessing follow-up](followup_preprocessing.md)

**Status:** plan, revised 27 October 2026; starts after the normalization brick's
Milestones 2b–4 (released as `v0.2.0`) and preprocessing P3b–P4 (N6); C5 does not depend
on them. Decision numbers N refer to §12 of the
[normalization plan](plan.md); D-numbers refer to CellPatcher's own `docs/plan.md`, and this
plan revises some of them (see the last section).
**Repository:** [CellMigrationLab/CellPatcher](https://github.com/CellMigrationLab/CellPatcher)

## Goal

CellPatcher keeps patching as it is and normalizes with three scopes (`biological_repeat`
across conditions, `image`, `patch`) and two image types (`fluorescence`, `rgb`), using the
normalization brick's settings and fitting. The image and patch scopes behave exactly as
today (N29).

## Today

- `NormalizationConfig` offers `condition`, `biological_repeat`, `technical_repeat`, `image`
  and `patch` scopes, the four per-channel methods, always per channel, with parameters
  computed in `cellpatcher/normalization.py` from the brick's `group_records` and pooling.
- The `preprocessing` extra pins `morphobricks-preprocessing[zarr]` to commit `1129776` and
  the normalization brick to `v0.1.0`; plane records, the CSV reader and pooling already come
  from `morphobricks_normalization`, image readers and manifest checks from
  `morphobricks_preprocessing`.
- Today's `biological_repeat` group is `<condition>/<biological_repeat>`: one group per
  condition.

## C1 — Use the brick's settings and fitting (no behaviour change)

- Raise the pins to normalization `v0.2.0` and the preprocessing release with P3–P4.
- Replace CellPatcher's own parameter code with the brick's `fit`, `apply`, `to_columns` and
  `from_columns` for the `image` and `patch` scopes and the four fluorescence methods.
- Read `plane_histograms.csv` schema version 3 (channels in columns, N41); a table of version
  2 is refused with the brick's message asking to rerun the preprocessing brick.
- **Parameter columns (N41, N42).** `patches.csv` replaces the `;`-separated
  `normalization_offset` and `normalization_scale` with `Normalization_Offset_C<i>` and
  `Normalization_Scale_C<i>`, one pair per channel of the image, grouped by channel. A
  `split_channels` patch fills only the pair of its own channel. `normalization_method`,
  `_scope`, `_group` and `_exact` keep their names. Normalizing raw patches on load
  (`iter_exported_patches`, `ExportedPatchTorchDataset`) reads the new columns and keeps
  reading the old `;`-separated ones of earlier tables; the brick's `from_columns` reads
  only the new columns, so CellPatcher converts old rows itself (N47).
- **Channel count.** Images with different numbers of channels are an error when
  normalization is enabled (N43).
- Store `NormalizationSettings.to_dict()` in `run_config.json`.
- Acceptance: identical patches and `run_config.json` (apart from the added settings)
  before and after, for every delivery mode; `patches.csv` identical apart from the
  parameter columns, whose values are unchanged.

## C2 — Three scopes

- `NORMALIZATION_SCOPES = ("biological_repeat", "image", "patch")`. The `condition` and
  `technical_repeat` scopes are removed straight away, without a deprecation period (N29): a
  configuration or older `run_config.json` naming them is refused with the brick's message
  listing the three scopes. Raw patches exported earlier still normalize on load, because
  their parameters are in `patches.csv`.
- `biological_repeat` pools one repeat group across all conditions, technical repeats merged
  (N30): the group key comes from the brick's `group_key` with the repeat groups read from
  `experiment_design.csv`. `normalization_group` is the repeat group label (previously
  `<condition>/<biological_repeat>`).
- New setting `normalization.experiment_design`: path of `experiment_design.csv`; None uses
  the one next to `plane_histograms.csv`. With `biological_repeat`:
  - a missing file, or a repeat group not confirmed, is an error before any image is read,
    asking to run the preprocessing brick's step 1;
  - a file whose SHA-256 differs from the one in the preprocessing manifest is an error;
  - an image whose condition and biological repeat are not in the file is an error.
- New setting `normalization.pooling`: `"pixels"` (default) or `"conditions"` (N32), used
  only by `biological_repeat`.
- The repeat groups missing some conditions give the brick's warning.
- `run_config.json` records the design file's name and SHA-256 with the other normalization
  inputs.

## C3 — RGB images

- New setting `normalization.image_type`: `"fluorescence"` (default) or `"rgb"`. With a
  preprocessing manifest, the type comes from it and a different value in the configuration
  is an error; without one, the configuration decides (N34).
- `rgb` uses only the StarDist percentile method (N33); another method is refused before any
  image is read. Default percentiles 1 and 99.8.
- `rgb` needs three channels kept together: `extraction.split_channels` other than `False`
  with `rgb` is an error (C5).
- At every scope, one offset and scale come from the R, G and B pixels pooled together;
  `normalization_offset` and `normalization_scale` hold the shared value three times.
- At the `patch` scope, the patch region (`mask`, `bbox`, `patch`) selects the pixels of all
  three channels.

## C4 — Notebook

- The normalization step offers the three scopes, with a line explaining that
  `biological_repeat` keeps differences between conditions and removes differences between
  repeats; the pooling choice appears only for that scope.
- The image type is shown from the manifest (or chosen when there is none); for RGB only the
  percentile method is offered.

## Tests and acceptance

- C1: identical outputs with the brick's fitting.
- C2: `biological_repeat` pools the same repeat across conditions and technical repeats;
  pairing in `experiment_design.csv` changes the groups; an unconfirmed or missing design
  stops the run before any image is read; removed scopes are refused with the new message;
  both pooling weights agree when conditions have equal pixel counts.
- C3: RGB patches get one shared offset and scale; the result matches
  `csbdeep.utils.normalize(patch, 1, 99.8, axis=(0, 1, 2))` at the patch scope (N40);
  an RGB patch without spread becomes zeros and is not discarded; `split_channels` with RGB
  is refused.
- C5: a channel selection gives one single-channel patch and row per selected channel;
  `True` and `False` give exactly today's patches; an index outside the images' channels is
  refused before reading.
- Every scope and type gives the same arrays in all delivery modes (normalized files, raw
  files normalized on load, eager and lazy in memory).

## C5 — Channel selection for single-channel patches

For deep-learning training, which often needs one channel per patch (N45). Feature
extraction keeps multichannel patches, with per-channel values in columns (N41).

- `extraction.split_channels` also accepts a list of channel indices: one single-channel
  patch per selected channel of each region. `True` keeps meaning every channel, and `False`
  (default) one multichannel patch, so existing configurations work unchanged.
- Each selected channel's patch is its own `patches.csv` row with the channel in `c`, as
  today; `position_id` is shared by the channel patches of one image, as today. An index
  that is not a channel of the images is an error before any image is read, naming the
  channel count.
- Normalization parameters stay per channel of the image (`Normalization_Offset_C<i>`,
  `Normalization_Scale_C<i>`); a single-channel patch fills only its channel's pair, from
  the group's `Parameters.select([i])` (N46).
- `rgb` images keep their three channels together, so any channel splitting with `rgb` is
  an error (C3).
- **Notebook:** a "Channels" choice with one checkbox per channel (`C0`, `C1`, …, with the
  channel names from the metadata when known): all ticked with "one patch per channel" gives
  every channel, as before; ticking some gives only those.

## Revisions to CellPatcher's earlier decisions

| CellPatcher decision | Revision |
|---|---|
| D4: channels are never pooled | Except for `rgb`, where R, G and B share one offset and scale (C3) |
| D5: scopes condition to patch | Three scopes; `biological_repeat` now spans all conditions (C2) |
| D12: deprecate before removing | Not for the removed scopes: removed straight away (N29) |
| D14 and contract §10: `normalization_offset`/`normalization_scale` as `;`-separated lists | One `Normalization_Offset_C<i>`/`Normalization_Scale_C<i>` pair per channel (C1, N41) |
| Contract §7: scope table | Rewritten for the three scopes and repeat groups (C2) |

These revisions are owner-approved through decisions N29–N45 and must be recorded in
CellPatcher's `docs/plan.md`, `docs/normalization_contract.md`, `docs/configuration_reference.md`
and `AGENTS.md` §5 when C1–C4 are implemented.
