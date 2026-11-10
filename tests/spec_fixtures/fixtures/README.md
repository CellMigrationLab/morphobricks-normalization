# Fixture catalogue: experiment description v1

Normative test data for [experiment-v1.md](../docs/experiment-v1.md). Each file starts with a
comment header (`Fixture`, `Expected`, `Source`, description). How these become executable tests
in preprocessing is described in the [conformance plan](../docs/conformance-plan.md); the audit tool
`tools/audit/refval.py` runs them here. The records files in `records/` have no header; see
their section.

## Rules

- A conforming reader MUST accept every file in `valid/` and report exactly the warnings listed
  (none unless stated).
- For every file in `invalid/` it MUST report exactly the listed **set** of error codes, all from
  the first failing validation phase (experiment-v1 §6), and nothing from later phases.
- "Homogeneous: **no**" marks a valid file whose units have different ordered channel lists. The
  spec accepts it; the Stage 1-2 runtime MUST refuse to process it with
  `R_HETEROGENEOUS_LAYOUT_NOT_ENABLED` (lifted at Stage 3).
- Physical cases that need images (within-unit divergence, header contradictions, mask pixel
  contradictions, reconciliation) cannot be expressed by a TOML file alone; they are listed as
  scenarios P01-P35 in the [conformance plan](../docs/conformance-plan.md#3-physical-scenarios-preprocessing-owned),
  built on the valid files below.
- `v18` MUST parse to the same model as `v01` (round trip); `v21` differs from `v01` only in
  `display_name`, `target` and `treatment_key` values and `v16` only in `display_name` values, so
  for both all four section digests equal `v01`'s (experiment-v1 §9.3).
- Every file carries the required `used_ids` table (experiment-v1 §4.9) listing at least all of its
  declared condition, channel, repeat and technical repeat IDs; `i92` is the file without it.
- Every condition and every technical repeat carries the required `source_folder` (experiment-v1
  §4.6a): the exact folder label, never restricted by the reserved or forbidden names. In
  fixtures that do not exercise the rule it equals the ID (the key), `single` for `single`;
  `v33`, `v34` and `t10`-`t12` separate the two. The eight undecodable
  or unparsable phase-1 fixtures (`i01`-`i08`) are unchanged and fail before it matters.
- `transitions/` holds thirteen files that are valid alone and are checked against a named published
  predecessor: refused with an exact set of publication codes (`R_CONFIRMED_ID_CHANGED`,
  `R_USED_ID_REUSED`, `R_USED_ID_DROPPED`, `R_CONFIRMED_ON_CHANGED`; experiment-v1 §4.4a, §4.6,
  §4.9; a changed or reassigned `source_folder` counts, §4.6a) or, for `t06`, **accepted** (adding members is not a changed entity, §4.6). They need the
  predecessor file and are not part of the single-file valid/invalid rules. Publication-base and
  folder-name rules need no TOML of their own: they run on named fixtures in scenarios P17-P22.
- Rules that need images, a run or a consumer (header matching, image type against the header,
  mask checks and conversions, mask type names, the FOV key, the `target` prefill, derived-CSV
  dates) have no single-file TOML form: each is a scenario P23-P33 of the conformance
  plan on a named fixture, listed in the coverage tables below.
- A copy of this tree vendored by a reader MUST hash to the pinned SHA's `fixtures` tree
  (conformance plan §1).
- Expected `experiment_design.csv` outputs (golden files, experiment-v1 §11) are in `expected/`,
  one per file listed under "Expected outputs" below; the other valid files have none yet. The fixed-SHA
  consumer audit (roadmap S0.1) is done (PR #5); its questions are answered by the third round of
  owner decisions (ADR 0001) and the fixtures `v28`-`v32`, `t06`-`t09`, `i100`-`i119` follow from
  them; the fourth round (FR1, FR2) adds `v33`, `v34`, `t10`-`t12` and `i120`-`i127`.

## Syntax check (6 November 2026)

Checked with Python 3.13 `tomllib` (`tomllib.loads` on the UTF-8 decoded bytes), after the third
round of owner decisions and again after the fourth (folder labels, `source_folder`):

- `valid/`: 34 of 34 parse.
- `transitions/`: 12 of 12 parse.
- `invalid/`: 119 of 127 parse as TOML and are invalid only by the v1 rules. The 8 phase-1 fixtures
  fail deliberately: `i01_bom` (`TOMLDecodeError`, BOM read as an invalid statement),
  `i02_invalid_utf8` (`UnicodeDecodeError`), `i03`-`i06` (`TOMLDecodeError`, "Cannot declare ...
  twice" / "Cannot overwrite a value"), `i07`, `i08` (`TOMLDecodeError`, "Invalid value" /
  "Expected newline").

## Matrix audit fixtures and executable validator (7 November 2026)

The matrix audit ([`docs/audits/matrix-audit-2026-11-07.md`](../docs/audits/matrix-audit-2026-11-07.md))
adds `v35`, `v36` and `i128`-`i130` for structures the catalogue lacked (a flat experiment, a
biological-repeat folder named like a forbidden key, `single` rules for condition folders and
members). The owner decisions of 7 November 2026 (ADR 0001, SQ6) add `i131`-`i133` (the
`single` rules for conditions) and `i134` (a duplicate key inside an inline table); the owner decision of 7 November 2026 on reserved names in `used_ids` (UR1) adds `v37` and `i135`. Its audit-only validator, [`tools/audit/refval.py`](../tools/audit/refval.py)
(`python -I tools/audit/refval.py fixtures .`), accepts 37 of 37 valid files with exactly the
listed warnings, refuses 135 of 135 invalid files with exactly the listed code set from the listed
phase, gives the listed `R_*` set for 12 of 12 transitions, and round-trips every valid and
transition file through the canonical writer. It is an audit tool, not a reader: the conformance
plan's single reference reader stays preprocessing.

## Reference-validator cross-check (6 November 2026, roadmap S0.1)

A reference validator (Python 3.13; now the audit tool [`tools/audit/refval.py`](../tools/audit/refval.py), not a runtime reader) implementing experiment-v1 §2-§6 phase by phase, plus the predecessor checks of §4.6,
§4.4a and §4.9 expressible from two files, was run on every fixture after the third round of
owner decisions and run again after the fourth (folder labels, `source_folder`). The validator
implements the text as written, including the new rules (`source_folder`, one folder one ID,
`used_ids.technical_repeats`, one code per broken rule, forbidden names on ID keys, the extended
reserved list, NFC of channel references, `image_type` and the RGB rules, "changed entity" as
removal or reassignment, a changed `source_folder` included). Results:

- `valid/`: 34 of 34 accepted with exactly the listed warnings.
- `invalid/`: 127 of 127 refused with exactly the listed code set, from the listed phase only.
- `transitions/`: 12 of 12 valid alone (`t03`, `t06`, `t07`, `t08` with
  `W_REPEAT_MISSING_CONDITIONS`); against their predecessors each gives exactly its listed set:
  `t01` `R_CONFIRMED_ID_CHANGED`; `t02` `R_CONFIRMED_ID_CHANGED` and `R_USED_ID_DROPPED`; `t03`
  `R_USED_ID_REUSED` only; `t04` `R_USED_ID_DROPPED`; `t05` `R_CONFIRMED_ON_CHANGED`; `t06` no
  code (accepted); `t07`, `t08`, `t09`, `t10`, `t11`, `t12` `R_CONFIRMED_ID_CHANGED`.
- `v18` parses to the same model as `v01`; `v16` and `v21` have the same four sections as `v01`
  once `display_name`, `target` and `treatment_key` are omitted.

This is a check of the fixtures against the text, not a conformance claim for any brick.

## Coverage of RFC #1 and its discussion

**RFC #1, M0 fixture list (12 cases)**, from the
[RFC body](https://github.com/CellMigrationLab/MorphoBricks-Spec/issues/1):

| # | RFC case | Fixtures |
|---|---|---|
| 1 | One control | `v01` |
| 2 | Several controls | `v02` |
| 3 | Differently named source folders paired into one canonical repeat | `v03` |
| 4 | Swapped channel order between imaging units | `v04` |
| 5 | Variable channel counts | `v05` |
| 6 | Two technical repeats of one biological repeat with different layouts | `v06` |
| 7 | One FOV diverging inside a technical repeat (invalid) | `i20` (no override can express it); physical P01, P02 |
| 8 | Instance masks | `v07` |
| 9 | Binary masks | `v08` |
| 10 | Incomplete/progressive masks | `v09` |
| 11 | Reference chains | `v10` |
| 12 | Invalid reference cycles | `i64`, `i65` |

**Ten adversarial tests** from the independent review comment in the
[RFC #1 discussion](https://github.com/CellMigrationLab/MorphoBricks-Spec/issues/1#issuecomment-6667477114)
(5 November 2026). Most are end-to-end obligations of later stages; the fixture fixes the input:

| # | Adversarial test | Fixtures | Obligation |
|---|---|---|---|
| 1 | Two technical repeats of different counts/orders stay ONE biological replicate | `v06` | Stage 2-3: N per repeat in normalization, embeddings, MorphoCoverage |
| 2 | One deviating FOV in a technical repeat refused with image/C-index evidence | `i20`; P01, P02 | Stage 1: `R_UNIT_NOT_HOMOGENEOUS` |
| 3 | Swapped C order never mixes ACTIN and DNA in normalization or CellPatcher | `v04`, `v06` | Stage 1 refusal; Stage 3 pooling by canonical ID |
| 4 | A homogeneous described dataset (folder-name IDs confirmed by the user) keeps the existing refusals; no parity against v0.4.0 (ADR 0002 AR10) | `v13` | Stage 1: byte-identical derived CSV (as reviewed for v0.5.0); no legacy numerical parity gate |
| 5 | `DNA` with confirmed header `DAPI` passes; unconfirmed contradiction refuses | `v11`; P03 | Stage 1: `R_HEADER_CONTRADICTION` |
| 6 | A binary region mask never becomes an instance object set | `v08` | Stage 2: `A_MASK_NOT_INSTANCES` |
| 7 | N35 confirmation survives reconciliation and reopens on discovery changes (the legacy-CSV migration is removed, ADR 0002 AR10 (9)) | `v13`, `i37`; P07 | Stage 1 reconciliation |
| 8 | Distinct source repeat names paired by the user give one canonical ID in MorphoCoverage | `v03` | Stage 2 (S2.5) |
| 9 | Missing channels in requested scope: refusal or recorded user subset, never silent drop; a channel no image has is the same refusal (SI1) | `v05`, `v17` | `A_SCOPE_INCOMPLETE`, shown at Stage 3 / gate G3 for both: their layouts are refused at Stages 1-2 (P04); owner decision GS2 for both |
| 10 | Metadata-only changes keep valid pixel-derived caches | `v01` vs `v21` | Stage 1: section digests |

**Further required cases:**

| Topic | Fixtures |
|---|---|
| Duplicate keys | `i03`-`i06`, `i134` (inline table) |
| Case collisions | `i48`-`i52`, `i68` (no folding when resolving), `i95` |
| NFC (refused, never normalized) | `v23`, `i78`, `i83`-`i85`; P21 |
| Display names | `v16`, `v24`, `i53`, `i76`, `i77`, `i86`-`i88` |
| Display name is not an ID | `i89`-`i91` |
| IDs never repaired, re-cased or renumbered | `v26`, `i48`-`i52`, `i55`, `i95` |
| `used_ids` registry | `v25`, `i92`-`i97`, `t03`, `t04`; P19, P20 |
| `confirmed_on` is a user fact | `i98`, `t05`; P07, P15, P22 |
| Publish only from the current base | P17, P18 (on `v01`, `v21`, `v16`) |
| Lab-tool provenance stays out of the file | `i99` |
| Unknown and forbidden fields | `i16`-`i19`; `i20`-`i26`; `i97`, `i99` |
| Implicit-conversion traps | `i07`, `i08`, `i12`-`i14`, `i19`, `i27`-`i30`, `i33`, `i34`, `i36`, `i38`, `i39`; valid side `v15` |
| Reference DAG | `i64`-`i68`, `i41`; `v10`, `v19` |
| Swapped channel order | `v04`, `v06` |
| Extra / missing channel | `v05`, `v17` (both Stage 3 / G3, GS2: P04 refuses their layouts at Stages 1-2), `v22`, `i69` |
| Within-unit divergence | `i20`; P01, P02 |
| Instance vs binary masks | `v07`, `v08`, `i42`, `i43`, `i111`, `i112`; P05, P27, P28 |
| Partial masks | `v09` |
| Immutable IDs (no renames) | `v16`, `i75`, `i82`, `t01`, `t02`; P08, P16 |
| Adding is not a change; removing or reassigning is | `t03`, `t06`, `t07`, `t08`; P07, P15 |
| Case-only correction needs a new ID; set of publication codes | `t02`, `t09`, `i95`; P16 |
| RGB images (declared type, fixed tokens, no aliases) | `v28`, `v29`, `i100`-`i106`, `i119`; P13, P23, P24 |
| Reserved names (one list for channel and mask IDs; also for `used_ids.channels`) | `i59`, `i61`, `i107`-`i109`, `i135`, `v32`, `v37` |
| Forbidden names apply to ID keys | `i113`-`i115` |
| Folder labels never restricted: condition and technical-repeat IDs separate from `source_folder` | `v33`, `v34`, `i113`, `i115`, `i120`-`i127`, `t10`-`t12`; P32, P33 |
| One folder, one ID; `single` iff no folder level (technical repeats: `i125`; conditions: `i131`-`i133`) | `i122`-`i125`, `i131`-`i133` |
| Folder change for an existing ID is a reassignment | `t10`-`t12`; P33 |
| One code per broken rule; NFC of channel references | `i116`-`i118` |
| FOV key keeps the source label, `repeat_group` added | `v03`, `v30`; P25 |
| Mask checks: when, unchecked masks, type names | `v09`; P05, P28, P29 |
| Descriptive keys prefill only | `v27`; P31 |
| Derived CSV dates (every row carries its repeat's one `confirmed_on`; no clock read) | `v13`; P30 |
| Derived CSV keeps the source folder labels | `v13`, `v33`; P32 |
| Aliases (header evidence) | `v11`, `v12`, `v31`, `i46`, `i81`, `i110`; P03, P26 |
| Versioning (offline constant, no default) | `i09`-`i15` |
| Identities | `v14`, `v15`, `v20`, `v23`, `v26`, `i54`-`i63`, `i78` |
| Target / treatment key (descriptive only) | `v01`, `v10`, `v21`, `v27`, `i71` |
| Encoding | `i01`, `i02` |

### Owner decisions of 6 November 2026 (ADR 0001) and their fixtures

| # | Decision | Fixtures / scenarios |
|---|---|---|
| D1 | Optional `target`, `treatment_key` (refined by Q4: descriptive only) | `v01`, `v10`, `v21`, `v27`; `i71` (non-string) |
| D2 | Channel ID = compartment name refused | `i61` |
| D3 | Unused channel is a warning | `v22` |
| D4 | Repeat ID = another repeat's source folder refused | `i75` |
| D5 | IDs immutable; `display_name` mutable and unique; no `retired_repeats` | `v16` (display names change, identity identical), `i53`, `i76`, `i77` (collisions), `i82` (no table), `t01`, `t02`; P16 |
| D6 | N35 reopens only on channel/layout/grouping change | P07, P15 |
| D7 | NFC identities (refined by Q3: non-NFC refused, never normalized; no NFC-collision code) | `v23`, `i78`, `i83`-`i85` |
| D8 | RGB refused (**superseded by AQ6**) | P13 (now: RGB accepted when declared) |
| D9 | Header names exact, case-sensitive | `v11`; P03 |
| D10 | `binary` mask with > 2 values refused | P05 |
| D11 | Drafts internal; `RunInputs` refuses them | `i37`; P14 |
| Q1 | Append-only `used_ids` per kind; no reuse, no drop | `v25`, `i92`-`i97`, `t03`, `t04`; P19, P20 |
| Q2 | Display names unique per kind after NFC + case folding | `v24`, `i86`-`i88` |
| Q3 | Reader refuses non-NFC identities | `v23`, `i78`, `i83`-`i85`; P21 |
| Q4 | `treatment_key`, `target` descriptive only | `v27`, `v21` |
| Q5 | Lab-tool links in the run manifest only | `i99` |
| A1 | Publish only from the current base SHA | P17, P18 |
| A2 | IDs never repaired, re-cased or renumbered | `v26`, `i95` |
| A3 | `confirmed_on` is a user fact | `i98`, `t05`; P22 |
| A4 | `display_name` never substitutes for an ID | `i89`-`i91` |
| A5 | Pin integrity, offline version check, no fallback rules | `i09`-`i11`; conformance plan §1-§2 |
| AQ1 | Adding a member is allowed; only removing or reassigning is a changed entity | `t03`, `t06`, `t07`, `t08`; P07, P15 |
| AQ2 | Case-only ID correction needs a new ID; publication reports every `R_*` code | `t02`, `t09`, `i95`; P16 |
| AQ3 | `target` descriptive; MorphoCoverage may only prefill, confirmed inside MorphoCoverage | `v27`; P31 |
| AQ4 | One reserved list: compartments plus MorphoRelate reserved column names | `i59`, `i61`, `i107`-`i109`, `v32` |
| UR1 | The reserved list also applies to `used_ids.channels` entries (they reserve those IDs) | `i135`, `v37` |
| AQ5 | Canonical repeat ID added as `repeat_group`; the FOV key keeps the source label | `v03`, `v30`; P25 |
| AQ6 | RGB accepted as a declared type (`image_type = "rgb"`, tokens `R`, `G`, `B`); supersedes D8 | `v28`, `v29`, `i100`-`i106`, `i119`; P13, P23, P24 |
| AQ7 | No trimming: header names exact after NFC of both sides | `v31`, `i110`; P26 |
| AQ8 | `parent_labels` masks out of scope in v1 | `i111` |
| AQ9 | Binary-as-instances only by an explicit, per-request, recorded conversion | `v08`, `i112`; P27 |
| AQ10 | Mask representation checked in the mask step; publication records "not yet checked"; unchecked masks refuse | `v09`; P05, P28 |
| AQ11 | A discovered mask type breaking the Label rules: `R_MASK_TYPE_NAME_INVALID` | P29 |
| AQ12 | Legacy groups with different `confirmed_on` take the latest date, with a note. **Superseded 10 November 2026** (ADR 0002 AR10 (9): no legacy-CSV migration) | none (P30 now tests the derived output only) |
| AQ13 | Forbidden names apply to ID keys; NFC of channel references; one code per broken rule | `i113`-`i118` |
| FR1 | Folder labels are never restricted by the reserved or forbidden names: conditions and technical repeats have a canonical ID separate from the required `source_folder`; one folder, one ID; a folder change for an existing ID is a reassignment; the derived CSV keeps the folder labels | `v33`, `v34`, `i113`, `i115`, `i120`-`i127`, `t10`-`t12`; P32, P33 |
| FR2 | Derived CSV byte-identity is guaranteed only for repeat groups with a single `confirmed_on`; a legacy group with several dates gets the latest, the migration note keeps the per-row dates. **Superseded 10 November 2026** (ADR 0002 AR10 (9): no legacy-CSV migration; every published repeat has one `confirmed_on`) | none (P30 now tests the derived output only) |
| RR1 | Record type `image_count_changed` (a confirmed member's image count changes; nothing reopens) | P34 |
| RR3 | `image_count_changed` is written for every confirmed member whose count changed, also when the publication changes other things | P34 |
| RR2 | Record type `removed` for every confirmed removal of a unit, technical repeat, repeat or condition | P35 |
| RF1 | Records file separators are `, ` and `: `; other whitespace refused | `ri25`, `ri27`, `ri30` |
| RF2 | Every records-file violation is a refusal (key set, order, escapes, duplicates, values) | `ri01`-`ri24` |
| RF3 | One code, `R_RECORDS_FILE_INVALID`, with `E_RECORDS_*` reasons | all `ri` records files |
| RF4 | Empty records file accepted; BOM and CR before LF refused; lower-case SHA hex | `rv06`, `ri13`, `ri24`, `ri26`, `ri31` |
| RF5 | `technical_repeats` not empty; `old_image_count` differs from `new_image_count` | `ri28`, `ri29` |
| RF6 | Line order within one publication stated; checked by writer tests, not by the file | P34, P35 |

## Valid fixtures (37)

| Fixture | Source | Expected | Homogeneous | Case |
|---|---|---|---|---|
| [`v01_single_control`](valid/v01_single_control.toml) | RFC case 1 | pass | yes | One control; trivial pairing, so canonical repeat IDs default to the folder names. |
| [`v02_multiple_controls`](valid/v02_multiple_controls.toml) | RFC case 2 | pass | yes | Two controls (DMSO, WT), each referenced by two conditions. |
| [`v03_paired_different_folder_names`](valid/v03_paired_different_folder_names.toml) | RFC case 3; adversarial 8 | pass | yes | Source folders rep1 (CTRL) and experiment_23 (DrugA) paired by the user into canonical repeat Day1; rep2/rep2_drug into Day2. MorphoCoverage must see Day1 and Day2 as the biological repeats: the FOV key keeps `rep1` / `experiment_23` in `biological_repeat` and the added `repeat_group` is Day1 (P25). |
| [`v04_swapped_channel_order`](valid/v04_swapped_channel_order.toml) | RFC case 4; adversarial 3 | pass | **no** | Same channels, opposite source order between conditions (CTRL C0=DNA, DrugA C0=ACTIN). Valid schema; heterogeneous, so Stage 1-2 runtime refuses (R_HETEROGENEOUS_LAYOUT_NOT_ENABLED). From Stage 3, ACTIN and DNA are never mixed by C index. |
| [`v05_variable_channel_counts`](valid/v05_variable_channel_counts.toml) | RFC case 5 | pass | **no** | Extra channel PHASE in DrugA/rep1 only; CTRL lacks it (missing elsewhere). Heterogeneous. |
| [`v06_technical_repeats_different_layouts`](valid/v06_technical_repeats_different_layouts.toml) | RFC case 6; adversarial 1 | pass | **no** | One biological repeat (R002) whose CTRL member has two technical repeats with different counts and orders. It stays ONE biological repeat (N = 2 repeats in total, not 3). Heterogeneous. |
| [`v07_instance_masks`](valid/v07_instance_masks.toml) | RFC case 8 | pass | yes | Two instance masks with source-channel provenance; canonical mask IDs differ from discovered source types. |
| [`v08_binary_masks`](valid/v08_binary_masks.toml) | RFC case 9; adversarial 6 | pass | yes | A binary region mask beside an instance mask. Valid; an analysis request using TumorRegion as an object set must refuse (A_MASK_NOT_INSTANCES) unless it carries an explicit, per-request, recorded conversion (P27). |
| [`v09_partial_masks`](valid/v09_partial_masks.toml) | RFC case 10 | pass | yes | Masks declared while segmentation is in progress (e.g. 0/1000 or 300/1000 FOVs). Completeness lives in the run manifest, so the file is valid at every stage; scope checks belong to the analysis layer (A_SCOPE_INCOMPLETE). |
| [`v10_reference_chain`](valid/v10_reference_chain.toml) | RFC case 11 | pass | yes | Reference chains VEC_SiCol6a1 -> VEC_SiNC -> VEC and THBS5_SiCol6a1 -> THBS5_SiNC -> THBS5; shared treatment_key across backgrounds. |
| [`v11_header_aliases`](valid/v11_header_aliases.toml) | Adversarial 5 | pass | yes | Canonical DNA confirmed with header evidence DAPI and Hoechst 33342; canonical ACTIN with an exact-match header name that still had to be confirmed. A header name outside these lists is R_HEADER_CONTRADICTION. |
| [`v12_unnamed_headers`](valid/v12_unnamed_headers.toml) | Extra: aliases | pass | yes | No header evidence at all (ImageJ TIFFs carry no channel names); the user's confirmation is the only guard. Undetectable physical swaps are a documented limit. |
| [`v13_homogeneous_legacy`](valid/v13_homogeneous_legacy.toml) | Adversarial 4 | pass | yes | A legacy homogeneous dataset: folder-name repeat IDs, one control, same layout everywhere (IDs proposed equal to the folder labels). The derived experiment_design.csv, which carries the `source_folder` labels, must be byte-identical to its golden file (a Stage 1 check); there is no numerical parity against v0.4.0 (ADR 0002 AR10). |
| [`v14_no_repeat_folders`](valid/v14_no_repeat_folders.toml) | Extra: single convention | pass | yes | No biological-repeat or technical-repeat folders: the source label is `single` (N31, one group for the whole experiment). |
| [`v15_toml_keys_are_text`](valid/v15_toml_keys_are_text.toml) | Extra: implicit-conversion traps (valid side) | pass | yes | Folder labels that YAML would retype (2026-01-01, 012, NO, 1.0, off) stay exact text in TOML keys and strings. The quoted key "1.0" is needed: a bare 1.0 would be a dotted key. |
| [`v16_display_name_change`](valid/v16_display_name_change.toml) | Owner refinement: immutable IDs | pass | yes | v01 with every `display_name` changed or added (conditions, channels, repeats). Identity is identical: all four section digests equal v01's, nothing may be invalidated. |
| [`v17_partial_channel_presence`](valid/v17_partial_channel_presence.toml) | Adversarial 9; addendum §3 | pass | **no** | PHASE exists only in CTRL/R002/acquisition_1. For a PHASE request R002 is partially represented and R001 unavailable: the analysis layer must refuse (A_SCOPE_INCOMPLETE) or record a user-selected subset, never drop or re-weight silently. A request for a channel ID that no unit of any fixture lists (the "absent from all" case, e.g. `Tubulin` against `v01`) is the same refusal with k = 0 (owner decision SI1); it needs no fixture of its own because the absent ID is a property of the request. |
| [`v18_inline_and_dotted_forms`](valid/v18_inline_and_dotted_forms.toml) | Extra: round trip | pass | yes | Same model as v01 written with inline tables, dotted keys and quoted keys; parse(v18) must equal parse(v01). |
| [`v19_warnings_unpaired`](valid/v19_warnings_unpaired.toml) | Extra: reference DAG semantics | pass + `W_REFERENCE_UNPAIRED`, `W_REPEAT_MISSING_CONDITIONS` | yes | Valid with warnings: CTRL only in rep1, DrugA only in rep2, so the repeats miss conditions (W_REPEAT_MISSING_CONDITIONS) and DrugA cannot be paired with its reference (W_REFERENCE_UNPAIRED). |
| [`v20_unicode_and_spaces`](valid/v20_unicode_and_spaces.toml) | Extra: identities | pass | yes | Condition folder labels with spaces and non-ASCII text are valid labels (quoted keys); channel and mask IDs stay tokens. |
| [`v21_metadata_only_edit`](valid/v21_metadata_only_edit.toml) | Adversarial 10 | pass | yes | v01 with only descriptive condition fields edited (display name, target removed, treatment_key changed). All four section digests equal v01's (descriptive fields are omitted, experiment-v1 9.3), so no pixel-derived product may be invalidated. |
| [`v22_unused_channel_warning`](valid/v22_unused_channel_warning.toml) | Owner decision 3 | pass + `W_UNUSED_CHANNEL` | yes | PHASE is declared but acquired by no unit: valid with a warning, never an error. |
| [`v23_nfc_identity`](valid/v23_nfc_identity.toml) | Owner decision Q3 | pass | yes | A condition key written composed (U+00E9, NFC) and referenced composed: the only form a writer emits and a reader accepts. |
| [`v24_display_name_as_typed`](valid/v24_display_name_as_typed.toml) | Owner decision Q2 | pass | yes | Display names kept as typed (one decomposed, one lower case); a channel and a condition share the display name "Drug A" because uniqueness is per kind. |
| [`v25_removed_condition_id_retained`](valid/v25_removed_condition_id_retained.toml) | Owner decision Q1 | pass | yes | v01 after DrugA was removed: DrugA stays listed in `used_ids.conditions`. Predecessor of t03 and t04 (and, with the recorded removal confirmation of experiment-v1 4.6, the removal of v01). |
| [`v26_ids_kept_as_written`](valid/v26_ids_kept_as_written.toml) | Audit B1, B2, B11 | pass | yes | IDs `cl001`, `CL1000`, `R999`, `R1000`: no width, prefix or case rule; kept exactly as written. |
| [`v27_descriptive_keys_collide`](valid/v27_descriptive_keys_collide.toml) | Owner decision Q4 | pass | yes | Identical and case-differing `target` / `treatment_key` values on different conditions: valid, no warning, no grouping. |
| [`v28_rgb_units`](valid/v28_rgb_units.toml) | Owner decision AQ6 (supersedes D8) | pass | yes | An RGB experiment (e.g. H&E): every unit declares `image_type = "rgb"` with the fixed channels R, G, B in that order and no aliases. Valid and homogeneous. The scientific meaning is left to each brick: a brick that does not declare RGB support refuses a request on these units (A_RGB_NOT_SUPPORTED, P24); preprocessing compares the declaration with the image header and never with pixels (P13, P23). The `rgb_support` manifest key of experiment-v1 §4.4b (owner decision RC1) is a property of the receiving brick, not of the file, so no TOML fixture can express it; P24 states the four manifest cases to run against this one. |
| [`v29_rgb_and_channels_mixed`](valid/v29_rgb_and_channels_mixed.toml) | Owner decision AQ6 | pass | **no** | One condition with an RGB brightfield unit and a two-channel fluorescence unit of the same biological repeat. Valid schema; heterogeneous (different lists), so the Stage 1-2 runtime refuses it (R_HETEROGENEOUS_LAYOUT_NOT_ENABLED). Both units stay ONE biological repeat. |
| [`v30_same_source_label_different_repeats`](valid/v30_same_source_label_different_repeats.toml) | Owner decision AQ5; S0.1 audit section 5 | pass | yes | Source folder label rep1 is in canonical repeat Day1 for CTRL and in Day2 for DrugA (and rep2 the other way round). Valid: no repeat ID equals another repeat's source label. The FOV key keeps the source label in `biological_repeat` and stays unique through the condition; the added `repeat_group` is Day1 or Day2, and grouping and pairing use it (experiment-v1 3.4, P25). |
| [`v31_header_alias_whitespace_and_decomposed`](valid/v31_header_alias_whitespace_and_decomposed.toml) | Owner decision AQ7 | pass | yes | Header evidence is compared exactly after NFC of both sides, with no trimming. DNA lists the aliases "DAPI", " DAPI" (leading space) and "DAPI " (trailing space): outer whitespace needs its own confirmed alias, each is a distinct entry. ACTIN lists a decomposed alias (written with a TOML escape and stored as typed), which matches the header name "Caf\u00e9 488" after NFC of both sides (P26). header_names are not identities and are not NFC-checked. |
| [`v32_reserved_name_near_misses`](valid/v32_reserved_name_near_misses.toml) | Owner decision AQ4 | pass | yes | Names next to the reserved list (experiment-v1 3.2) are valid: channels c1, t0, z2 and label, masks Patches and Labels. Only the exact names, compared case-folded, are reserved (t, z0, label_id, patch_id, ...). |
| [`v33_condition_folder_on_forbidden_list`](valid/v33_condition_folder_on_forbidden_list.toml) | Owner decision FR1 | pass | yes | The condition folder on disk is `control`, a forbidden name (experiment-v1 4.7), so it cannot be the ID. It is recorded as `source_folder` (never restricted) and the user confirmed the ID `ctrl`, with `display_name` `control`. DrugA references `ctrl`, the ID. The derived experiment_design.csv still has `control` in the `condition` column (experiment-v1 11). |
| [`v34_technical_repeat_folder_on_forbidden_list`](valid/v34_technical_repeat_folder_on_forbidden_list.toml) | Owner decision FR1 | pass | yes | Repeat rep1 has two technical-repeat folders, `files` and `overrides`, both forbidden names (experiment-v1 4.7). They are the `source_folder` values; the confirmed IDs are `acq1` and `acq2`, and the same ID carries the same folder in both members. rep2 has no technical-repeat folder level (`single`). Homogeneous. |
| [`v35_flat_experiment`](valid/v35_flat_experiment.toml) | Matrix audit 2026-11-07, case F01 | pass | yes | The experiment folder is itself the leaf: condition, repeat and technical repeat are all `single`. The condition `single` has `source_folder = "single"` ("no condition folder level", 4.2, 3.2) and is the only condition (owner decision SQ6, matrix audit MS-1). |
| [`v36_repeat_folder_forbidden_name`](valid/v36_repeat_folder_forbidden_name.toml) | Matrix audit 2026-11-07, case F16 | pass | yes | Biological-repeat folders `z` paired into repeat `R1`; `source_repeat` is never restricted. The folder label `z` is not an allowed repeat ID (`i130`), so no ID is proposed and the user names `R1` (4.6a; owner decision SQ6, matrix audit MS-2). |
| [`v37_used_ids_reserved_near_misses`](valid/v37_used_ids_reserved_near_misses.toml) | Owner decision UR1 (Spec issue 11) | pass | yes | `v32` with the retired channel IDs `Ringer` and `label2` listed in `used_ids.channels`: names next to the reserved list are valid there too (experiment-v1 4.9). |

## Expected outputs (3)

Golden `experiment_design.csv` files (experiment-v1 §11), byte for byte. Each is the derivation of the
named valid file by the stated rules; `tools/audit/refval.py` compares them.

| Golden file | Derived from | Notes |
|---|---|---|
| [`v13_homogeneous_legacy.experiment_design.csv`](expected/v13_homogeneous_legacy.experiment_design.csv) | `valid/v13` | One control (`CTRL`, referenced by DrugA and DrugB); every `technical_repeat_count` 0 (only `single`). |
| [`v33_condition_folder_on_forbidden_list.experiment_design.csv`](expected/v33_condition_folder_on_forbidden_list.experiment_design.csv) | `valid/v33` | `condition` holds the folder label `control`, never the ID `ctrl`; `DrugA` sorts before `control` (code point order). |
| [`v34_technical_repeat_folder_on_forbidden_list.experiment_design.csv`](expected/v34_technical_repeat_folder_on_forbidden_list.experiment_design.csv) | `valid/v34` | Two technical repeats (folders `files`, `overrides`) give `technical_repeat_count` 2; `rep2` has only `single`, so 0. |

## Transition fixtures (13)

Valid alone; checked against the named published predecessor (refused with the listed set, or accepted).

| Fixture | Predecessor | Expected | Case |
|---|---|---|---|
| [`t01_repeat_id_changed`](transitions/t01_repeat_id_changed.toml) | `valid/v01` | `R_CONFIRMED_ID_CHANGED` unless re-confirmation is recorded | Repeats rep1/rep2 (same member folders) are now Day1/Day2; rep1, rep2 stay listed in `used_ids`. |
| [`t02_channel_id_changed`](transitions/t02_channel_id_changed.toml) | `valid/v01` | `R_CONFIRMED_ID_CHANGED`, `R_USED_ID_DROPPED` | Channel ACTIN (same header evidence and positions) is now Actin, and ACTIN is no longer listed in `used_ids`. Listing it beside `Actin` would be `E_ID_CASE_COLLISION`, so a case-only ID change cannot be published in place (owner decision AQ2): it needs a new ID, `t09`. |
| [`t03_used_id_reused`](transitions/t03_used_id_reused.toml) | `valid/v25` | `R_USED_ID_REUSED` (only) | DrugA (retired in v25) is declared again for a new source folder. rep1 gains a member, which is not a changed entity (owner decision AQ1), so `R_CONFIRMED_ID_CHANGED` does not apply. Valid alone, with `W_REPEAT_MISSING_CONDITIONS` (rep2 has no DrugA member). |
| [`t04_used_id_dropped`](transitions/t04_used_id_dropped.toml) | `valid/v25` | `R_USED_ID_DROPPED` | DrugA is no longer listed in `used_ids.conditions`. |
| [`t05_confirmed_on_changed`](transitions/t05_confirmed_on_changed.toml) | `valid/v01` | `R_CONFIRMED_ON_CHANGED` (N35 not reopened) | rep1 unchanged except `confirmed_on` 2026-11-07. |
| [`t06_member_added`](transitions/t06_member_added.toml) | `valid/v01` | no R_* code: accepted (against predecessor valid/v01) | Adding is not a change of entity. Against v01: a new condition DrugB joins the existing repeats rep1 and rep2 as a member, and a new folder CTRL/rep3 (new repeat rep3) joins the existing condition CTRL. Every member folder, position and alias of v01 is still there. The confirmation of rep1 and rep2 is reopened by the new member (grouping change, experiment-v1 10) and re-confirmed on 2026-11-07, so the changed `confirmed_on` is not R_CONFIRMED_ON_CHANGED. DrugB and rep3 are new IDs, listed in `used_ids`. Valid alone with W_REPEAT_MISSING_CONDITIONS (rep3). |
| [`t07_member_removed`](transitions/t07_member_removed.toml) | `valid/v01` | R_CONFIRMED_ID_CHANGED (against predecessor valid/v01, without a recorded removal confirmation) | Removing is a change of entity. The member DrugA/rep2 of repeat rep2 is gone (rep2 reopened and re-confirmed on 2026-11-07). The removal is accepted only with the recorded user confirmation of experiment-v1 4.6 and 10; without it the candidate is refused. All IDs stay listed in `used_ids`. Valid alone with W_REPEAT_MISSING_CONDITIONS. |
| [`t08_member_reassigned`](transitions/t08_member_reassigned.toml) | `valid/v03` | R_CONFIRMED_ID_CHANGED (against predecessor valid/v03, without a recorded confirmation) | Reassigning is a change of entity. DrugA/experiment_23 was a member of Day1 in v03 and is now the member of a new repeat Day3, so Day1 lost a member it held. Accepted only with the recorded user confirmation of experiment-v1 4.6. Day3 is a new ID listed in `used_ids`. Valid alone with W_REPEAT_MISSING_CONDITIONS. |
| [`t09_case_correction_new_id`](transitions/t09_case_correction_new_id.toml) | `valid/v01` | R_CONFIRMED_ID_CHANGED (against predecessor valid/v01, accepted only with a recorded re-confirmation) | How a case-only mistake is corrected. v01's channel ACTIN was meant to be Actin; the corrected spelling cannot be used (it collides with the retired ACTIN, see t02 and i95), so the channel gets the new, non-colliding ID ActinStain at the same positions with the same header evidence, and `display_name` "Actin" carries the appearance. ACTIN stays listed in `used_ids`. The set is exactly {R_CONFIRMED_ID_CHANGED}: no R_USED_ID_DROPPED, no R_USED_ID_REUSED, no E_* code. |
| [`t10_condition_id_changed_same_folder`](transitions/t10_condition_id_changed_same_folder.toml) | `valid/v33` | R_CONFIRMED_ID_CHANGED (against predecessor valid/v33, without a recorded re-confirmation) | The folder `control` was confirmed as condition `ctrl` in v33 and now carries the ID `control_cells`. A confirmed ID never changes in place, also when the folder is the same (experiment-v1 4.6, 4.6a). `ctrl` stays listed in used_ids, so only the changed ID is refused. Valid alone. |
| [`t11_technical_repeat_id_changed_same_folder`](transitions/t11_technical_repeat_id_changed_same_folder.toml) | `valid/v34` | R_CONFIRMED_ID_CHANGED (against predecessor valid/v34, without a recorded re-confirmation) | The folder `files` was confirmed as technical repeat `acq1` in v34 and now carries the ID `acqA` in both members. A confirmed technical repeat ID never changes in place (experiment-v1 4.6, 4.6a). `acq1` stays listed in used_ids. Valid alone. |
| [`t12_condition_folder_reassigned`](transitions/t12_condition_folder_reassigned.toml) | `valid/v33` | R_CONFIRMED_ID_CHANGED (against predecessor valid/v33, without a recorded confirmation) | The ID `ctrl` held the folder `control` in v33 and now holds the folder `ctl` (a folder renamed on disk and attached to the same ID without a recorded confirmation). A folder change for an existing ID is a reassignment, so a changed entity (experiment-v1 4.6, 7). Valid alone. |
| [`t13_confirmed_on_with_image_count_only`](transitions/t13_confirmed_on_with_image_count_only.toml) | `valid/v01` | `R_CONFIRMED_ON_CHANGED` (N35 not reopened) | Against v01 only CTRL/rep1 `image_count` (12 to 17, new FOVs) and rep1 `confirmed_on` (2026-11-07) change: new FOVs do not reopen the confirmation (experiment-v1 10), so the new date is refused. |

## Invalid fixtures (135)

| Fixture | Source | Expected code(s) | Phase | Case |
|---|---|---|---|---|
| [`i01_bom`](invalid/i01_bom.toml) | Extra: encoding | `E_FILE_ENCODING` | 1 | The file starts with a UTF-8 byte-order mark. |
| [`i02_invalid_utf8`](invalid/i02_invalid_utf8.toml) | Extra: encoding | `E_FILE_ENCODING` | 1 | A label contains the byte 0xFF (not UTF-8). |
| [`i03_duplicate_condition_table`](invalid/i03_duplicate_condition_table.toml) | Extra: duplicate keys; audit §3.12 | `E_DUPLICATE_KEY` | 1 | [conditions.CTRL] declared twice (YAML would keep the second silently). |
| [`i04_duplicate_key_quoted_vs_bare`](invalid/i04_duplicate_key_quoted_vs_bare.toml) | Extra: duplicate keys | `E_DUPLICATE_KEY` | 1 | [conditions."CTRL"] is the same key as [conditions.CTRL]. |
| [`i05_duplicate_field`](invalid/i05_duplicate_field.toml) | Extra: duplicate keys | `E_DUPLICATE_KEY` | 1 | `display_name` written twice in one condition. |
| [`i06_duplicate_member_dotted_vs_table`](invalid/i06_duplicate_member_dotted_vs_table.toml) | Extra: duplicate keys | `E_DUPLICATE_KEY` | 1 | A member declared as a table and again by a dotted key (two members of one condition in one repeat). |
| [`i07_unquoted_bareword`](invalid/i07_unquoted_bareword.toml) | Extra: implicit conversion (YAML `NO` -> False) | `E_TOML_SYNTAX` | 1 | source_repeat = NO, unquoted. |
| [`i08_leading_zero_integer`](invalid/i08_leading_zero_integer.toml) | Extra: implicit conversion (YAML `012` -> 10) | `E_TOML_SYNTAX` | 1 | source_repeat = 012, unquoted. |
| [`i09_schema_version_2`](invalid/i09_schema_version_2.toml) | Extra: versioning | `E_SCHEMA_VERSION_UNSUPPORTED` | 2 | schema_version = 2 is refused, never read best effort. |
| [`i10_schema_version_0`](invalid/i10_schema_version_0.toml) | Extra: versioning | `E_SCHEMA_VERSION_UNSUPPORTED` | 2 | schema_version = 0. |
| [`i11_schema_version_missing`](invalid/i11_schema_version_missing.toml) | Extra: versioning | `E_SCHEMA_VERSION_MISSING` | 2 | No schema_version. |
| [`i12_schema_version_string`](invalid/i12_schema_version_string.toml) | Extra: implicit conversion | `E_TYPE` | 2 | schema_version = "1" (a string is not converted to an integer). |
| [`i13_schema_version_float`](invalid/i13_schema_version_float.toml) | Extra: implicit conversion | `E_TYPE` | 2 | schema_version = 1.0. |
| [`i14_schema_version_bool`](invalid/i14_schema_version_bool.toml) | Extra: implicit conversion | `E_TYPE` | 2 | schema_version = true (a boolean is not the integer 1). |
| [`i15_schema_id_wrong`](invalid/i15_schema_id_wrong.toml) | Extra: versioning | `E_SCHEMA_ID` | 2 | schema = "morphobricks-experiments". |
| [`i16_unknown_top_level_field`](invalid/i16_unknown_top_level_field.toml) | Extra: unknown fields | `E_UNKNOWN_FIELD` | 3 | Top-level `notes`. |
| [`i17_unknown_condition_field`](invalid/i17_unknown_condition_field.toml) | Extra: unknown fields | `E_UNKNOWN_FIELD` | 3 | Condition field `colour`. |
| [`i18_misspelled_field`](invalid/i18_misspelled_field.toml) | Extra: unknown fields | `E_MISSING_FIELD`, `E_UNKNOWN_FIELD` | 3 | `representaton` (typo) is not silently ignored; `representation` is then also missing. |
| [`i19_dotted_float_key`](invalid/i19_dotted_float_key.toml) | Extra: implicit conversion | `E_UNKNOWN_FIELD` | 3 | [conditions.1.0] is condition `1` with a sub-table `0`, not condition `1.0`. |
| [`i20_per_fov_override`](invalid/i20_per_fov_override.toml) | Extra: within-unit divergence; ADR (no per-FOV overrides) | `E_FORBIDDEN_FIELD` | 3 | A per-FOV channel override inside a technical repeat. |
| [`i21_mask_axes_shared`](invalid/i21_mask_axes_shared.toml) | Extra: masks; preprocessing #173 | `E_FORBIDDEN_FIELD` | 3 | Mask `axes = "YX"` and `shared = true` (v1 is 2-D only; a mask has its image's axes). |
| [`i22_brick_routing`](invalid/i22_brick_routing.toml) | Extra: ADR (no brick routing) | `E_FORBIDDEN_FIELD` | 3 | A channel routed to bricks. |
| [`i23_calibration_in_file`](invalid/i23_calibration_in_file.toml) | Extra: calibration stays in calibration.csv | `E_FORBIDDEN_FIELD` | 3 | pixel_size on a repeat. |
| [`i24_control_boolean`](invalid/i24_control_boolean.toml) | Extra: audit §3.3 | `E_FORBIDDEN_FIELD` | 3 | Legacy boolean `control` (derived from `reference` in v1). |
| [`i25_experiment_field`](invalid/i25_experiment_field.toml) | Extra: audit §3.4 | `E_FORBIDDEN_FIELD` | 3 | Condition `experiment` (an FDR family; MorphoCoverage configuration). |
| [`i26_source_index_field`](invalid/i26_source_index_field.toml) | Extra: RFC (order is the mapping) | `E_FORBIDDEN_FIELD` | 3 | A unit states `source_index` beside its ordered list. |
| [`i27_toml_date_source_repeat`](invalid/i27_toml_date_source_repeat.toml) | Extra: implicit conversion (YAML date) | `E_TYPE` | 3 | source_repeat = 2026-01-01 is a TOML date, not text. |
| [`i28_integer_source_repeat`](invalid/i28_integer_source_repeat.toml) | Extra: implicit conversion | `E_TYPE` | 3 | source_repeat = 12 (integer). |
| [`i29_bool_label`](invalid/i29_bool_label.toml) | Extra: implicit conversion (YAML `off`) | `E_TYPE` | 3 | display_name = false. |
| [`i30_confirmed_on_toml_date`](invalid/i30_confirmed_on_toml_date.toml) | Extra: implicit conversion | `E_TYPE` | 3 | confirmed_on = 2026-11-06 (TOML date literal, not a string). |
| [`i31_confirmed_on_bad_format`](invalid/i31_confirmed_on_bad_format.toml) | Extra: preprocessing contract §12 (spreadsheet dates) | `E_DATE_FORMAT` | 3 | confirmed_on = "06/11/2026". |
| [`i32_confirmed_on_impossible_date`](invalid/i32_confirmed_on_impossible_date.toml) | Extra | `E_DATE_FORMAT` | 3 | confirmed_on = "2026-02-30". |
| [`i33_image_count_string`](invalid/i33_image_count_string.toml) | Extra: implicit conversion | `E_TYPE` | 3 | image_count = "12". |
| [`i34_image_count_float`](invalid/i34_image_count_float.toml) | Extra: implicit conversion | `E_TYPE` | 3 | image_count = 12.0. |
| [`i35_image_count_zero`](invalid/i35_image_count_zero.toml) | Extra | `E_VALUE_RANGE` | 3 | image_count = 0. |
| [`i36_confirmed_string`](invalid/i36_confirmed_string.toml) | Extra: implicit conversion (legacy yes/no) | `E_TYPE` | 3 | confirmed = "yes". |
| [`i37_repeat_unconfirmed`](invalid/i37_repeat_unconfirmed.toml) | Adversarial 7 (N35) | `E_REPEAT_UNCONFIRMED` | 3 | A published file with an unconfirmed repeat. |
| [`i38_mixed_type_channel_list`](invalid/i38_mixed_type_channel_list.toml) | Extra: implicit conversion | `E_TYPE` | 3 | channels = ["DNA", 2]. |
| [`i39_channels_as_string`](invalid/i39_channels_as_string.toml) | Extra | `E_TYPE` | 3 | channels = "DNA,ACTIN" (no splitting of text). |
| [`i40_empty_channel_list`](invalid/i40_empty_channel_list.toml) | Extra | `E_EMPTY` | 3 | A unit with channels = []. |
| [`i41_reference_list`](invalid/i41_reference_list.toml) | Extra: reference DAG | `E_TYPE` | 3 | Two references in an array; v1 has one primary reference. |
| [`i42_representation_invalid`](invalid/i42_representation_invalid.toml) | Extra: masks | `E_ENUM` | 3 | representation = "labels". |
| [`i43_representation_missing`](invalid/i43_representation_missing.toml) | Extra: masks (never inferred from pixels) | `E_MISSING_FIELD` | 3 | A mask without representation. |
| [`i44_missing_channels_table`](invalid/i44_missing_channels_table.toml) | Extra | `E_MISSING_FIELD` | 3 | No [channels] at all. |
| [`i45_member_without_technical_repeats`](invalid/i45_member_without_technical_repeats.toml) | Extra | `E_MISSING_FIELD` | 3 | A member with no technical_repeats. |
| [`i46_header_name_empty`](invalid/i46_header_name_empty.toml) | Extra: aliases | `E_TEXT_EMPTY` | 3 | header_names = [""]. |
| [`i47_label_whitespace_only`](invalid/i47_label_whitespace_only.toml) | Extra | `E_TEXT_EMPTY` | 3 | display_name = "  ". |
| [`i48_case_collision_conditions`](invalid/i48_case_collision_conditions.toml) | Extra: case collisions | `E_ID_CASE_COLLISION` | 4 | Conditions CTRL and ctrl. |
| [`i49_case_collision_channels`](invalid/i49_case_collision_channels.toml) | Extra: case collisions | `E_ID_CASE_COLLISION` | 4 | Channels DNA and dna. |
| [`i50_case_collision_source_repeats`](invalid/i50_case_collision_source_repeats.toml) | Extra: case collisions; preprocessing check_identity_case | `E_ID_CASE_COLLISION` | 4 | Source folders Rep1 (CTRL) and rep1 (DrugA): a label is spelled one way in the whole run. |
| [`i51_case_collision_repeat_ids`](invalid/i51_case_collision_repeat_ids.toml) | Extra: case collisions | `E_ID_CASE_COLLISION` | 4 | Canonical repeats rep1 and REP1. |
| [`i52_case_collision_technical_repeats`](invalid/i52_case_collision_technical_repeats.toml) | Extra: case collisions | `E_ID_CASE_COLLISION` | 4 | Technical repeats acq_1 and Acq_1 in different members. |
| [`i53_display_name_collision_conditions`](invalid/i53_display_name_collision_conditions.toml) | Owner refinement | `E_DISPLAY_NAME_COLLISION` | 5 | Two conditions with display_name "Control". |
| [`i54_channel_id_missing_text`](invalid/i54_channel_id_missing_text.toml) | Extra: identities (decision D-e) | `E_ID_TEXT` | 4 | Channel ID `NA` (tables read it as missing). |
| [`i55_condition_outer_whitespace`](invalid/i55_condition_outer_whitespace.toml) | Extra: identities (decision D-e) | `E_ID_TEXT` | 4 | Condition key " CTRL" with a leading space. |
| [`i56_source_repeat_null`](invalid/i56_source_repeat_null.toml) | Extra: identities | `E_ID_TEXT` | 4 | source_repeat = "null". |
| [`i57_channel_id_hyphen`](invalid/i57_channel_id_hyphen.toml) | Extra: identities (token rule) | `E_ID_SYNTAX` | 4 | Channel ID F-actin. |
| [`i58_channel_id_double_underscore`](invalid/i58_channel_id_double_underscore.toml) | Extra: identities (feature-column separator) | `E_ID_SYNTAX` | 4 | Channel ID DNA__2. |
| [`i59_mask_id_reserved`](invalid/i59_mask_id_reserved.toml) | Extra: MorphoFeature object-set rule | `E_ID_RESERVED` | 4 | Mask ID `Nucleus` (reserved compartment, any case). |
| [`i60_mask_id_reserved_in`](invalid/i60_mask_id_reserved_in.toml) | Extra: MorphoFeature object-set rule | `E_ID_RESERVED` | 4 | Mask ID `Cells_in_Tumor` (an `in` part). |
| [`i61_channel_id_reserved`](invalid/i61_channel_id_reserved.toml) | Extra: audit §3.8 | `E_ID_RESERVED` | 4 | Channel ID `Ring` (MorphoRelate compartment token). |
| [`i62_single_wrong_case`](invalid/i62_single_wrong_case.toml) | Extra: single convention | `E_SINGLE_MISUSE` | 4 | Technical repeat `Single` (spelled one way, so no case collision). |
| [`i63_single_not_alone`](invalid/i63_single_not_alone.toml) | Extra: single convention | `E_SINGLE_MISUSE` | 4 | Technical repeat `single` beside `acquisition_2`. |
| [`i64_reference_cycle`](invalid/i64_reference_cycle.toml) | RFC case 12 | `E_REFERENCE_CYCLE` | 5 | CTRL -> DrugA -> CTRL. |
| [`i65_reference_cycle_long`](invalid/i65_reference_cycle_long.toml) | RFC case 12 | `E_REFERENCE_CYCLE` | 5 | CTRL -> DrugB -> DrugA -> CTRL. |
| [`i66_reference_self`](invalid/i66_reference_self.toml) | Extra: reference DAG | `E_REFERENCE_SELF` | 5 | DrugA references itself. |
| [`i67_reference_unknown`](invalid/i67_reference_unknown.toml) | Extra: reference DAG | `E_REFERENCE_UNKNOWN` | 5 | Reference to undeclared DMSO. |
| [`i68_reference_case_mismatch`](invalid/i68_reference_case_mismatch.toml) | Extra: case rules (no folding on resolution) | `E_REFERENCE_UNKNOWN` | 5 | reference = "ctrl" does not resolve to CTRL. |
| [`i69_unknown_channel_in_unit`](invalid/i69_unknown_channel_in_unit.toml) | Extra: extra channel not declared | `E_UNKNOWN_CHANNEL` | 5 | A unit lists PHASE, which is not declared. |
| [`i70_duplicate_channel_in_unit`](invalid/i70_duplicate_channel_in_unit.toml) | Extra | `E_DUPLICATE_CHANNEL_IN_UNIT` | 5 | channels = ["DNA", "ACTIN", "DNA"]. |
| [`i71_target_not_text`](invalid/i71_target_not_text.toml) | Owner decision 1 | `E_TYPE` | 3 | target = 5 (plain strings only). |
| [`i72_member_unknown_condition`](invalid/i72_member_unknown_condition.toml) | Extra | `E_UNKNOWN_CONDITION` | 5 | A member for undeclared condition DrugB. |
| [`i73_condition_without_members`](invalid/i73_condition_without_members.toml) | Extra | `E_CONDITION_WITHOUT_MEMBERS` | 5 | Condition DrugB declared, no member. |
| [`i74_source_folder_in_two_repeats`](invalid/i74_source_folder_in_two_repeats.toml) | Extra: repeat pairing | `E_SOURCE_REPEAT_REUSED` | 5 | Source folder CTRL/exp7 belongs to repeats Day2 and Day3. |
| [`i75_repeat_id_swap`](invalid/i75_repeat_id_swap.toml) | Extra: renames | `E_REPEAT_ID_SHADOWS_SOURCE` | 5 | Repeat rep1 holds folder rep2 and repeat rep2 holds folder rep1 for CTRL. |
| [`i76_display_name_collision_repeats`](invalid/i76_display_name_collision_repeats.toml) | Owner refinement | `E_DISPLAY_NAME_COLLISION` | 5 | Repeats rep1 and rep2 both display as "Day 1". |
| [`i77_display_name_collision_channels`](invalid/i77_display_name_collision_channels.toml) | Owner refinement | `E_DISPLAY_NAME_COLLISION` | 5 | Channels DNA and ACTIN both display as "DAPI". |
| [`i78_non_nfc_condition_key`](invalid/i78_non_nfc_condition_key.toml) | Owner decision Q3 | `E_ID_NOT_NFC` | 4 | The condition key is decomposed "Cafe\u0301"; refused, never normalized. |
| [`i79_mask_source_channel_unknown`](invalid/i79_mask_source_channel_unknown.toml) | Extra: masks | `E_UNKNOWN_CHANNEL` | 5 | source_channels names undeclared Hoechst. |
| [`i80_mask_source_type_reused`](invalid/i80_mask_source_type_reused.toml) | Extra: masks | `E_MASK_SOURCE_TYPE_REUSED` | 5 | Two masks with source_type nuclei. |
| [`i81_header_names_duplicate`](invalid/i81_header_names_duplicate.toml) | Extra: aliases | `E_DUPLICATE_VALUE` | 5 | header_names = ["DAPI", "DAPI"]. |
| [`i82_retired_repeats_forbidden`](invalid/i82_retired_repeats_forbidden.toml) | Owner refinement | `E_FORBIDDEN_FIELD` | 3 | A `retired_repeats` table: IDs are immutable, v1 has no rename table. |
| [`i83_non_nfc_reference`](invalid/i83_non_nfc_reference.toml) | Owner decision Q3 | `E_ID_NOT_NFC` | 4 | A composed key referenced decomposed: refused in phase 4, not resolved through normalization. |
| [`i84_non_nfc_source_repeat`](invalid/i84_non_nfc_source_repeat.toml) | Owner decision Q3 | `E_ID_NOT_NFC` | 4 | A decomposed `source_repeat`. |
| [`i85_non_nfc_used_id`](invalid/i85_non_nfc_used_id.toml) | Owner decisions Q1, Q3 | `E_ID_NOT_NFC` | 4 | A decomposed retired ID in `used_ids`. |
| [`i86_display_name_collision_case`](invalid/i86_display_name_collision_case.toml) | Owner decision Q2 | `E_DISPLAY_NAME_COLLISION` | 5 | "Control" and "control". |
| [`i87_display_name_collision_casefold`](invalid/i87_display_name_collision_casefold.toml) | Owner decision Q2 | `E_DISPLAY_NAME_COLLISION` | 5 | Channels "Stra\u00dfe" and "STRASSE" (full case folding). |
| [`i88_display_name_collision_nfc_and_case`](invalid/i88_display_name_collision_nfc_and_case.toml) | Owner decision Q2 | `E_DISPLAY_NAME_COLLISION` | 5 | Repeats "Cafe\u0301 1" and "CAF\u00c9 1". |
| [`i89_reference_names_display_name`](invalid/i89_reference_names_display_name.toml) | Audit L1; owner decision A4 | `E_REFERENCE_UNKNOWN` | 5 | `reference` is the display name "Control" of CTRL. |
| [`i90_member_keyed_by_display_name`](invalid/i90_member_keyed_by_display_name.toml) | Audit L1 | `E_UNKNOWN_CONDITION` | 5 | A member keyed by the display name "Control". |
| [`i91_unit_lists_channel_display_name`](invalid/i91_unit_lists_channel_display_name.toml) | Audit L1 | `E_UNKNOWN_CHANNEL` | 5 | A unit lists the channel display name "DAPI". |
| [`i92_used_ids_missing`](invalid/i92_used_ids_missing.toml) | Owner decision Q1 | `E_MISSING_FIELD` | 3 | No `used_ids` table. |
| [`i93_used_id_not_listed`](invalid/i93_used_id_not_listed.toml) | Owner decision Q1 | `E_USED_ID_MISSING` | 5 | DrugA declared but not in `used_ids.conditions`. |
| [`i94_used_ids_duplicate`](invalid/i94_used_ids_duplicate.toml) | Owner decision Q1 | `E_DUPLICATE_VALUE` | 5 | rep1 listed twice. |
| [`i95_used_id_case_collision`](invalid/i95_used_id_case_collision.toml) | Audit B2, B11; owner decision Q1 | `E_ID_CASE_COLLISION` | 4 | Retired `ctrl` against declared CTRL. |
| [`i96_used_ids_mapping`](invalid/i96_used_ids_mapping.toml) | Owner decision Q1 | `E_TYPE` | 3 | `used_ids.repeats` is a mapping table, not a list. |
| [`i97_used_ids_unknown_kind`](invalid/i97_used_ids_unknown_kind.toml) | Owner decision Q1 | `E_UNKNOWN_FIELD` | 3 | `used_ids.masks`. |
| [`i98_confirmed_on_missing`](invalid/i98_confirmed_on_missing.toml) | Audit L8; owner decision A3 | `E_MISSING_FIELD` | 3 | No `confirmed_on`: no default exists. |
| [`i99_labchronicle_provenance`](invalid/i99_labchronicle_provenance.toml) | Owner decision Q5 | `E_FORBIDDEN_FIELD` | 3 | A top-level `labchronicle` table. |
| [`i100_image_type_unknown_value`](invalid/i100_image_type_unknown_value.toml) | Owner decision AQ6 | `E_ENUM` | 3 | `image_type = "RGB"` (upper case): the only value is exactly `rgb`. |
| [`i101_image_type_not_text`](invalid/i101_image_type_not_text.toml) | Owner decision AQ6 | `E_TYPE` | 3 | `image_type = true`: a boolean is not converted to text. |
| [`i102_rgb_unit_wrong_order`](invalid/i102_rgb_unit_wrong_order.toml) | Owner decision AQ6 | `E_RGB_UNIT_CHANNELS` | 5 | An RGB unit lists R, B, G: the order is fixed (R, G, B). |
| [`i103_rgb_unit_other_channels`](invalid/i103_rgb_unit_other_channels.toml) | Owner decision AQ6 | `E_RGB_UNIT_CHANNELS` | 5 | A unit declared `rgb` lists the fluorescence channels DNA and ACTIN: an RGB unit has exactly R, G, B. |
| [`i104_rgb_token_in_plain_unit`](invalid/i104_rgb_token_in_plain_unit.toml) | Owner decision AQ6 | `E_RGB_TOKEN_MISUSE` | 5 | Units that list R, G, B without `image_type = "rgb"`: the tokens are only for declared RGB units (and an undeclared RGB image is never inferred). |
| [`i105_rgb_token_lowercase_in_plain_unit`](invalid/i105_rgb_token_lowercase_in_plain_unit.toml) | Owner decision AQ6 | `E_RGB_TOKEN_MISUSE` | 5 | A fluorescence unit lists a channel `r`: R, G, B are reserved for RGB units in any case. |
| [`i106_rgb_channel_header_alias`](invalid/i106_rgb_channel_header_alias.toml) | Owner decision AQ6 | `E_RGB_ALIAS` | 5 | The channel R carries `header_names`: R, G, B have no aliases. |
| [`i107_channel_id_reserved_column`](invalid/i107_channel_id_reserved_column.toml) | Owner decision AQ4 | `E_ID_RESERVED` | 4 | Channel ID `label_id`, a MorphoRelate reserved column name (registry/_expand.py:41-59). |
| [`i108_channel_id_reserved_casefold`](invalid/i108_channel_id_reserved_casefold.toml) | Owner decision AQ4 | `E_ID_RESERVED` | 4 | Channel ID `T`: reserved names are compared case-folded, so `T` is the reserved `t`. |
| [`i109_mask_id_reserved_column`](invalid/i109_mask_id_reserved_column.toml) | Owner decision AQ4 | `E_ID_RESERVED` | 4 | Mask ID `patch_id`, a MorphoRelate reserved column name: one reserved list for channel and mask IDs. |
| [`i110_header_names_duplicate_after_nfc`](invalid/i110_header_names_duplicate_after_nfc.toml) | Owner decision AQ7 | `E_DUPLICATE_VALUE` | 5 | `header_names` = ["Caf\u00e9", "Cafe\u0301"]: header names are compared after NFC, so these are the same alias twice. |
| [`i111_mask_representation_parent_labels`](invalid/i111_mask_representation_parent_labels.toml) | Owner decision AQ8 | `E_ENUM` | 3 | `representation = "parent_labels"` (MorphoRelate's grouped masks): out of scope in v1. |
| [`i112_mask_conversion_field_forbidden`](invalid/i112_mask_conversion_field_forbidden.toml) | Owner decision AQ9 | `E_FORBIDDEN_FIELD` | 3 | `binary_mask_policy = "connected_components"` in a mask table: a binary-to-instances conversion is a per-request, recorded choice of the analysis request, never part of the file. |
| [`i113_forbidden_name_condition_key`](invalid/i113_forbidden_name_condition_key.toml) | Owner decision AQ13a | `E_FORBIDDEN_FIELD` | 3 | A condition ID `control`: forbidden names apply to ID keys as well as fields (reported at the condition and at its member keys). The folder `control` itself is describable under another ID (`v33`); only the key is refused. |
| [`i114_forbidden_name_channel_key`](invalid/i114_forbidden_name_channel_key.toml) | Owner decision AQ13a | `E_FORBIDDEN_FIELD` | 3 | A channel ID `files`: a forbidden name used as a channel key. |
| [`i115_forbidden_name_technical_repeat_key`](invalid/i115_forbidden_name_technical_repeat_key.toml) | Owner decision AQ13a | `E_FORBIDDEN_FIELD` | 3 | A technical repeat ID `overrides`: a forbidden name used as a technical-repeat key. The folder `overrides` is describable under another ID (`v34`). |
| [`i116_non_nfc_unit_channel`](invalid/i116_non_nfc_unit_channel.toml) | Owner decision AQ13b | `E_ID_NOT_NFC` | 4 | A unit lists "ACTIN\u0301" (decomposed). The reference is checked for NFC in phase 4 and is not reported as E_UNKNOWN_CHANNEL (phase 5). |
| [`i117_non_nfc_source_channel`](invalid/i117_non_nfc_source_channel.toml) | Owner decision AQ13b | `E_ID_NOT_NFC` | 4 | A mask's `source_channels` lists "DNA\u0301" (decomposed): NFC before resolution, so not E_UNKNOWN_CHANNEL. |
| [`i118_channel_id_whitespace_and_syntax`](invalid/i118_channel_id_whitespace_and_syntax.toml) | Owner decision AQ13c | `E_ID_TEXT`, `E_ID_SYNTAX` | 4 | Channel ID " DNA" (leading space): it breaks two rules, the Label rule (outer whitespace) and the Token expression, and gets both codes (one code per broken rule). |
| [`i119_rgb_unit_two_channels`](invalid/i119_rgb_unit_two_channels.toml) | Owner decision AQ6 | `E_RGB_UNIT_CHANNELS` | 5 | An RGB unit lists only R and G: exactly three channels are required. |
| [`i120_condition_missing_source_folder`](invalid/i120_condition_missing_source_folder.toml) | Owner decision FR1 | `E_MISSING_FIELD` | 3 | Condition DrugA has no `source_folder`: it is required, with no default (experiment-v1 4.2). |
| [`i121_technical_repeat_missing_source_folder`](invalid/i121_technical_repeat_missing_source_folder.toml) | Owner decision FR1 | `E_MISSING_FIELD` | 3 | The unit rep2/DrugA/single has no `source_folder`: required for every technical repeat, `single` included (experiment-v1 4.4). |
| [`i122_two_condition_ids_one_folder`](invalid/i122_two_condition_ids_one_folder.toml) | Owner decision FR1 | `E_SOURCE_FOLDER_REUSED` | 5 | Condition IDs CTRL and DrugA both name the folder `control`: one folder, one ID (experiment-v1 4.6a). |
| [`i123_two_technical_repeat_ids_one_folder`](invalid/i123_two_technical_repeat_ids_one_folder.toml) | Owner decision FR1 | `E_SOURCE_FOLDER_REUSED` | 5 | Technical repeat IDs acq1 and acq2 of one member both name the folder `acquisition_1` (experiment-v1 4.6a). |
| [`i124_one_technical_repeat_id_two_folders`](invalid/i124_one_technical_repeat_id_two_folders.toml) | Owner decision FR1 | `E_SOURCE_FOLDER_CONFLICT` | 5 | The technical repeat ID acq1 is declared with the folder `acquisition_1` in CTRL and `acq_1` in DrugA: one ID, one folder over the whole file (experiment-v1 4.6a). |
| [`i125_single_id_with_real_folder`](invalid/i125_single_id_with_real_folder.toml) | Owner decision FR1 | `E_SINGLE_MISUSE` | 4 | The technical repeat ID `single` names the folder `acquisition_1`: `single` means no technical-repeat folder level and is the ID of `source_folder = "single"` only (experiment-v1 3.2). |
| [`i126_used_ids_technical_repeats_missing`](invalid/i126_used_ids_technical_repeats_missing.toml) | Owner decision FR1 | `E_MISSING_FIELD` | 3 | `used_ids` has no `technical_repeats` list: required, like the other three kinds (experiment-v1 4.9). |
| [`i127_technical_repeat_id_not_in_used_ids`](invalid/i127_technical_repeat_id_not_in_used_ids.toml) | Owner decision FR1 | `E_USED_ID_MISSING` | 5 | The declared technical repeat ID `single` is not listed in `used_ids.technical_repeats` (experiment-v1 4.9). |
| [`i128_condition_folder_single_wrong_case`](invalid/i128_condition_folder_single_wrong_case.toml) | Matrix audit 2026-11-07, case F18 | `E_SINGLE_MISUSE` | 4 | A condition `source_folder` `Single` (source folders included in 3.2). |
| [`i129_source_repeat_single_not_alone`](invalid/i129_source_repeat_single_not_alone.toml) | Matrix audit 2026-11-07 | `E_SINGLE_MISUSE` | 4 | CTRL has the member `single` and the member `rep2`. |
| [`i130_repeat_id_from_forbidden_folder`](invalid/i130_repeat_id_from_forbidden_folder.toml) | Matrix audit 2026-11-07, case F16 | `E_FORBIDDEN_FIELD` | 3 | Repeat folders `z` kept as the repeat ID `z`; no ID is proposed for such a folder (4.6a), the user names it (`v36`). |
| [`i131_condition_id_single_with_real_folder`](invalid/i131_condition_id_single_with_real_folder.toml) | Owner decision SQ6, matrix audit MS-1 | `E_SINGLE_MISUSE` | 4 | The condition ID `single` names the folder `CTRL`: `single` is the ID of `source_folder = "single"` only (experiment-v1 3.2). |
| [`i132_condition_single_folder_other_id`](invalid/i132_condition_single_folder_other_id.toml) | Owner decision SQ6, matrix audit MS-1 | `E_SINGLE_MISUSE` | 4 | The condition `flat` has `source_folder = "single"`: a condition ID is `single` if and only if its folder is `single` (experiment-v1 3.2). |
| [`i133_condition_single_not_alone`](invalid/i133_condition_single_not_alone.toml) | Owner decision SQ6, matrix audit MS-1 | `E_SINGLE_MISUSE` | 4 | The condition `single` (no condition folder level) next to the condition DrugA: it must be the only condition (experiment-v1 3.2). |
| [`i134_duplicate_key_inline_table`](invalid/i134_duplicate_key_inline_table.toml) | Stage 1 design PR 0, experiment-v1 2.2 | `E_DUPLICATE_KEY` | 1 | `display_name` written twice inside the inline table of the channel DNA: a key declared twice in any spelling, inline tables included, is `E_DUPLICATE_KEY`, not `E_TOML_SYNTAX` (experiment-v1 2.2). |
| [`i135_used_id_channel_reserved`](invalid/i135_used_id_channel_reserved.toml) | Owner decision UR1 (Spec issue 11) | `E_ID_RESERVED` | 4 | The retired channel ID `Nucleus` in `used_ids.channels`: a listed ID reserves the ID, so it follows the reserved list of 3.2 (case-folded) like a declared channel ID (experiment-v1 4.9). |

## Records fixtures (37)

Byte-exact test data for `morphobricks.records.jsonl` (experiment-v1 §4.6, owner decisions RF1-RF6
of 9 November 2026). Unlike the TOML fixtures a JSONL file has no comment header, so each case is
listed here. Every file is checked by `tools/audit/refrecords.py` through `refval.py fixtures`: an
`accepted` file gives no error and is byte-identical to the canonical encoding of its records; a
`refused` file gives exactly the listed reason and its first error is on the listed line. A reader
MUST accept the `rv` files and refuse the `ri` files with `R_RECORDS_FILE_INVALID`; the
`E_RECORDS_*` names are its reasons (experiment-v1 §8).

| Fixture | Expected | Case |
|---|---|---|
| [`rv01_reconfirmation_old_bytes`](records/rv01_reconfirmation_old_bytes.jsonl) | `accepted` | Three `reconfirmation` lines exactly as written before the `removed` and `image_count_changed` types existed (a `null` `new_id`, an `old_sha256` of `absent`); the bytes are unchanged and equal the canonical encoding. |
| [`rv02_image_count_changed`](records/rv02_image_count_changed.jsonl) | `accepted` | Two `image_count_changed` lines (RR1, RR3): one unit, and two units sorted by `id`, with a space inside a `source_folder`. |
| [`rv03_removed_every_entity`](records/rv03_removed_every_entity.jsonl) | `accepted` | `removed` lines of all four entities: the entity lines (condition, repeat, technical repeat) first, then each `unit` line (RR2); a key the entity does not name is `null`. |
| [`rv04_old_and_new_mixed`](records/rv04_old_and_new_mixed.jsonl) | `accepted` | One file mixing old and new records, as a records file grows: `reconfirmation`, `removed` and `image_count_changed` lines in append order. |
| [`rv05_non_ascii_unescaped`](records/rv05_non_ascii_unescaped.jsonl) | `accepted` | Non-ASCII text written unescaped (umlaut, accent, CJK, an astral character), a U+2028 inside a string that must not end the line, and an escaped quote, backslash and tab. |
| [`rv06_empty_file`](records/rv06_empty_file.jsonl) | `accepted` | Zero bytes: no record and no torn line (RF4). |
| [`ri01_missing_final_lf`](records/ri01_missing_final_lf.jsonl) | refused: `E_RECORDS_NO_FINAL_LF` (line 2) | The last line is a complete record without its LF (a torn append). |
| [`ri02_malformed_line`](records/ri02_malformed_line.jsonl) | refused: `E_RECORDS_NOT_JSON_OBJECT` (line 2) | A line cut in the middle of an object; the lines after it are checked too. |
| [`ri03_line_not_an_object`](records/ri03_line_not_an_object.jsonl) | refused: `E_RECORDS_NOT_JSON_OBJECT` (line 2) | A line that is a JSON array. |
| [`ri04_blank_line`](records/ri04_blank_line.jsonl) | refused: `E_RECORDS_NOT_JSON_OBJECT` (line 2) | An empty line between two records. |
| [`ri05_unknown_type`](records/ri05_unknown_type.jsonl) | refused: `E_RECORDS_UNKNOWN_TYPE` (line 2) | `type` is `renamed`, none of the three. |
| [`ri06_type_missing`](records/ri06_type_missing.jsonl) | refused: `E_RECORDS_UNKNOWN_TYPE` (line 2) | No `type` key. |
| [`ri07_key_missing`](records/ri07_key_missing.jsonl) | refused: `E_RECORDS_KEY_SET` (line 2) | An `image_count_changed` line without `recorded_on`. |
| [`ri08_key_of_another_type`](records/ri08_key_of_another_type.jsonl) | refused: `E_RECORDS_KEY_SET` (line 1) | A `reconfirmation` line that also carries `recorded_on`, a key of another type. |
| [`ri09_key_order`](records/ri09_key_order.jsonl) | refused: `E_RECORDS_KEY_ORDER` (line 2) | The keys of a `reconfirmation` in the order of the text, not in code point order. |
| [`ri10_nested_key_order`](records/ri10_nested_key_order.jsonl) | refused: `E_RECORDS_KEY_ORDER` (line 2) | Inside `technical_repeats`, `source_folder` before `id`. |
| [`ri11_non_ascii_escaped`](records/ri11_non_ascii_escaped.jsonl) | refused: `E_RECORDS_NON_ASCII_ESCAPED` (line 2) | `\u00e9` written for a non-ASCII character. |
| [`ri12_invalid_utf8`](records/ri12_invalid_utf8.jsonl) | refused: `E_RECORDS_ENCODING` (line 2) | The byte 0xE9 alone (Latin-1) in a line. |
| [`ri13_byte_order_mark`](records/ri13_byte_order_mark.jsonl) | refused: `E_RECORDS_ENCODING` (line 1) | The file starts with a byte-order mark (RF4). |
| [`ri14_duplicate_key`](records/ri14_duplicate_key.jsonl) | refused: `E_RECORDS_NOT_JSON_OBJECT` (line 2) | `confirmed_on` twice in one object; the second value would silently win. |
| [`ri15_removed_unit_with_null_name`](records/ri15_removed_unit_with_null_name.jsonl) | refused: `E_RECORDS_VALUE` (line 2) | A `unit` line with a `null` where the unit names all six keys. |
| [`ri16_removed_repeat_names_extra_key`](records/ri16_removed_repeat_names_extra_key.jsonl) | refused: `E_RECORDS_VALUE` (line 2) | A `repeat` line with a `condition_id` that is not `null`. |
| [`ri17_removed_entity_unknown`](records/ri17_removed_entity_unknown.jsonl) | refused: `E_RECORDS_VALUE` (line 2) | `entity` is `channel`. |
| [`ri18_image_count_zero`](records/ri18_image_count_zero.jsonl) | refused: `E_RECORDS_VALUE` (line 2) | `new_image_count` is 0 (at least 1). |
| [`ri19_image_count_float`](records/ri19_image_count_float.jsonl) | refused: `E_RECORDS_VALUE` (line 2) | `new_image_count` is `15.0`, not an integer. |
| [`ri20_date_impossible`](records/ri20_date_impossible.jsonl) | refused: `E_RECORDS_VALUE` (line 2) | `confirmed_on` is `2026-02-30`. |
| [`ri21_technical_repeats_unsorted`](records/ri21_technical_repeats_unsorted.jsonl) | refused: `E_RECORDS_VALUE` (line 2) | `technical_repeats` not sorted by `id`. |
| [`ri22_id_kind_unknown`](records/ri22_id_kind_unknown.jsonl) | refused: `E_RECORDS_VALUE` (line 2) | `id_kind` is `masks`. |
| [`ri23_bad_line_after_good_new_records`](records/ri23_bad_line_after_good_new_records.jsonl) | refused: `E_RECORDS_VALUE` (line 5) | Four valid lines of three types, then an `image_count_changed` line with `old_image_count` 0; the line after it is valid. |
| [`ri24_sha256_not_hex`](records/ri24_sha256_not_hex.jsonl) | refused: `E_RECORDS_VALUE` (line 2) | `new_sha256` is 64 upper-case `B` (lower-case hex, RF4). |
| [`ri25_compact_separators`](records/ri25_compact_separators.jsonl) | refused: `E_RECORDS_FORM` (line 2) | Key order and ASCII rules met, but no space after `,` and `:` (RF1). |
| [`ri26_crlf_line_ends`](records/ri26_crlf_line_ends.jsonl) | refused: `E_RECORDS_FORM` (line 1) | A CR before the LF (RF4). |
| [`ri27_optional_ascii_escape`](records/ri27_optional_ascii_escape.jsonl) | refused: `E_RECORDS_FORM` (line 2) | `\u0031` written for the ASCII character `1`: only the minimal escapes are allowed. |
| [`ri28_technical_repeats_empty`](records/ri28_technical_repeats_empty.jsonl) | refused: `E_RECORDS_VALUE` (line 2) | An `image_count_changed` line with `technical_repeats` `[]` (RF5). |
| [`ri29_image_count_unchanged`](records/ri29_image_count_unchanged.jsonl) | refused: `E_RECORDS_VALUE` (line 2) | `old_image_count` equals `new_image_count`: the record states no change (RF5). |
| [`ri30_extra_whitespace`](records/ri30_extra_whitespace.jsonl) | refused: `E_RECORDS_FORM` (line 2) | A space after `{` and a trailing space before the LF (RF1). |
| [`ri31_sha256_trailing_newline`](records/ri31_sha256_trailing_newline.jsonl) | refused: `E_RECORDS_VALUE` (line 2) | `new_sha256` is 64 lower-case hex digits followed by an escaped LF (`\n`): the line is canonical bytes, but the value is not 64 hex digits (RF4). Regression for a `$` anchor that matched before a final LF. |
