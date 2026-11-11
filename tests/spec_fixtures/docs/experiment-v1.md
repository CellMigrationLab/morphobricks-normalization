# MorphoBricks experiment description, schema version 1 (normative)

- **Status:** Normative specification, Stage 0 (6 November 2026). No reader implements it yet.
  The fixed-SHA audit of current consumers (roadmap S0.1) is done
  ([PR #5](https://github.com/CellMigrationLab/MorphoBricks-Spec/pull/5), audit
  [`s01-consumer-audit-2026-11-06.md`](audits/s01-consumer-audit-2026-11-06.md)); the owner answered
  its questions AQ1-AQ13 on 6 November 2026 ([ADR 0001](adr/0001-experiment-description-v1.md),
  third round) and this text applies them. A fourth decision of the same day (ADR 0001, fourth
  round: folder labels are never restricted) gives conditions and technical repeats a canonical ID
  separate from their source folder label (§4.6a). A statement about a consumer is as audited as that
  audit; it is not a conformance claim for any brick.
- **Decisions:** [ADR 0001](adr/0001-experiment-description-v1.md).
- **Fixtures:** [fixtures/README.md](../fixtures/README.md). **Conformance:**
  [conformance-plan.md](conformance-plan.md).

The key words MUST, MUST NOT, SHOULD and MAY are normative. "Reader" means any program that
parses the file (in v1, preprocessing is the only reference reader); "writer" means any program
that produces it.

## 1. Scope

The file `morphobricks.toml` describes **what one preprocessing experiment is**: conditions,
canonical channels, biological-repeat pairing, technical-repeat acquisition layouts, mask
semantics and an optional primary reference per condition. It MUST NOT contain file inventory,
fingerprints, calibration, exclusions, per-FOV statements, axes, brick names or routing, feature
requests, normalization, statistics, histogram or model settings. Those stay in
`calibration.csv`, the run manifest, analysis requests and MorphoCoverage configuration.

v1 analyses 2-D planes only. A Z stack is projected by a recorded preprocessing step
(preprocessing #176) before it is described; the file has no Z, axes or shared-plane field.
Time-lapses (T > 1) are allowed: the file has no T field, and the homogeneity of an imaging unit
(§8, `R_UNIT_NOT_HOMOGENEOUS`) compares what each axis means, never the number of time points.

## 2. File, encoding and parsing

1. The file name is `morphobricks.toml`, in the preprocessing output folder. It is written by the
   preprocessing notebook/CLI under the output folder's lock, atomically, and never by `run`.
   Readers MUST refuse a symlinked file (`R_FILE_SYMLINK`).
2. The bytes MUST be UTF-8 without a byte-order mark (`E_FILE_ENCODING`) and MUST be valid TOML
   1.0 (`E_TOML_SYNTAX`). A key declared twice, in any spelling TOML treats as the same key
   (bare vs quoted, table vs dotted vs inline), MUST be refused as `E_DUPLICATE_KEY`.
3. Readers MUST NOT convert types. A value whose TOML type differs from §4 is `E_TYPE`. In
   particular a TOML integer, float, boolean, date, time or datetime where text is required, a
   string where an integer or boolean is required, and a boolean where an integer is required
   (implementations whose booleans are integers MUST check the exact type).
4. Every table has a closed set of fields. A field not listed in §4 is `E_UNKNOWN_FIELD`, except
   the names in §4.7, which are `E_FORBIDDEN_FIELD`. There is no extension namespace.
5. Comments are allowed and carry no meaning. Writers do not preserve them (§9).
6. The SHA-256 of the exact bytes read is the file's provenance identity (run manifest). It is not
   a cache key (§9.3).
7. A reader has no source of rules other than the normative text and fixtures of the Spec commit it
   pins. It MUST NOT carry a fallback copy or older variant of the rules, and MUST NOT obtain any
   enumeration (in v1 only `representation`, §4.5) or vocabulary from outside the reader at run
   time. An enumeration it cannot resolve is a refusal, never free text.

## 3. Identities

### 3.1 Kinds of identity

| Identity | Where | Canonical or source | Rule set |
|---|---|---|---|
| Condition ID | key of `conditions` | Canonical (user-confirmed; proposed = the source folder label when allowed, §4.6a) | Label |
| Condition source folder | `conditions.<ID>.source_folder` | Source (condition folder label, or `single`; not an identity key) | Label |
| Canonical repeat ID | key of `repeats` | Canonical (user-confirmed; proposed = the source folder label when allowed, §4.6a) | Label |
| Source repeat | `source_repeat` | Source (biological-repeat folder label) | Label |
| Technical repeat ID | key of `technical_repeats` | Canonical (user-confirmed; proposed = the source folder label when allowed, §4.6a; or `single`) | Label |
| Technical-repeat source folder | `technical_repeats.<ID>.source_folder` | Source (technical-repeat folder label, or `single`; not an identity key) | Label |
| Image type | `image_type` | Declared by the user per imaging unit (§4.4b) | Enumeration |
| Channel ID | key of `channels` | Canonical | Token |
| Mask ID | key of `masks` | Canonical (object-set name) | Token + object set |
| Mask source type | `source_type` | Source (discovered mask type) | Label |
| Display name | `display_name` | Display only, never an identity | Non-empty text (§4.8) |

### 3.2 Rule sets

- **Label** (preprocessing decision D-e, `check_identity_text`, preprocessing v0.4.0 `bc7b872`): text;
  no leading or trailing whitespace (nothing is trimmed: refused as written); not `""`, `NA`, `nan`, `none`, `null`, `n/a` or `<NA>` in any
  case (`E_ID_TEXT`).
- **Token** (MorphoFeature `naming.py`, MorphoFeature v0.1.0 `182b373`): the Label rules, plus the
  regular expression `^[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*$` (`E_ID_SYNTAX`; this also excludes
  the feature-column separator `__`). The two checks are independent rules: an identity that breaks
  both is reported with both codes (§6, one code per broken rule).
- **Reserved list** (one list, shared by channel IDs and mask IDs; AQ4). A channel ID or a mask ID
  whose case-folded form (`str.casefold()`) is in the list is `E_ID_RESERVED`; `ring`, `Ring` and
  `RING` are all refused. An entry of `used_ids.channels` is refused the same way, since a listed
  ID reserves that ID (§4.9). The list is closed: a new entry needs a new `schema_version`.

  | Names | Source |
  |---|---|
  | `whole`, `cell`, `nucleus`, `cytoplasm`, `ring`, `expanded_nucleus` | MorphoRelate compartment names; MorphoFeature `naming.py:44-59` (`182b373`) |
  | `condition`, `biological_repeat`, `repeat`, `technical_repeat`, `position_id`, `image_file`, `image_path`, `mask_file`, `patch_file`, `patch_id`, `label_id`, `z`, `z0`, `z1`, `c`, `t` | MorphoRelate reserved column names, `registry/_expand.py:41-59` (MorphoRelate v0.1.1 `160ad23`; name rules `:68-115`) |

  The reserved list is not the forbidden-name list of §4.7, which is compared exactly and applies
  to field names and ID keys. A name in both (`z`) is reported by the earlier phase only. Neither
  list ever applies to a folder label: a `source_folder` or `source_repeat` value is never
  restricted by them (§4.6a).
- **Object set** (MorphoFeature `is_reserved_object_set`): in addition to the reserved list, a mask
  ID whose case-folded `_`-separated parts include `in` is `E_ID_RESERVED`.
- **RGB tokens:** `R`, `G` and `B` are the fixed channel IDs of RGB units (§4.4b); they are not on
  the reserved list and are valid channel IDs, with the extra rules of §4.4b.
- **`single`:** an identity of any kind whose case-folded form is `single` (a source folder
  included) MUST be exactly `single`. A technical repeat `single` MUST be the only technical
  repeat of its member; a member with `source_repeat = "single"` MUST be the only member of its
  condition in the file. A technical repeat ID is `single` if and only if its `source_folder` is
  `single` (`single` stands for "no technical-repeat folder level"; `E_SINGLE_MISUSE`;
  preprocessing contract §1). Conditions follow the same rule: a condition ID is `single` if and
  only if its `source_folder` is `single` (`single` stands for "no condition folder level", an
  experiment folder that is itself the leaf), and a condition whose `source_folder` is `single`
  MUST be the only condition in the file (`E_SINGLE_MISUSE`).

### 3.3 Normalization, uniqueness and case

- **NFC.** Identities (the keys and identity fields of §3.1, and every entry of `used_ids`) are
  Unicode NFC. A writer MUST emit NFC. A reader MUST refuse an identity that is not NFC
  (`E_ID_NOT_NFC`, reported for every such identity, including `reference` values, members' keys,
  `source_folder` values, unit `channels` entries and `source_channels` entries). The check of a reference is NFC only and
  belongs to phase 4, so a non-NFC channel reference is `E_ID_NOT_NFC` and never
  `E_UNKNOWN_CHANNEL` (AQ13). A reader never transforms an identity: no normalization, trimming, case folding, parsing
  or repair of any kind, and so it never resolves a reference through a transformed spelling.
  Display names are not identities and are not checked (§4.8). A folder label read from disk that
  is not NFC is refused by preprocessing (`R_FOLDER_NOT_NFC`), never normalized and never renamed;
  two folders equal after NFC are therefore also refused, because one of them is not NFC. The
  stored `source_folder` is the label exactly as read, so it is NFC by the same rule.
- Identities are compared exactly, code point by code point.
- **No repair.** No reader or writer ever repairs, re-cases, pads, trims or renumbers an
  identity. A malformed, colliding or non-NFC ID is refused with its code and left as written.
  There is no ID allocator: IDs have no width, prefix or number rule (`R999`, `R1000` and `cl001`
  are valid and stay as they are).
- Within each of these scopes, two identities that differ after NFC but whose `str.casefold()`
  forms are equal are `E_ID_CASE_COLLISION`: condition IDs; condition source folders; repeat IDs;
  source repeats over the whole file; technical repeat IDs over the whole file; technical-repeat
  source folders over the whole file; channel IDs; mask IDs; mask source types.
  (Preprocessing `check_identity_case`: a label is spelled one way in the whole run.) Case folding
  is never applied when resolving a reference (`ctrl` does not resolve to `CTRL`).
- Identities are never parsed. `rep1`, `2026-01-01`, `012` and `acquisition_2` are opaque text;
  no number, date, order or meaning is derived from them.
- **Immutability and no reuse.** Once a repeat, condition, technical repeat or channel is
  confirmed, its ID never changes (§4.6) and is never reused after removal (§4.9). Only
  `display_name` may change, at any time. The `source_folder` of an existing ID is not freely
  editable: a change is a reassignment (§4.6).

### 3.4 Keys downstream: the source label stays, the canonical repeat ID is added (AQ5)

- The existing FOV key (`condition`, `biological_repeat`, `technical_repeat`, `position_id`) is
  **unchanged**. `biological_repeat` in it stays the **source repeat**, the source folder label.
  FOV, patch and object identities, fitted-scaling keys, caches and exports keep their current
  spelling; a consumer MUST NOT replace the source label by the canonical repeat ID in any of
  them.
- The canonical repeat ID is **added** next to the key, as the field `repeat_group` (the column
  name of the derived `experiment_design.csv`, §11). Grouping and pairing MUST use `repeat_group`: pooling of
  normalization and scaling, pairing of conditions and families in MorphoCoverage, and any
  statement about biological N. A consumer MUST NOT group or pair by `biological_repeat` when a
  description exists, and MUST NOT fall back to folder-name grouping.
- `repeat_group` is a function of the FOV key (a member belongs to exactly one repeat,
  `E_SOURCE_REPEAT_REUSED`), so adding it creates no new identity: two FOVs with equal keys have
  equal `repeat_group`. Products that pool over a repeat key their caches on the `repeats` section
  digest (§9.3); products of one FOV keep keying on the FOV key.
- Consequence: `rep1` of `CTRL` and `rep1` of `DrugA` may be different repeats (`v30`) and the same
  source label may be in two repeats under different conditions; the key stays unique because it
  includes the condition.
- **Conditions and technical repeats follow the same pattern (fourth round).** In the existing
  FOV key, `condition` and `technical_repeat` stay the **source folder labels**
  (`conditions.<ID>.source_folder`, `technical_repeats.<ID>.source_folder`); a consumer MUST NOT
  replace them by the canonical IDs in FOV, patch or object identities, fitted-scaling keys,
  caches or exports. The canonical IDs are **added** next to the key, beside `repeat_group`, under
  the names `condition_id` and `technical_repeat_id`, which Stage 1 fixes for every consumer.
  Each is a function of the FOV key (one folder, one ID, §4.6a), so adding them creates no new
  identity. Everything that names a condition by identity (`reference`, families, pairing,
  contrasts, `used_ids`) uses the canonical condition ID, and a consumer MUST NOT group, pair or
  reference by the source folder label when a description exists.

## 4. Tables and fields

Types are TOML types. "Text" is a TOML string. "Date" is a TOML **string** `YYYY-MM-DD` naming a
real calendar day (`E_DATE_FORMAT`); a TOML date literal is `E_TYPE`. "Non-empty text" refuses
`""` and whitespace-only strings (`E_TEXT_EMPTY`). "Req" = required; a missing required field is
`E_MISSING_FIELD`. An empty table or array where at least one entry is required is `E_EMPTY`.

### 4.1 Top level

| Key | Type | Req | Rule |
|---|---|---|---|
| `schema` | text | yes | Exactly `morphobricks-experiment` (`E_SCHEMA_ID`) |
| `schema_version` | integer | yes | Exactly `1` (missing: `E_SCHEMA_VERSION_MISSING`; other integer: `E_SCHEMA_VERSION_UNSUPPORTED`) |
| `conditions` | table of condition | yes | ≥ 1 entry |
| `channels` | table of channel | yes | ≥ 1 entry |
| `repeats` | table of repeat | yes | ≥ 1 entry |
| `masks` | table of mask | no | May be absent or empty |
| `used_ids` | table of ID lists | yes | Append-only registry of every ID ever confirmed, per kind (§4.9) |

### 4.2 `conditions.<condition ID>`

| Key | Type | Req | Rule |
|---|---|---|---|
| `source_folder` | text | yes | Exact condition folder label as read from disk, or `single` when the experiment has no condition folder level (Label rules, NFC); a source value, **not an identity key**; not restricted by the reserved or forbidden names (§4.6a) |
| `display_name` | non-empty text | no | Display only (§4.8); MAY be the folder label |
| `reference` | text | no | One condition ID: the primary reference (§5) |
| `target` | non-empty text | no | Descriptive plain string, e.g. `SRC` (§4.2a) |
| `treatment_key` | non-empty text | no | Descriptive plain string, e.g. the same perturbation under different backgrounds (§4.2a) |

Every condition MUST be the condition of at least one member (`E_CONDITION_WITHOUT_MEMBERS`).

**§4.2a Descriptive only.** `target` and `treatment_key` are descriptive free text, written and
shown as typed. They are not identities: no Label or Token rule, no case-collision scope, no
uniqueness (two conditions may carry the same value, or values differing only in case). A consumer
MUST NOT use either for grouping, pairing, joins, contrasts, family definitions or any cache key
(§9.3 omits them from every digest). A future use as a grouping key needs a new `schema_version`
that makes it an identity.

**Prefill only (AQ3).** A consumer MAY offer the values as a *suggestion* to prefill an input of
its own, for example MorphoCoverage's reagent grouping. The suggestion is shown to the user, who
confirms or edits it **inside that consumer**; the confirmed grouping is the consumer's own input,
recorded in its own configuration, and is the only thing the analysis reads. An unconfirmed
prefill is never used, and a later edit of `target` or `treatment_key` in the file changes no
confirmed grouping, no digest and no cache key.

### 4.3 `channels.<channel ID>`

| Key | Type | Req | Rule |
|---|---|---|---|
| `display_name` | non-empty text | no | Display only (§4.8), e.g. `DAPI` |
| `header_names` | array of non-empty text | no (default `[]`) | Header channel names the user confirmed as evidence for this channel; stored as typed (not identities: no Label, Token or NFC check; outer whitespace is part of the name); no duplicates, where two entries equal after NFC are duplicates (`E_DUPLICATE_VALUE`); order not semantic. Not allowed for `R`, `G`, `B` (§4.4b) |

A channel that appears in no technical repeat's `channels` is **valid**; the reader reports the warning `W_UNUSED_CHANNEL`. A declared but unused channel is never an error, and never an implicit zero or default.

Header evidence (enforced by preprocessing, §8): a non-empty header channel name read at source
index *i* of an image in a unit MUST match a member of `header_names` of the channel at position
*i* of that unit's list, else `R_HEADER_CONTRADICTION`. **Matching rule (AQ7):** the name read from
the header and the `header_names` entry are each converted to NFC, and the results are compared
code point by code point. Nothing else is applied: **no trimming** (outer whitespace is part of
the name, so a header ` DAPI` matches only an entry ` DAPI`), no case folding, no similarity. The
NFC form is a comparison key only; it is never stored, and the header name is recorded as read.
Any other spelling needs a user-confirmed alias (a new `header_names` entry). Preprocessing
therefore MUST stop trimming header channel names (it trims today, `channel_names_trimmed`,
preprocessing `zarr_input.py:68`): this is a Stage 1 change and the raw name is what is recorded
and compared. This holds even when the header
name equals the channel ID or display name: aliases are only ever user-confirmed, never inferred from
name similarity, case or pixels. An image without header names needs no evidence. The same header
name MAY be evidence for different channels (fluorophores are reused across acquisitions).

### 4.4 `repeats.<canonical repeat ID>`

| Key | Type | Req | Rule |
|---|---|---|---|
| `display_name` | non-empty text | no | Display only (§4.8) |
| `confirmed` | boolean | yes | MUST be `true` in a published file (`E_REPEAT_UNCONFIRMED`): the N35 answer "prepared and acquired under the same circumstances" |
| `confirmed_on` | date | yes | Date of the user's confirmation (§4.4a) |
| `members` | table of member | yes | ≥ 1 entry |

`repeats.<R>.members.<condition ID>` (one member = one source biological-repeat folder of one
condition; at most one per condition per repeat, guaranteed by TOML key uniqueness):

| Key | Type | Req | Rule |
|---|---|---|---|
| `source_repeat` | text | yes | Exact source folder label (Label rules) |
| `image_count` | integer ≥ 1 | yes | Images to analyse at confirmation (N35 evidence; `E_VALUE_RANGE`) |
| `technical_repeats` | table of technical repeat | yes | ≥ 1 entry |

`repeats.<R>.members.<C>.technical_repeats.<technical repeat ID>` (one imaging unit):

| Key | Type | Req | Rule |
|---|---|---|---|
| `source_folder` | text | yes | Exact technical-repeat folder label as read from disk, or `single` when the member has no such folder level (Label rules, NFC); a source value, **not an identity key**; not restricted by the reserved or forbidden names (§4.6a) |
| `display_name` | non-empty text | no | Display only (§4.8); MAY be the folder label |
| `channels` | array of channel IDs | yes | ≥ 1 entry; ordered: entry *i* is source channel C*i*; every entry declared (`E_UNKNOWN_CHANNEL`); no repeats (`E_DUPLICATE_CHANNEL_IN_UNIT`) |
| `image_type` | text | no | Only the value `rgb` (`E_ENUM`: exact, lower case); absent = an ordinary channel image. User-declared, never inferred (§4.4b) |

**§4.4b RGB imaging units (AQ6; supersedes the D8 refusal).** RGB (interleaved colour) is an
accepted, **declared** image type, per imaging unit:

- A unit with `image_type = "rgb"` MUST list exactly `["R", "G", "B"]`, in this order, as its
  `channels` (source samples 0, 1, 2): any other count, order, case or IDs is
  `E_RGB_UNIT_CHANNELS`.
- `R`, `G`, `B` are fixed tokens. A unit without `image_type = "rgb"` MUST NOT list a channel whose
  case-folded ID is `r`, `g` or `b` (`E_RGB_TOKEN_MISUSE`).
- They have **no aliases**: a channel whose case-folded ID is `r`, `g` or `b` MUST NOT have a
  non-empty `header_names` (`E_RGB_ALIAS`). `display_name` is allowed (display only).
- The image type is declared by the user and is **never inferred** from pixels, dtype, channel
  count or file name. Preprocessing checks it against the image **header metadata** only (the
  colour interpretation and samples per pixel that the format states, for example TIFF
  photometric interpretation or OME `Interleaved`), in both directions: a unit declared `rgb` whose
  header says it is not, or a unit not declared `rgb` whose header says it is, is
  `R_IMAGE_TYPE_CONTRADICTS_HEADER`. A format whose header states no colour interpretation cannot
  contradict the declaration, which then stands on user confirmation and is recorded in the
  manifest as not header-verified.
- The spec gives RGB **no scientific meaning**: no grey conversion, luminance, stain separation
  or joint-channel rule. Each brick owns the meaning and MUST declare whether it supports RGB;
  undeclared means unsupported. An analysis request that includes an RGB unit's channels for a
  brick without RGB support is refused at analysis (`A_RGB_NOT_SUPPORTED`, §8), never converted or
  split into grey channels silently.
- **Machine-readable declaration (`rgb_support`).** A brick that publishes a provider manifest
  declares RGB support there, as the optional top-level key `rgb_support`, whose value is a
  boolean. `true` declares support (the brick owns what RGB means for it); `false` and an absent key
  both mean unsupported, so a manifest written before this key existed is read as unsupported and
  needs no edit. A value that is not a boolean is a manifest error: it is never read as `true`. A
  brick without a provider manifest states the same declaration in its documented capabilities.
  The key declares support only; it gives RGB no meaning (the bullet above), and it does not
  change a description: `image_type = "rgb"` is still user-declared and never inferred.
- The refusal applies to a **described** run (one opened from a published description). There is
  no other run: a run without a published description is refused with `A_DESCRIPTION_REQUIRED`
  before any capability is checked (ADR 0002 AR10, which supersedes owner decision RC1's clause
  that such runs keep working with a warning).
- A file MAY hold RGB and non-RGB units; their lists differ, so that layout is heterogeneous
  (`R_HETEROGENEOUS_LAYOUT_NOT_ENABLED` in Stage 1-2). A unit's `image_type` is part of the
  `repeats` section digest (§9.3).

**§4.4a `confirmed_on` is a user fact.** It is the calendar date of the user's confirmation
action. A writer MUST NOT default, infer or refresh it (not from the write time, the read time,
file times, another repeat or a template); a reader supplies no default (absent is
`E_MISSING_FIELD`). A reconciliation that does not reopen the N35 confirmation (§10) MUST keep
`confirmed_on` of every unchanged repeat byte-identical; a candidate that changes it without a
reopening and a new user confirmation is refused (`R_CONFIRMED_ON_CHANGED`).

Repeat rules:

- The member key MUST be a declared condition ID (`E_UNKNOWN_CONDITION`).
- A source folder `(condition, source_repeat)` MUST belong to exactly one repeat
  (`E_SOURCE_REPEAT_REUSED`).
- If a canonical repeat ID R equals, ignoring case, the `source_repeat` S of a member, that member
  MUST belong to R (`E_REPEAT_ID_SHADOWS_SOURCE`). This forbids swapped labels such as repeat
  `rep1` holding folder `rep2` while repeat `rep2` holds folder `rep1`.
- One folder, one ID (§4.6a): a condition `source_folder` belongs to exactly one condition ID and
  a technical-repeat `source_folder` to exactly one technical repeat ID over the whole file
  (`E_SOURCE_FOLDER_REUSED`); one technical repeat ID carries the same `source_folder` in every
  unit that declares it (`E_SOURCE_FOLDER_CONFLICT`).
- A repeat MAY lack some conditions (warning `W_REPEAT_MISSING_CONDITIONS`).
- Technical repeats inside one biological repeat MAY have different channel counts and orders.
  They remain one biological repeat; they never create additional biological replicates.

### 4.5 `masks.<mask ID>`

| Key | Type | Req | Rule |
|---|---|---|---|
| `source_type` | text | yes | Exact discovered mask type (Label rules); one mask per source type (`E_MASK_SOURCE_TYPE_REUSED`) |
| `representation` | text | yes | `instances` or `binary` (`E_ENUM`); user-confirmed, never inferred from dtype, values or names. A mask declared `binary` with more than two distinct values is refused by preprocessing's mask step (`R_MASK_CONTRADICTS_REPRESENTATION`, §4.5a); the user must declare `instances` |
| `source_channels` | array of channel IDs | no | Provenance only (what the mask was derived from); not a restriction on measured channels; each declared (`E_UNKNOWN_CHANNEL`); no duplicates (`E_DUPLICATE_VALUE`); order not semantic |
| `display_name` | non-empty text | no | Display only (§4.8) |

Mask files, pairing, completeness and fingerprints stay in the run manifest. A declared mask with
files for none, some or all FOVs is valid (progressive segmentation, 0/1000 → 1000/1000).

**§4.5a When the representation is checked (AQ10).** `R_MASK_CONTRADICTS_REPRESENTATION` needs
mask pixels. It is checked **in the preprocessing mask step** (the step that reads mask pixels),
for every mask file as it is added: for a progressive mask, **for each new batch**, on the new
files. Publishing `morphobricks.toml` MUST NOT read mask pixels; it records, per mask and file
set in the run manifest, the status `checked` (with the checked files and their fingerprints) or
`not yet checked`. A run that uses a mask file whose status is `not yet checked` MUST refuse
(`A_MASK_UNCHECKED`, §8); it never checks implicitly and never assumes. A new batch of files is
`not yet checked` until the mask step has read it. Pixels cannot prove `instances` (an integer
mask holding only 0 and 1 is accepted as declared).

**§4.5b Grouped (`parent_labels`) masks are out of scope in v1 (AQ8).** MorphoRelate's third
representation `parent_labels` (preprocessing kind `grouped`) has no v1 value:
`representation = "parent_labels"` is `E_ENUM`. A grouped mask cannot be declared `instances`
(wrong objects) or `binary` (more than two values: `R_MASK_CONTRADICTS_REPRESENTATION`), so it
is not described, and not analysed through a description, in v1. A design for it follows with a
later `schema_version` (v1.1); nothing here predicts its form.

**§4.5c A binary mask is never an object set by default (AQ9).** A request that uses a `binary`
mask as an object set is `A_MASK_NOT_INSTANCES`. The only way around it is an **explicit,
per-request, recorded conversion** to instances chosen in the request, for example CellPatcher
`binary_mask_policy="connected_components"`: it is recorded in the run manifest with the mask ID,
the method and its parameters, applies to that request only, and is never automatic, never a
default and never stored in `morphobricks.toml`, where `representation` stays `binary`
(`binary_mask_policy` and `mask_conversion` are `E_FORBIDDEN_FIELD`, §4.7). The numerical method
stays in the brick that owns it.

### 4.6 Immutable IDs

- Once confirmed, the ID of a repeat, condition, technical repeat or channel MUST NOT change.
  There is no rename and no retired-ID table. A rename in the file would be a different ID for the
  same data.
- **What a changed entity is (AQ1).** An entity is what the published file says an ID holds: for a
  repeat, its member folders `(condition, source_repeat)`; for a condition, its `source_folder`
  and its source repeat folders over all repeats; for a technical repeat, its `source_folder` and
  the members `(condition, source_repeat)` that declare it; for a channel, its positions `(condition, source_repeat, technical
  repeat, index)` and its `header_names`. An entity is *changed* only when something it held in
  the published file is **removed or reassigned**. **Adding** does not change it: a new member
  folder joining an existing repeat, a new repeat folder of an existing condition, a new unit
  of an existing technical repeat ID, a new unit or position of a channel, a new alias, is not
  `R_CONFIRMED_ID_CHANGED`. (It may reopen the N35
  confirmation, §10; reopening is not an ID change.)
- Preprocessing MUST refuse, as `R_CONFIRMED_ID_CHANGED`, a candidate file (compared with the
  published file it was drafted from, §10) in which (a) a repeat with the same member folders
  `(condition, source_repeat)` carries another ID, (b) a channel's ID is replaced by another ID at
  every unit position it held with the same header evidence, (b2) a condition or technical repeat
  with the same `source_folder` carries another ID, or (c) an ID that is declared in both
  files names a changed entity (some member folder, position or alias it held is removed or
  reassigned, or its `source_folder` is another folder). Removing or reassigning is accepted only with an explicit, recorded user
  confirmation (the same record as for a mistyped ID, below); the confirmation of a removal that
  §10 requires is that record, and every removal is also written as a `removed` record (below). An ID that was removed and comes back is `R_USED_ID_REUSED`
  (§4.9). A condition or technical repeat ID is not its folder label (§4.6a): a folder that is
  renamed or replaced on disk under an existing ID is a removed folder plus a new pending one
  (§7), and giving the existing ID the new `source_folder` is a reassignment under (c).
- A mistyped ID is fixed only by an explicit, recorded re-confirmation: a user action that
  writes a new file and records old ID, new ID, date and both file SHA-256. The record is made
  at publication, which happens outside a run: preprocessing appends it, under the output
  folder's publication lock (§10), to the append-only file `morphobricks.records.jsonl` next to
  `morphobricks.toml`, and copies the records unchanged into the manifest of the next run. The
  records file is never rewritten or truncated, and nothing about a re-confirmation is stored in
  `morphobricks.toml`. Products keyed on that ID are invalidated.
- **Encoding of the records file (owner decisions RF1, RF4).** UTF-8 without a byte-order mark,
  one JSON object per line, each line ending in LF (a CR before the LF is not allowed), keys in
  code point order at every level (the objects inside `technical_repeats` included), non-ASCII
  characters unescaped, the separators `, ` between members and `: ` after a key, and no other
  whitespace outside strings. A string uses only the escapes `\"`, `\\`, `\b`, `\f`, `\n`, `\r`,
  `\t` and, for the other control characters U+0000 to U+001F, `\u00xx` with lower-case hex; no
  other character is escaped. A file of zero bytes holds no record and is accepted. The key
  `type` tells the record types
  apart: `"reconfirmation"`, `"image_count_changed"` and `"removed"`. Each type has its own fixed
  key set; a line of one type carries no key of another. The first type is the only one written
  before this addition, and its lines are unchanged. Keys of `reconfirmation`: `type`,
  `id_kind` (`conditions`, `channels`, `repeats` or `technical_repeats`), `old_id`, `new_id`
  (`null` when the same ID lost or was reassigned something), `confirmed_on` (`YYYY-MM-DD`, the
  date of the user's action), `old_sha256` (of the published file replaced, or `absent`) and
  `new_sha256` (of the file written). A SHA-256 is 64 lower-case hex digits. The records are
  appended after the file is replaced.
- **Refusal of a records file (owner decisions RF2, RF3, RF4, RF5; `R_RECORDS_FILE_INVALID`).** A
  reader refuses the whole file, before anything is published, when any of these holds (the reason
  is given in parentheses, one file may have several): the bytes are not UTF-8 or start with a
  byte-order mark (`E_RECORDS_ENCODING`); the last line lacks its LF (`E_RECORDS_NO_FINAL_LF`); a
  line is not a JSON object, including an empty line and an object with a key twice
  (`E_RECORDS_NOT_JSON_OBJECT`); `type` is missing or none of the three (`E_RECORDS_UNKNOWN_TYPE`);
  the keys are not exactly the key set of the type (`E_RECORDS_KEY_SET`); keys are not in code point
  order at every level (`E_RECORDS_KEY_ORDER`); a non-ASCII character is written as an escape
  (`E_RECORDS_NON_ASCII_ESCAPED`); a value is invalid: an enumeration (`id_kind`, `entity`) outside
  its values, a date that is not a real `YYYY-MM-DD`, a SHA-256 that is not 64 lower-case hex
  digits (`old_sha256` may also be `absent`), a key that must be `null` (or must not be) in a
  `removed` line, a count that is not an integer of at least 1, `old_image_count` equal to
  `new_image_count`, `technical_repeats` empty or not sorted by `id` (`E_RECORDS_VALUE`); or the
  line differs from the canonical bytes above in anything else, whitespace, a CR, an optional
  escape (`E_RECORDS_FORM`). The reasons are named by the reference validator
  (`tools/audit/refrecords.py`); the code is the Spec's. The record values that name IDs and
  folder labels are text; the Spec states no further rule for them here.
- **Record of a removal (`removed`, owner decision RR2).** Written for **every** removal of an
  imaging unit, technical repeat, repeat or condition that the user confirms (§10), including a
  removal whose ID is held nowhere else afterwards; it is written in addition to a
  `reconfirmation` where §4.6 requires one. Keys: `type`, `entity` (`unit`, `technical_repeat`,
  `repeat` or `condition`), `condition_id`, `condition_source_folder`, `repeat_id`,
  `source_repeat`, `technical_repeat_id`, `technical_repeat_source_folder`, and `confirmed_on`
  (`YYYY-MM-DD`, the date of the user's confirmation; a `removed` line is itself that
  confirmation). A key that the entity does not name is `null`: a `unit` (one technical repeat
  entry of one member of one repeat) names all six; a `technical_repeat` names its two keys; a
  `repeat` its `repeat_id`; a `condition` its two condition keys. The source folder labels are the
  exact labels of the removed published file. A removed member folder is written as one `unit`
  record per imaging unit it held, and an entity that held units is followed by the `unit`
  records of those units, so every removed folder label appears in a record. Removing an entity
  never removes its ID from `used_ids` (§4.9).
- **Record of an image-count change (`image_count_changed`, owner decisions RR1 and RR3).**
  Written for **every** confirmed member whose `image_count` changes at a publication, including
  a publication that also changes something else (a removal, a new member, an ID correction,
  §10); the other changes do not suppress it and it suppresses none of their records. It needs no
  confirmation and reopens nothing: a count change alone is not a reopening, and
  `R_CONFIRMED_ON_CHANGED` (§4.4a) is unchanged. Keys: `type`, `condition_id`,
  `condition_source_folder`, `repeat_id`, `source_repeat`, `technical_repeats` (the member's
  imaging units as a non-empty array of objects `{"id", "source_folder"}`, sorted by `id` in code
  point order), `old_image_count`, `new_image_count` (integers, each at least 1, and different:
  the record states a change) and `recorded_on`
  (`YYYY-MM-DD`, the date the change was recorded at publication; it is not a confirmation and is
  never copied into `confirmed_on`).
- **Order and content of the `removed` lines of one publication (RR2, clarified with RR3).** Each
  vanished ID is one entity line; the `unit` lines of the units those entities held are written
  once, after all entity lines of the publication, each unit once. An entity takes the
  `confirmed_on` date of the first removal answer that touches it. When a removal is combined
  with an ID correction in the same publication, the lines carry the IDs as published (before the
  correction).
- **Order of the lines of one publication (owner decision RF6).** Preprocessing writes them as one
  append: the `reconfirmation` lines (by `id_kind`, `old_id`, `new_id`, `confirmed_on`), then the
  `removed` lines (entity lines, then `unit` lines, above), then the `image_count_changed` lines
  by `condition_id`, then `repeat_id`. The file has no publication boundary, so a reader of the
  file alone cannot check this order or that each unit is written once; the writer tests of
  scenarios P34 and P35 check it, and a file that breaks it is not refused for that.
- **A case-only correction needs a new ID (AQ2).** `ACTIN` cannot be corrected to `Actin`: the
  retired spelling stays in `used_ids` and the new one collides with it
  (`E_ID_CASE_COLLISION`, §4.9). The correction is a new, non-colliding ID (for example
  `ActinStain`) with the recorded re-confirmation above, and `display_name` (here `Actin`) carries
  the appearance. The old spelling stays listed and is never reused.
- The friendly name of any entity is `display_name` (§4.8), not the ID.
- **Publication reports a set (AQ2).** When several `R_*` refusals apply to one candidate,
  preprocessing reports **every** applicable code, not the first (`t02` is
  `R_CONFIRMED_ID_CHANGED` and `R_USED_ID_DROPPED`).

### 4.6a Folder labels and canonical IDs (fourth round)

Conditions and technical repeats are identified exactly like biological repeats: the key is a
**canonical ID** (immutable, user-confirmed, the only thing identity uses), and the folder it was
discovered under is recorded separately in the required field `source_folder`.

- **`source_folder` is a source value, not an identity.** It holds the folder label exactly as
  read from disk (Label rules, NFC; never trimmed, normalized or renamed). It is never a table
  key, a `reference` value, a member key or a `used_ids` entry, and no reader resolves an ID
  through it.
- **Folder labels are never restricted by the reserved list (§3.2) or the forbidden names
  (§4.7).** Any folder label that satisfies the Label rules and NFC may be a `source_folder`,
  including `control`, `files` or `z`. The restrictions stay on the **IDs** (keys), where the
  forbidden names would collide with fields.
- **One folder, one ID.** Over the whole file a condition `source_folder` names exactly one
  condition ID, and a technical-repeat `source_folder` exactly one technical repeat ID
  (`E_SOURCE_FOLDER_REUSED` when two IDs of one kind share a folder; `E_SOURCE_FOLDER_CONFLICT`
  when one technical repeat ID is declared with two different folders). A condition has one
  folder; a technical repeat ID has the same folder in every unit that declares it. Folders that
  differ only in case are `E_ID_CASE_COLLISION` (§3.3), as before.
- **Proposed ID.** When preprocessing meets a new folder it proposes the folder label as the ID
  if that is allowed: it passes the Label rules, is not an exact forbidden name (§4.7), and does
  not collide (exactly or by case folding) with a declared or `used_ids` ID of its kind. If it
  is not allowed (for example the folder `control`), preprocessing proposes a different ID that
  is allowed, and the user confirms it or types another; the user MAY also set `display_name` to
  the folder label. A proposal is never adopted unconfirmed, and no ID is repaired or derived by
  a reader (§3.3).
- **Repeat IDs follow the same rule.** For a new biological-repeat folder the proposed repeat ID is
  the folder label if it is allowed in the same sense (Label rules, not an exact forbidden name,
  no exact or case-folded collision with a declared or `used_ids` repeat ID). If it is not
  allowed (for example the folder `z`), preprocessing makes **no proposal**: it never repairs or
  derives an ID from the label (§3.3), and the user names the repeat ID. The reader validates what
  the user typed like any ID; `source_repeat` keeps the folder label.
- **Existing ID, other folder.** Giving an existing condition or technical repeat ID another
  `source_folder` is a reassignment (§4.6): `R_CONFIRMED_ID_CHANGED` without the recorded user
  confirmation. A new folder next to the existing ones is a new pending member or unit (§10).
- **`single`.** The technical repeat `single` has `source_folder = "single"`: the label denotes
  "no technical-repeat folder level", not a folder (§3.2). The same holds for a condition: the
  condition `single` has `source_folder = "single"`, which denotes "no condition folder level" (an
  experiment folder that is itself the leaf, with the member `source_repeat = "single"` and the
  technical repeat `single`), and it is the only condition of the file (§3.2).
- A folder label that breaks the Label rules (for example `NA`) is still refused at discovery as
  today; this decision adds no new code for it.

### 4.7 Forbidden fields

These names are refused as `E_FORBIDDEN_FIELD` in **any** table, with a message naming where the
information belongs, and they apply both to **field names** and to **ID keys** (the key of a
condition, channel, repeat, member, technical repeat or mask; AQ13). The comparison is exact and
case-sensitive. A forbidden name is reported once, at its name; its value is not examined. A
source folder label equal to one of these names (a condition folder `control`, a technical-repeat
folder `files`) cannot be used as the **ID** (the key) but **is** describable: it is stored as the
`source_folder` value, which is never restricted (§4.6a), and the user confirms another ID
(`ctrl`, `acq1`) that preprocessing proposes. The folder is never renamed on disk. A
`source_repeat` value is not a key and was never affected.

| Field | Belongs to |
|---|---|
| `axes`, `shared`, `plane_scope`, `z`, `projection` | Image/manifest records; v1 is 2-D only (preprocessing #173, #176) |
| `pixel_size`, `calibration` | `calibration.csv` |
| `fovs`, `files`, `paths`, `overrides`, `exclude`, `fingerprints` | Run manifest / preprocessing inputs; no per-FOV statements |
| `bricks`, `routes`, `features`, `normalization`, `statistics`, `contrasts` | Analysis requests / brick and MorphoCoverage configuration |
| `binary_mask_policy`, `mask_conversion` | The analysis request that records a per-request binary-to-instances conversion (§4.5c) |
| `control`, `roles`, `experiment` | Derived from `reference` / analysis requests / MorphoCoverage FDR families |
| `source_index`, `channel_count` | Implied by the order and length of `channels` |
| `retired_repeats`, `replaced_by` | None: IDs are immutable (§4.6); the only registry is `used_ids` (§4.9), IDs only and never a mapping |
| `labchronicle`, `provenance`, `source_database` | Run manifest provenance (preprocessing, §9.3): lab-tool links are never part of this file |

### 4.8 Display names

`display_name` (conditions, repeats, technical repeats, channels, masks) is for display only (reports, notebooks,
plots). It MUST NOT be used for identity, joins, pairing, references or cache keys, and changing
or adding it invalidates nothing (§9.3). A consumer MUST NOT accept a display name where an ID is
expected, even when the two strings are equal and even when the display name is unique: a
`reference`, member key or unit channel entry that names a display name is refused with the code
of an unknown ID (`E_REFERENCE_UNKNOWN`, `E_UNKNOWN_CONDITION`, `E_UNKNOWN_CHANNEL`).

Display names are kept exactly as typed: never normalized, trimmed or case-changed on read or
write. They are unique within one kind (conditions, repeats, technical repeats, channels, masks;
the same string may appear in two kinds, and a display name may equal a `source_folder`): two
display names of one kind whose keys are equal are `E_DISPLAY_NAME_COLLISION`; for technical
repeats the names of one technical repeat ID in several units never collide with each other, only
with those of another ID, and need not agree; where the key of a name *x* is `NFC(casefold(NFC(x)))` with
`str.casefold()` (so `Control` and `control`, `Straße` and `STRASSE`, decomposed and composed
`Café` collide). The key is used for the comparison only and is never stored or used for lookup.

### 4.9 Used IDs

`used_ids` is an append-only registry of IDs, per kind. It holds IDs only: never a mapping, a
date or a reason.

| Key | Type | Req | Rule |
|---|---|---|---|
| `conditions` | array of text | yes | Every condition ID ever confirmed |
| `channels` | array of text | yes | Every channel ID ever confirmed |
| `repeats` | array of text | yes | Every canonical repeat ID ever confirmed |
| `technical_repeats` | array of text | yes | Every canonical technical repeat ID ever confirmed (`single` included) |

- Entries follow the identity rules of their kind (Label for conditions, repeats and technical
  repeats, Token for channels; NFC, §3.3), have no duplicates (`E_DUPLICATE_VALUE`), and share the case-collision
  scope of the kind's declared IDs (`E_ID_CASE_COLLISION`): a retired `ctrl` cannot come back as
  `CTRL`. At least one entry per kind (`E_EMPTY`).
- **Reserved names in `used_ids.channels`** (owner decision UR1, 7 November 2026). A channel entry
  follows the reserved list of §3.2 like a declared channel ID: an entry whose case-folded form is
  in the list is `E_ID_RESERVED`. A listed ID reserves that ID, so a name no channel may bear
  cannot be reserved by being listed. The registry has no list for mask IDs (it holds the four
  kinds above), so the list applies to channels only; the other three kinds are unchanged.
- Source folder labels are not IDs and are never listed. Every declared ID of a kind MUST be
  listed (`E_USED_ID_MISSING`). A listed ID that is not
  declared is a **retired** ID: its entity was removed.
- Publication MUST refuse a candidate that (a) omits an ID listed in the published file it was
  drafted from (`R_USED_ID_DROPPED`: removing an entity keeps its ID listed), or (b) declares as
  an entity an ID that is listed but not declared there (`R_USED_ID_REUSED`). Rule (b) applies
  even when the same source folder returns: it needs a new ID. The registry carries no mapping,
  so it cannot say which entity an ID once named. A case-only correction therefore needs a new,
  non-colliding ID (§4.6).
- Scope of the guarantee: the published file's lineage inside one output folder (§10 compares each
  publication with the file it replaces). A new output folder starts a new lineage and the
  registry cannot see across it; the run manifest history is the only cross-folder record
  (ADR 0001, decision Q1).
- `used_ids` is not part of any section digest (§9.3).

## 5. Reference graph

- `reference` names the primary matched control of a condition. It is experiment metadata, not
  a contrast table; further contrasts belong to MorphoCoverage.
- It MUST name a declared condition, exactly (`E_REFERENCE_UNKNOWN`; `ctrl` does not resolve to
  `CTRL`), MUST NOT name its own condition (`E_REFERENCE_SELF`), and the graph of all `reference`
  edges MUST be acyclic (`E_REFERENCE_CYCLE`). A condition has at most one `reference` (an array
  is `E_TYPE`).
- Chains are allowed (`VEC_SiCol6a1 → VEC_SiNC → VEC`). A condition is a *control* iff some
  condition references it.
- If a condition and its reference share no repeat, MorphoCoverage cannot pair them: warning
  `W_REFERENCE_UNPAIRED`.

## 6. Validation phases and error catalogue

A reader MUST run the phases in order and stop after the first phase that reports an error. It
MUST report every error of that phase, each with the code, the table path (e.g.
`repeats.Day1.members.CTRL.technical_repeats.single.channels[2]`) and the offending value.
Warnings are reported only for a file without errors and never block publication. Conformance
compares the **set of codes** of the reported phase ([fixtures](../fixtures/README.md)).
**One code per broken rule (AQ13):** each rule has its own code and a reader reports every rule
that an identity or value breaks, so ` DNA` as a channel ID is `E_ID_TEXT` (outer whitespace) and
`E_ID_SYNTAX` (the regular expression), and a decomposed key is `E_ID_NOT_NFC` in addition to any
other rule it breaks. A rule is never reported twice under two codes, and a reference is checked
for NFC only (phase 4) before it is resolved (phase 5).

| Phase | Code | Meaning |
|---|---|---|
| 1 Decode | `E_FILE_ENCODING` | Not UTF-8, or starts with a byte-order mark |
| 1 | `E_TOML_SYNTAX` | Not TOML 1.0 (includes unquoted text such as `NO`, and `012`) |
| 1 | `E_DUPLICATE_KEY` | A key or table declared twice |
| 2 Header | `E_SCHEMA_ID` | `schema` missing or not `morphobricks-experiment` |
| 2 | `E_SCHEMA_VERSION_MISSING` | `schema_version` missing |
| 2 | `E_SCHEMA_VERSION_UNSUPPORTED` | `schema_version` an integer other than 1 |
| 2 | `E_TYPE` | `schema` or `schema_version` of the wrong TOML type |
| 3 Structure | `E_UNKNOWN_FIELD` | Field not defined for its table |
| 3 | `E_FORBIDDEN_FIELD` | Field name or ID key from §4.7 |
| 3 | `E_MISSING_FIELD` | Required field absent |
| 3 | `E_TYPE` | Wrong TOML type (no conversion) |
| 3 | `E_EMPTY` | Required table/array is empty |
| 3 | `E_TEXT_EMPTY` | Non-empty text is empty or whitespace only |
| 3 | `E_ENUM` | Value outside its enumeration |
| 3 | `E_DATE_FORMAT` | Not a real `YYYY-MM-DD` date string |
| 3 | `E_VALUE_RANGE` | Integer outside its range |
| 3 | `E_REPEAT_UNCONFIRMED` | `confirmed = false` in a published file |
| 4 Identity | `E_ID_TEXT` | Label rule violated |
| 4 | `E_ID_SYNTAX` | Token regular expression violated |
| 4 | `E_ID_RESERVED` | Name on the reserved list of §3.2 (channel and mask IDs, and entries of `used_ids.channels`, §4.9), or a mask ID with an `in` part |
| 4 | `E_ID_NOT_NFC` | An identity (key, reference, source repeat, `source_folder`, unit `channels` or `source_channels` entry, `used_ids` entry) is not Unicode NFC; never normalized |
| 4 | `E_ID_CASE_COLLISION` | Two identities of one scope equal ignoring case (`used_ids` entries share the scope of their kind; source folders have their own scopes, §3.3) |
| 4 | `E_SINGLE_MISUSE` | `single` in another case or not alone, or a technical repeat ID or condition ID `single` whose `source_folder` is not `single` (or the reverse) |
| 5 Semantics | `E_UNKNOWN_CONDITION` | Member key is not a declared condition ID |
| 5 | `E_UNKNOWN_CHANNEL` | Unit or `source_channels` names an undeclared channel ID |
| 5 | `E_DUPLICATE_CHANNEL_IN_UNIT` | A channel twice in one unit |
| 5 | `E_DUPLICATE_VALUE` | Duplicate in `header_names` (equal after NFC), `source_channels` or a `used_ids` list |
| 5 | `E_RGB_UNIT_CHANNELS` | A unit with `image_type = "rgb"` whose `channels` is not exactly `["R", "G", "B"]` (§4.4b) |
| 5 | `E_RGB_TOKEN_MISUSE` | A unit without `image_type = "rgb"` lists a channel whose case-folded ID is `r`, `g` or `b` (§4.4b) |
| 5 | `E_RGB_ALIAS` | A channel whose case-folded ID is `r`, `g` or `b` has non-empty `header_names` (§4.4b) |
| 5 | `E_DISPLAY_NAME_COLLISION` | Two display names of one kind equal after NFC and case folding (§4.8) |
| 5 | `E_USED_ID_MISSING` | A declared condition, channel, repeat or technical repeat ID is not in `used_ids` (§4.9) |
| 5 | `E_SOURCE_FOLDER_REUSED` | One `source_folder` on two condition IDs, or on two technical repeat IDs (§4.6a) |
| 5 | `E_SOURCE_FOLDER_CONFLICT` | One technical repeat ID declared with two different `source_folder` values (§4.6a) |
| 5 | `E_CONDITION_WITHOUT_MEMBERS` | Declared condition in no repeat |
| 5 | `E_SOURCE_REPEAT_REUSED` | One source folder in two repeats |
| 5 | `E_REPEAT_ID_SHADOWS_SOURCE` | Repeat ID equal to the source folder name of another repeat's member |
| 5 | `E_MASK_SOURCE_TYPE_REUSED` | Two masks with one `source_type` |
| 5 | `E_REFERENCE_UNKNOWN` | `reference` names no declared condition ID (a display name is not an ID) |
| 5 | `E_REFERENCE_SELF` | `reference` names its own condition |
| 5 | `E_REFERENCE_CYCLE` | The reference graph has a cycle |
| warn | `W_UNUSED_CHANNEL` | A declared channel is in no unit |
| warn | `W_REPEAT_MISSING_CONDITIONS` | A repeat lacks some declared conditions |
| warn | `W_REFERENCE_UNPAIRED` | A condition and its reference share no repeat |

Phase-2 `E_TYPE` concerns only `schema`/`schema_version`. An unsupported or missing version stops
validation: a reader MUST NOT try to interpret fields of a version it does not support.

## 7. Versioning and migration

- `schema_version` is an integer. Any change to fields, types, rules or meanings requires a new
  version. Readers MUST refuse any version they do not implement (`E_SCHEMA_VERSION_UNSUPPORTED`)
  with a message naming the supported versions; they MUST NOT read a newer file "best effort".
  The supported set is a constant of the reader (in v1: `{1}`). It is checked without network
  access, against nothing fetched. A missing version is `E_SCHEMA_VERSION_MISSING` and is never
  mapped to a default version.
- A migration between versions MUST be an explicit, recorded operation that writes a new file and
  never changes canonical identities silently.
- **Repeat, condition, technical repeat and channel IDs are immutable** (§4.6) and never reused
  (§4.9). There is no rename migration; a friendlier name is `display_name`. A mistyped ID is corrected only by a
  recorded re-confirmation, which retires the old ID in `used_ids`. No reader or writer repairs,
  re-cases or renumbers an ID (§3.3).
- **Source folder renamed or moved:** a removed member plus a new, unpaired, pending member. It is
  never auto-paired with the removed one; the user may assign it to the existing canonical repeat
  (the canonical ID does not change). For a **condition** or **technical repeat** folder the same
  holds (the folder is removed, the new folder is pending with a proposed ID, §4.6a); giving the
  new folder to the existing ID changes that ID's `source_folder`, which is a reassignment and
  follows the removed/reassigned rules of §4.6 (recorded user confirmation, else
  `R_CONFIRMED_ID_CHANGED`). The ID never changes.
- **No migration from a legacy `experiment_design.csv`** (ADR 0002 AR10 (9), owner decision of
  10 November 2026). Preprocessing MUST NOT read an `experiment_design.csv` found in an output
  folder to propose or prefill a description or a draft: a description is created only from
  discovery of the experiment folder and the user's answers (§10). The derived
  `experiment_design.csv` of §11 is an output only and is never read back.

## 8. Validation boundaries

| Layer | Owns | Codes |
|---|---|---|
| Spec (this document) | Syntax, types, identities, cross-references, graphs; no files read except this one | `E_*`, `W_*` |
| Preprocessing (reference reader) | Physical agreement of the file with the experiment folder; reconciliation; publication | `R_*` |
| Analysis layer (MorphoFeature, MorphoRelate, fiberfeatures, MorphoEmbed, MorphoCoverage, CellPatcher, normalization) | Applicability of a request to the described units | `A_*` |

Required preprocessing refusals (`R_*`):

| Code | Meaning |
|---|---|
| `R_FILE_SYMLINK` | `morphobricks.toml` is a symlink |
| `R_CONFIRMED_ID_CHANGED` | A confirmed ID was changed, or a declared ID names a changed entity (something it held was removed or reassigned, a `source_folder` included; adding is not a change), without a recorded re-confirmation (§4.6) |
| `R_USED_ID_REUSED` | A candidate declares an ID that `used_ids` lists as retired (§4.9) |
| `R_USED_ID_DROPPED` | A candidate omits an ID listed in `used_ids` of the file it replaces (§4.9) |
| `R_CONFIRMED_ON_CHANGED` | `confirmed_on` of a repeat whose confirmation was not reopened differs from the published file (§4.4a) |
| `R_RECORDS_FILE_INVALID` | `morphobricks.records.jsonl` breaks the encoding, key set or value rules of §4.6; reasons `E_RECORDS_ENCODING`, `E_RECORDS_NO_FINAL_LF`, `E_RECORDS_NOT_JSON_OBJECT`, `E_RECORDS_UNKNOWN_TYPE`, `E_RECORDS_KEY_SET`, `E_RECORDS_KEY_ORDER`, `E_RECORDS_NON_ASCII_ESCAPED`, `E_RECORDS_VALUE`, `E_RECORDS_FORM` |
| `R_PUBLISHED_FILE_CHANGED` | The SHA-256 of the published file at publication differs from the draft's recorded base (§10) |
| `R_DRAFT_BASE_MISSING` | A draft has no recorded base SHA-256 (§10) |
| `R_FOLDER_NOT_NFC` | A folder label or a discovered mask type name read from disk is not NFC (§3.3) |
| `R_UNIT_NOT_FOUND` | A declared imaging unit has no included image |
| `R_UNIT_NOT_DECLARED` | A discovered imaging unit is not in the file (reconciliation draft required) |
| `R_CHANNEL_COUNT_MISMATCH` | An image's channel count differs from its unit's list length |
| `R_UNIT_NOT_HOMOGENEOUS` | FOVs of one unit differ in channel count, image type, interpreted axes or header names at one C index; the message names FOV, path and C index |
| `R_HEADER_CONTRADICTION` | A header name at C*i* is not, after NFC of both sides and with no trimming (§4.3), in the confirmed `header_names` of the unit's channel at *i* |
| `R_IMAGE_TYPE_CONTRADICTS_HEADER` | A unit's declared image type (`rgb` or absent, §4.4b) contradicts the image header metadata, in either direction; pixels are never examined |
| `R_HETEROGENEOUS_LAYOUT_NOT_ENABLED` | Stage 1–2 only: the units do not all have the same ordered list; names the units; lifted at Stage 3 |
| `R_MASK_TYPE_NOT_DECLARED` | A discovered mask type has no `masks` entry |
| `R_MASK_TYPE_NAME_INVALID` | A discovered mask type name breaks the Label rules (for example `NA`, ` cells`), so it cannot be a `source_type`; refused at discovery (AQ11) |
| `R_MASK_CONTRADICTS_REPRESENTATION` | A mask declared `binary` holds more than two distinct values; raised by the mask step on each new batch, not at publication (§4.5a) |

Within one unit, `R_UNIT_NOT_HOMOGENEOUS` is reported first and replaces
`R_CHANNEL_COUNT_MISMATCH` and `R_HEADER_CONTRADICTION` for that unit: intra-unit discrepancies
are detected before channel identity is checked.

Undetectable by any layer: identical-shape, unnamed images with physically swapped fluorophores.
Representative previews and explicit confirmation are the guard; image-based checks are advisory.

Required analysis-layer refusals (`A_*`): `A_SCOPE_INCOMPLETE` (a requested canonical channel or
mask is absent from part of the requested scope, **or from all of it**: one rule, one code, no
separate code for "no image has it" (owner decision SI1); the refusal names the channel or mask and
the images that lack it, as "present in k of n acquisitions" (`k` may be 0); refuse unless the user
selects and records an eligible subset; never drop FOVs, impute, re-weight technical repeats or
change biological N);
`A_MASK_NOT_INSTANCES` (a `binary` mask requested as an object set without the explicit recorded per-request conversion of §4.5c); `A_MASK_UNCHECKED` (a run uses a mask file whose representation check is `not yet checked`, §4.5a); `A_RGB_NOT_SUPPORTED` (a request on a described run includes the channels of an RGB unit, §4.4b, for a brick whose manifest does not declare `rgb_support = true`, or that does not declare it at all); `A_DRAFT_REFUSED` (§10: `RunInputs` given a draft or an unconfirmed file). Each biological repeat is
classified **fully represented**, **partially represented** or **unavailable** per request and
recorded in the run manifest. Consumers resolve channels by canonical ID per unit, never by C
index alone, and group by canonical repeat ID (`repeat_group`, §3.4) while keeping the source label
in every key.

## 9. Serialization, round trip and caching

### 9.1 Canonical serialization (writers)

Writers MUST produce deterministic bytes: UTF-8, no BOM, LF, a final newline, no comments.
Order: `schema`, `schema_version`, then `conditions`, `channels`, `repeats`, `masks` (an empty
optional table is omitted), `used_ids` (kinds in the order `conditions`, `channels`, `repeats`,
`technical_repeats`; each array sorted in code point order, inline); identities are NFC. Entries of each table sorted by key in code
point order; fields in the order of §4; each table as a standard `[header]`; arrays inline;
`header_names`, `source_channels` and `used_ids` arrays sorted, unit `channels` in declared order; keys bare when
they match `[A-Za-z0-9_-]+`, else basic quoted strings; strings as TOML basic strings with the
minimal escapes.

### 9.2 Round trip

For every valid file F: `parse(write(parse(F)))` MUST equal `parse(F)`, and `write` MUST be
idempotent. Any field a reader would drop is refused instead (§2.4), so no confirmed semantics can
be lost silently. Reordering tables or entries, or switching between bare/quoted keys or
standard/inline/dotted tables, never changes meaning.

### 9.3 Provenance vs cache keys

The whole-file SHA-256 is provenance. Preprocessing SHOULD also record per-section digests (SHA-256
of the canonical serialization of `conditions`, `channels`, `repeats`, `masks`) so that consumers
key derived products only on the sections they use. **The bytes of one section digest** are the
section's table blocks exactly as the canonical writer emits them in the file (§9.1), each block
followed by one blank line, with `display_name`, `target` and `treatment_key` left out and nothing
before or after; a section with no tables (no `masks`) is the SHA-256 of the empty byte string.
`source_folder` is part of the `conditions` and
`repeats` digests (it fixes which folder an ID names; a reassignment changes them). Digests are computed with every `display_name`,
`target` and `treatment_key` omitted (descriptive fields are never cache keys, §4.2a), and
`used_ids` is not a section: a change to any of them MUST NOT change any digest, a change to `reference` MUST NOT
invalidate pixel-derived products; a change to a unit's channel list MUST invalidate the products
that depend on that unit's channels.

Provenance from lab tools (for example LabChronicle: repository URL, commit SHA, experiment path
and per-repeat lab-tool repeat codes) is an optional, user-supplied preprocessing input recorded
**only in the run manifest**, never in `morphobricks.toml` (`labchronicle`, `provenance` and
`source_database` are `E_FORBIDDEN_FIELD`). It is excluded from every identity, pairing and cache
key. A mapping from a lab-tool code such as `R001` to a canonical repeat ID MUST be confirmed by
the user: equal strings are not evidence, because the namespaces are unrelated.

## 10. Reconciliation and confirmation (preprocessing behaviour)

On every run, preprocessing compares fresh discovery with the published file and classifies each
unit **unchanged**, **new**, **removed** or **conflicting**. It never rewrites confirmed semantics
silently; edits go to a separate draft and are published atomically after validation.

- **Publish only from the current base.** Every draft records the SHA-256 of the exact bytes of
  the published `morphobricks.toml` it was derived from, or the explicit value `absent` when none
  existed (a draft without it: `R_DRAFT_BASE_MISSING`). Under the output folder's lock, at the
  moment of publication, preprocessing MUST compute the SHA-256 of the file currently published
  (or find none) and refuse with `R_PUBLISHED_FILE_CHANGED` unless it equals the recorded base.
  The lock prevents concurrent writes; the base check prevents a stale draft from replacing a newer
  published file. The refused draft is never merged automatically: it is re-derived from the
  current file.
- **`confirmed_on` is kept** byte-identical for every repeat whose confirmation is not reopened
  (§4.4a); the draft copies it from the published file, never from the clock.

- **N35 reopening.** The confirmation of a repeat is reopened (draft `confirmed = false`) only when
  channels, layout or grouping change: a new or removed unit or member, a unit's ordered channel
  list, or the pairing of folders into repeats. New FOVs (or removed ones) inside an existing unit
  whose confirmed layout still holds do **not** reopen it; they are reported and recorded. A
  changed `image_count` of a confirmed member is recorded as an
  `image_count_changed` record (§4.6) whether or not anything else changes in the same
  publication (RR3). A
  changed `image_count` alone is not a reopening: a new `confirmed_on` beside only a
  changed `image_count` is `R_CONFIRMED_ON_CHANGED` (§4.4a).
- **Adding a member is allowed (AQ1).** A new member folder or unit joins an existing repeat or
  condition without a changed-ID refusal (§4.6); it is a grouping change, so it reopens the N35
  confirmation of that repeat, and it never changes an existing ID. Only removing or reassigning
  existing members counts as a changed entity.
- A removed member is reported and removed only on user confirmation, which is the recorded
  confirmation of §4.6 (without it the candidate is `R_CONFIRMED_ID_CHANGED`); every confirmed
  removal of a unit, technical repeat, repeat or condition, whether or not an ID survives it, is
  also written as a `removed` record (§4.6); a repeat left
  without members is removed from the file, and its ID stays in `used_ids` and is never reused
  (`R_USED_ID_REUSED`, `R_USED_ID_DROPPED`; §4.9).
- A new folder whose label equals an existing repeat ID is proposed for that repeat but stays
  pending until confirmed.
- **Condition and technical-repeat folders (fourth round).** Discovery matches a folder to an ID
  by its `source_folder`, never by the ID. A folder equal to a published `source_folder` is the
  same entity. A folder not in the file is new and pending, with an ID proposed under §4.6a (the
  label if allowed, else another proposal); it joins no existing ID without the user's
  confirmation, which for an existing ID is a reassignment (§4.6). A published `source_folder`
  with no folder on disk is a removed folder, removed only on user confirmation like a removed
  member.
- A conflicting unit (`R_*`) blocks publication; no per-FOV exception is created.

## 11. Derived `experiment_design.csv` (compatibility output)

Preprocessing MUST derive the file (schema version 1, preprocessing contract §12)
deterministically from a published file, with the contract's byte rules:

| CSV column | Value |
|---|---|
| `condition` | member's condition's `source_folder` (the folder label, not the condition ID) |
| `biological_repeat` | member's `source_repeat` |
| `repeat_group` | canonical repeat ID |
| `control` | `true` iff some condition's `reference` names this condition |
| `image_count` | member's `image_count` |
| `technical_repeat_count` | 0 if the only technical repeat is `single` (its `source_folder` is `single`), else the number of technical repeats |
| `confirmed` | `true` |
| `confirmed_on` | repeat's `confirmed_on` |
| `schema_version` | `1` |

One row per member, sorted by `condition` (the source folder label), then `biological_repeat`, in
code point order. **Bytes** (preprocessing contract §12, rule `Rows`, and the `confirmed_on`
row): UTF-8, LF line endings, one header row naming the nine columns in the order of the table
above, booleans written `true` and `false`, `confirmed_on` written as the repeat's `confirmed_on`
(`YYYY-MM-DD`). The contract does not state the delimiter, the quoting, the final line terminator
or a byte-order mark, so these four are not yet normative; the golden files use a comma, no
quoting (no golden value needs it), LF after every row including the last, and no byte-order
mark, as today's writer does. The CSV carries the **source folder labels exactly as today** and no technical
repeat or condition ID, so the choice of a condition or technical repeat ID (for example `ctrl` for
the folder `control`) never changes it.

Expected `experiment_design.csv` outputs (golden files) for `v13`, `v33` and `v34` are in
`fixtures/expected/`; a derived file MUST equal the golden file byte for byte.

## 12. Drafts

A draft (reconciliation proposal, unconfirmed repeats) is internal
to preprocessing and records its base SHA-256 (§10). It is never named `morphobricks.toml`. Downstream consumers (`RunInputs`) read
only the confirmed, published `morphobricks.toml` and MUST refuse a draft or any file with an
unconfirmed repeat (`A_DRAFT_REFUSED`; a published-path file with `confirmed = false` is also
`E_REPEAT_UNCONFIRMED`).
