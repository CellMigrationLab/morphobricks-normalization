"""Stage 3 (roadmap S3.2, ADR 0003): statistics and fits by canonical channel ID on described trees.

The owner's case first: three biological repeats share three stains, and
Biological_repeat_2 has a fourth, EXTRA_STAIN. Then the Spec fixtures ``v04`` (same stains,
other source order), ``v17`` (partial presence), ``v22`` (a declared channel no unit has),
``v05``/``v06`` (heterogeneous counts and technical repeats), one discrepant field of view, an
injected measurement failure, a channel ID with ``_``, and byte determinism.

Tolerances (documented here, no comparison with an older release, ADR 0002 AR10): the pixels
are uint8 with exact histograms, so min-max, percentile and robust z-score parameters equal
the formulas on the raw pixels exactly (tolerance 0); z-score parameters are sums and are
compared with ``rel=1e-12``. Fits of the same pixels under two layouts are compared with
``==`` (bit for bit).
"""

import csv

import numpy as np
import pytest

from described import FIXTURES, fixture, images, parse, pixels_of, record, records, renamed
from morphobricks_normalization import (
    ChannelMeasurementError,
    ChannelMeasurementFailure,
    ChannelScopeIncompleteError,
    NormalizationSettings,
    Parameters,
    PooledHistogram,
    fit,
    group_name,
    image_summaries,
    percentile_range,
    read_plane_histogram_table,
    scope_groups,
    summary_columns,
    write_plane_histograms,
)

OWNER_CASE = (
    """
schema = "morphobricks-experiment"
schema_version = 1
[conditions.CTRL]
source_folder = "CTRL"
[conditions.DrugA]
source_folder = "DrugA"
reference = "CTRL"
[channels.DNA]
[channels.ACTIN]
[channels.TUBULIN]
[channels.EXTRA_STAIN]
"""
    + "".join(
        f"""
[repeats.Biological_repeat_{r}]
confirmed = true
confirmed_on = "2026-11-10"
"""
        + "".join(
            f"""
[repeats.Biological_repeat_{r}.members.{c}]
source_repeat = "Biological_repeat_{r}"
image_count = 2
[repeats.Biological_repeat_{r}.members.{c}.technical_repeats.single]
source_folder = "single"
channels = {channels}
"""
            for c in ("CTRL", "DrugA")
        )
        for r, channels in (
            (1, '["DNA", "ACTIN", "TUBULIN"]'),
            (2, '["DNA", "ACTIN", "TUBULIN", "EXTRA_STAIN"]'),
            (3, '["DNA", "ACTIN", "TUBULIN"]'),
        )
    )
    + """
[used_ids]
conditions = ["CTRL", "DrugA"]
channels = ["ACTIN", "DNA", "EXTRA_STAIN", "TUBULIN"]
repeats = ["Biological_repeat_1", "Biological_repeat_2", "Biological_repeat_3"]
technical_repeats = ["single"]
"""
)
SHARED = ("ACTIN", "DNA", "TUBULIN")
SUBSET = {"partial_channels": "planes_with_channel"}  # the user's recorded choice
METHODS = ("minmax", "percentile", "zscore", "robust_zscore")


def direct(x, method):
    """Contract §3.2 on raw pixels (inverse-CDF percentiles)."""
    if method == "minmax":
        return x.min(), x.max() - x.min()
    if method == "percentile":
        low, high = (np.percentile(x, q, method="inverted_cdf") for q in (1.0, 99.8))
        return low, high - low
    if method == "zscore":
        return x.mean(), x.std()
    median = np.percentile(x, 50, method="inverted_cdf")
    return median, np.percentile(np.abs(x - median), 50, method="inverted_cdf") * 1.482602218505602


def assert_direct(parameters, channel, pixels, method):
    k = parameters.channels.index(channel)
    offset, scale = direct(pixels, method)
    if method == "zscore":
        assert parameters.offset[k] == pytest.approx(offset, rel=1e-12)
        assert parameters.scale[k] == pytest.approx(scale, rel=1e-12)
    else:
        assert (parameters.offset[k], parameters.scale[k]) == (offset, scale)


def in_repeat(group):
    """Select the images of one canonical repeat, for :func:`pixels_of`."""
    return lambda unit, path, t: unit.repeat == group


def table(tmp_path, description, rows, name="plane_histograms.csv"):
    path = write_plane_histograms(rows, tmp_path / name, channel_ids=description.channel_ids)
    return path, read_plane_histogram_table(path)


def repeat_fits(description, rows, method="percentile", **options):
    settings = NormalizationSettings(method=method, scope="biological_repeat", **options)
    groups, _ = scope_groups(rows, settings, description.repeat_groups())
    return {group_name(k): fit(v, settings, group=group_name(k)) for k, v in groups.items()}


# --- The owner's case: an extra stain in one biological repeat ---------------------------------


@pytest.fixture(scope="module")
def owner(tmp_path_factory):
    description = parse(OWNER_CASE)
    path, read = table(tmp_path_factory.mktemp("owner"), description, records(description))
    return description, path, read.records


def test_owner_case_table_lists_every_id_and_leaves_the_extra_stain_blank_elsewhere(owner):
    description, path, rows = owner
    assert read_plane_histogram_table(path).channel_ids == (*SHARED[:2], "EXTRA_STAIN", "TUBULIN")
    assert len(rows) == 3 * 2 * 2 * 2  # one row per plane: repeats x conditions x images x T
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            extra = [
                v for k, v in row.items() if k.endswith("_EXTRA_STAIN") or "_EXTRA_STAIN_" in k
            ]
            if row["biological_repeat"] == "Biological_repeat_2":
                assert row["channels"] == "DNA;ACTIN;TUBULIN;EXTRA_STAIN"
                assert all(v != "" for v in extra)
            else:
                assert row["channels"] == "DNA;ACTIN;TUBULIN"
                assert set(extra) == {""}  # blank, never a zero or NaN histogram


def test_the_header_lists_the_declared_channel_ids_not_retired_ones(tmp_path):
    # Open question 1 of PR #90: the caller passes the description's declared channels (its
    # [channels] table), not used_ids.channels, which keeps retired IDs no image can have.
    used = 'channels = ["ACTIN", "DNA", "EXTRA_STAIN", "TUBULIN"]'
    assert used in OWNER_CASE
    description = parse(OWNER_CASE.replace(used, used.replace('"TUBULIN"', '"OLD", "TUBULIN"')))
    assert description.channel_ids == ("DNA", "ACTIN", "TUBULIN", "EXTRA_STAIN")
    _, read = table(tmp_path, description, records(description))
    assert read.channel_ids == ("ACTIN", "DNA", "EXTRA_STAIN", "TUBULIN")  # sorted, no OLD


@pytest.mark.parametrize("method", METHODS)
def test_owner_case_shared_stains_pool_across_repeats_extra_only_within_its_repeat(owner, method):
    description, _, rows = owner
    refused = NormalizationSettings(method=method, scope="biological_repeat")
    with pytest.raises(
        ChannelScopeIncompleteError, match="channel EXTRA_STAIN is present in 8 of 24"
    ) as e:
        fit(rows, refused, group="all")  # by default a channel some planes lack is refused
    assert list(e.value.channels) == ["EXTRA_STAIN"]
    assert {p.split("/")[1] for p in e.value.channels["EXTRA_STAIN"]} == {
        "Biological_repeat_1",
        "Biological_repeat_3",
    }
    assert fit(rows, refused, group="all", channels=SHARED).channels == SHARED
    settings = NormalizationSettings(method=method, scope="biological_repeat", **SUBSET)
    pooled = fit(rows, settings, group="all")  # every repeat pooled (a global fit), opted in
    assert pooled.channels == ("ACTIN", "DNA", "EXTRA_STAIN", "TUBULIN")
    for channel in SHARED:
        assert_direct(pooled, channel, pixels_of(description, channel), method)
    in_repeat_2 = pixels_of(description, "EXTRA_STAIN")
    assert_direct(pooled, "EXTRA_STAIN", in_repeat_2, method)
    rep2 = [r for r in rows if r.biological_repeat == "Biological_repeat_2"]
    extra_alone = fit(rep2, settings, group="all", channels=["EXTRA_STAIN"])
    assert pooled.select(["EXTRA_STAIN"]) == extra_alone  # bit for bit
    (warning,) = [i for i in pooled.issues if i.code == "channel_partially_present"]
    assert warning.channel == "EXTRA_STAIN"
    assert {r.split("/")[1] for r in warning.items} == {
        "Biological_repeat_1",
        "Biological_repeat_3",
    }


@pytest.mark.parametrize("method", METHODS)
def test_owner_case_repeat_scope_fits_the_channels_each_repeat_has(owner, method):
    description, _, rows = owner
    fits = repeat_fits(description, rows, method)
    assert {g: p.channels for g, p in fits.items()} == {
        "Biological_repeat_1": SHARED,
        "Biological_repeat_2": ("ACTIN", "DNA", "EXTRA_STAIN", "TUBULIN"),
        "Biological_repeat_3": SHARED,
    }
    for group, parameters in fits.items():
        assert [i.code for i in parameters.issues] == []  # every plane of a repeat has them
        for channel in parameters.channels:
            where = in_repeat(group)
            assert_direct(parameters, channel, pixels_of(description, channel, where), method)


def test_owner_case_planes_with_every_channel_fit_bit_for_bit_like_the_shared_layout(owner):
    # Evidence for "all channels present gives today's numbers": the shared stains' fits from
    # the heterogeneous table equal those of a homogeneous table of the same pixels.
    description, _, rows = owner
    homogeneous = [
        record(unit, path, t, pixels[: len(SHARED)], channels=unit.channels[:3])
        for unit, path, t, pixels in images(description)
    ]
    for method in METHODS:
        for options in ({}, {"pooling": "conditions"}):
            a = repeat_fits(description, rows, method, **options)
            b = repeat_fits(description, homogeneous, method, **options)
            for group in a:
                assert a[group].select(SHARED) == b[group]


def test_owner_case_parameter_columns_are_named_by_id_and_blank_where_absent(owner):
    description, _, rows = owner
    fits = repeat_fits(description, rows)
    header = {name for p in fits.values() for name in p.to_columns()}
    assert "Normalization_Offset_EXTRA_STAIN" in header
    assert not any("_C0" in name or "_C3" in name for name in header)
    settings = NormalizationSettings(scope="biological_repeat")
    for parameters in fits.values():
        written = parameters.to_columns()
        row = {name: written.get(name, "") for name in sorted(header)}
        assert Parameters.from_columns(row, settings) == parameters


def test_owner_case_summary_columns_hold_the_extra_stain_only_where_acquired(owner):
    description, _, rows = owner
    columns = summary_columns(description.channel_ids)
    assert "Intensity_MedianIntensity_EXTRA_STAIN" in columns
    for summary in image_summaries(rows):
        has_it = summary.biological_repeat == "Biological_repeat_2"
        assert ("Histogram_Exact_EXTRA_STAIN" in summary.to_row()) == has_it


# --- v04: same stains, other source order ------------------------------------------------------


def test_v04_swapped_order_gives_the_same_results_per_id_as_the_unswapped_layout(tmp_path):
    swapped = fixture("v04_swapped_channel_order")
    assert {u.condition: u.channels for u in swapped.units} == {
        "CTRL": ("DNA", "ACTIN"),
        "DrugA": ("ACTIN", "DNA"),
    }
    straight = parse(_unswapped_v04_text())  # the same description, DrugA as DNA, ACTIN
    _, a = table(tmp_path, swapped, records(swapped), "a.csv")
    _, b = table(tmp_path, straight, records(straight), "b.csv")
    for method in METHODS:
        for scope in ("biological_repeat", "image"):
            settings = NormalizationSettings(method=method, scope=scope)
            ga, _ = scope_groups(a.records, settings)
            gb, _ = scope_groups(b.records, settings)
            assert list(ga) == list(gb)
            for key in ga:
                assert fit(ga[key], settings) == fit(gb[key], settings)  # bit for bit
    assert image_summaries(a.records) == image_summaries(b.records)
    # By C index the two stains of DrugA would be pooled with the other stain of CTRL.
    by_index = percentile_range(PooledHistogram([r.histograms[0] for r in a.records]), 1, 99.8)
    by_id = fit(a.records, NormalizationSettings()).select(["DNA"])
    assert (by_index.offset, by_index.scale) != (by_id.offset[0], by_id.scale[0])


def _unswapped_v04_text():
    text = (FIXTURES / "v04_swapped_channel_order.toml").read_text(encoding="utf-8")
    section = "[repeats.rep1.members.DrugA.technical_repeats.single]"
    head, tail = text.split(section)
    swapped = tail.replace('channels = ["ACTIN", "DNA"]', 'channels = ["DNA", "ACTIN"]', 1)
    return head + section + swapped


# --- v17: partial channel presence -------------------------------------------------------------


def test_v17_partial_presence_is_refused_by_default(tmp_path):
    description = fixture("v17_partial_channel_presence")
    _, read = table(tmp_path, description, records(description))
    settings = NormalizationSettings(scope="biological_repeat")
    groups, _ = scope_groups(read.records, settings, description.repeat_groups())
    assert fit(groups[("R001",)], settings, group="R001").channels == ("ACTIN", "DNA")
    with pytest.raises(
        ChannelScopeIncompleteError,
        match="A_SCOPE_INCOMPLETE: in group 'R002', channel PHASE is present in",
    ) as e:
        fit(groups[("R002",)], settings, group="R002")
    assert {item.split("/")[2] for item in e.value.channels["PHASE"]} == {"acquisition_2"}
    fitted = fit(groups[("R002",)], settings, group="R002", channels=["ACTIN", "DNA"])
    assert fitted.channels == ("ACTIN", "DNA") and fitted.issues == ()


def test_v17_partial_presence_is_fitted_where_present_when_selected_and_reported(tmp_path):
    description = fixture("v17_partial_channel_presence")
    path, read = table(tmp_path, description, records(description))
    fits = repeat_fits(description, read.records, **SUBSET)
    assert fits["R001"].channels == ("ACTIN", "DNA") and fits["R001"].issues == ()
    assert fits["R002"].channels == ("ACTIN", "DNA", "PHASE")
    (warning,) = fits["R002"].issues
    assert (warning.code, warning.channel) == ("channel_partially_present", "PHASE")
    assert {item.split("/")[2] for item in warning.items} == {"acquisition_2"}

    def where(unit, path, t):
        return unit.technical_repeat == "acquisition_1"

    assert_direct(fits["R002"], "PHASE", pixels_of(description, "PHASE", where), "percentile")
    settings = NormalizationSettings(scope="biological_repeat", **SUBSET)
    groups, _ = scope_groups(read.records, settings, description.repeat_groups())
    with pytest.raises(ValueError, match="No plane of group 'R001' has channel\\(s\\) PHASE"):
        fit(groups[("R001",)], settings, group="R001", channels=["PHASE"])


# --- v22: a declared channel that no unit has (a missing optional channel) ---------------------


def test_v22_a_channel_no_image_has_is_in_the_header_blank_and_never_fitted(tmp_path):
    description = fixture("v22_unused_channel_warning")
    path, read = table(tmp_path, description, records(description))
    assert read.channel_ids == ("ACTIN", "DNA", "PHASE")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {v for row in rows for k, v in row.items() if k.endswith("_PHASE")} == {""}
    assert fit(read.records, NormalizationSettings()).channels == ("ACTIN", "DNA")
    with pytest.raises(ValueError, match="never fitted from zeros"):
        fit(read.records, NormalizationSettings(), channels=["PHASE"])


# --- v05, v06: heterogeneous counts and technical repeats --------------------------------------


@pytest.mark.parametrize(
    "name", ["v05_variable_channel_counts", "v06_technical_repeats_different_layouts"]
)
def test_heterogeneous_layouts_keep_one_row_per_plane_and_one_group_per_repeat(tmp_path, name):
    description = fixture(name)
    rows = records(description)
    _, read = table(tmp_path, description, rows)
    assert len(read.records) == len(rows)  # N41: the row is the plane
    with pytest.raises(ChannelScopeIncompleteError):  # PHASE in part of a repeat: refused
        repeat_fits(description, read.records)
    fits = repeat_fits(description, read.records, **SUBSET)
    assert sorted(fits) == sorted({u.repeat for u in description.units})  # never per acquisition
    for group, parameters in fits.items():
        for channel in parameters.channels:
            where = in_repeat(group)
            assert_direct(parameters, channel, pixels_of(description, channel, where), "percentile")


def test_a_channel_id_with_an_underscore_names_its_columns(tmp_path):
    description = renamed(fixture("v06_technical_repeats_different_layouts"), "PAXILLIN", "PAX_488")
    path, read = table(tmp_path, description, records(description))
    assert "PAX_488" in read.channel_ids
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert (
        "Histogram_ValidPixelCount_PAX_488" in header and "Histogram_Count_PAX_488_00255" in header
    )
    parameters = repeat_fits(description, read.records, **SUBSET)["R001"]
    assert "Normalization_Scale_PAX_488" in parameters.to_columns()


# --- One discrepant field of view, a failed measurement, determinism ---------------------------


def test_a_single_field_of_view_without_a_channel_is_named_and_never_zero_filled(tmp_path):
    description = fixture("v13_homogeneous_legacy")
    rows = []
    for unit, path, t, pixels in images(description):
        odd = path == "DrugA/rep2/img_1.tif"  # acquired without ACTIN
        rows.append(record(unit, path, t, pixels[:1] if odd else pixels, ("DNA",) if odd else None))
    _, read = table(tmp_path, description, rows)
    with pytest.raises(ChannelScopeIncompleteError, match="channel ACTIN is present in 10 of 12"):
        repeat_fits(description, read.records)
    fits = repeat_fits(description, read.records, **SUBSET)
    (warning,) = fits["rep2"].issues
    assert warning.channel == "ACTIN"
    assert warning.items == ("DrugA/rep2/img_1.tif::z=0,t=0", "DrugA/rep2/img_1.tif::z=0,t=1")

    def where(unit, path, t):
        return unit.repeat == "rep2" and path != "DrugA/rep2/img_1.tif"

    assert_direct(fits["rep2"], "ACTIN", pixels_of(description, "ACTIN", where), "percentile")
    assert fits["rep1"].issues == fits["rep3"].issues == ()


def test_an_injected_measurement_failure_writes_no_table_and_names_plane_and_channel(tmp_path):
    description = fixture("v06_technical_repeats_different_layouts")
    rows = records(description)
    failure = ChannelMeasurementFailure(rows[5].plane_id, rows[5].channels[1], "read error")
    with pytest.raises(ChannelMeasurementError, match=f"plane {rows[5].plane_id}, channel"):
        write_plane_histograms(
            rows,
            tmp_path / "plane_histograms.csv",
            channel_ids=description.channel_ids,
            failures=[failure],
        )
    assert list(tmp_path.iterdir()) == []


def test_tables_are_byte_identical_serially_in_shards_and_on_resume(tmp_path):
    description = fixture("v06_technical_repeats_different_layouts")
    rows = sorted(records(description), key=lambda r: (r.source_relative_path, r.z, r.t))
    serial = table(tmp_path, description, rows, "serial.csv")[0].read_bytes()
    assert table(tmp_path, description, rows, "again.csv")[0].read_bytes() == serial
    by_file = {}
    for row in rows:
        by_file.setdefault(row.source_relative_path, []).append(row)
    shards = {}  # one per file, written in reverse order (a parallel run's completion order)
    for k, (path, members) in enumerate(reversed(list(by_file.items()))):
        shard = table(tmp_path, description, members, f"shard{k}.csv")[0]
        shards[path] = shard.read_bytes().splitlines(keepends=True)
    assert len({lines[0] for lines in shards.values()}) == 1  # every shard got every ID
    # Merged by the final row order (path, then z and t within each staged file).
    merged = [line for path in sorted(shards) for line in shards[path][1:]]
    assert b"".join([shards[path][0], *merged]) == serial
    # A resumed run: half the shards from before, half written again, the same bytes.
    resumed = {**shards, **{p: shards[p] for p in sorted(shards)[::2]}}
    assert (
        b"".join([shards[path][0], *[ln for p in sorted(resumed) for ln in resumed[p][1:]]])
        == serial
    )
