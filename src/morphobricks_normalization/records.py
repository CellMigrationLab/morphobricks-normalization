"""One row of the output table: an XYC plane's labels, coordinates and histograms."""

from __future__ import annotations

import numbers
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .channels import check_channel_id, check_channel_ids
from .histogram import PlaneHistogram

# Label used for hierarchy levels a branch of the folder tree does not have.
SINGLE = "single"

# Hierarchy levels from the top down. A group at one level is identified by the
# labels of that level and every level above it, because labels are only unique
# among siblings (bio_rep1 under two conditions are different samples).
LEVELS = ("condition", "biological_repeat", "technical_repeat")

# Planes or measurements named in one message before the rest are counted.
_MAX_LISTED = 5


@dataclass(frozen=True, eq=False)
class PlaneRecord:
    """An XYC plane's identity, experimental labels and per-channel histograms.

    One plane is every channel of one image at one Z and T position (contract
    §1.3, N41), so a record holds one histogram per channel. Each channel is named
    by its canonical channel ID (ADR 0003 HV2), never by its index in the image.

    Attributes
    ----------
    source_relative_path
        Path of the source file relative to the experiment root, with forward
        slashes.
    condition, biological_repeat, technical_repeat
        Folder-derived labels; ``"single"`` when the level is absent.
    histograms
        One :class:`~morphobricks_normalization.histogram.PlaneHistogram` per
        channel, in the order of ``channels``. Every channel has the same pixel
        type, bin count and size. A single histogram is accepted for a one-channel
        plane.
    z, t
        Zero-based plane coordinates. An axis the image does not have, or has
        with length 1, has coordinate 0; None is stored as 0. Booleans,
        fractions, text and negative numbers are refused with ``ValueError``.
    position_id
        Required, keyword-only, non-empty text: the XY position within its technical
        repeat (MorphoBricks decision D02), from reliable metadata, or the file stem
        when one file is one position. A record without it is refused.
    channels
        Required, keyword-only: the canonical channel IDs of the plane's channels in
        source order, entry ``i`` naming source channel ``C<i>`` (the imaging unit's
        confirmed ``channels`` list, experiment-v1 §4.4). These are the channels
        *present* in the plane; a channel of the experiment that is not listed is
        absent, which is not a measurement and never a zero histogram (ADR 0003 HV1).
        Never guessed or filled in: a record without it is refused.
    """

    source_relative_path: str
    condition: str
    biological_repeat: str
    technical_repeat: str
    histograms: Sequence[PlaneHistogram]
    z: int = 0
    t: int = 0
    position_id: str = field(kw_only=True)
    channels: tuple[str, ...] = field(kw_only=True)

    def __post_init__(self) -> None:
        # Every plane has both coordinates (contract §6.2), so a plane is
        # identified the same way whether or not its file has the axis.
        for name in ("z", "t"):
            object.__setattr__(self, name, _coordinate(getattr(self, name), name, self))
        histograms = self.histograms
        if isinstance(histograms, PlaneHistogram):
            histograms = (histograms,)
        object.__setattr__(self, "histograms", tuple(histograms))
        self._check_position_id()
        object.__setattr__(
            self, "channels", check_channel_ids(self.channels, f"Plane {self.plane_id}")
        )
        self._check_channels()

    @property
    def channel_count(self) -> int:
        """Number of channels of the plane (of its source image)."""
        return len(self.histograms)

    def has_channel(self, channel: str) -> bool:
        """Whether the plane has (lists as present) the channel with ID ``channel``."""
        return channel in self.channels

    def channel_histogram(self, channel: str) -> PlaneHistogram:
        """Histogram of the channel with canonical ID ``channel``.

        Raises
        ------
        ValueError
            If ``channel`` is not a channel ID (an index is refused), or the plane does
            not have that channel (it is absent from the plane's imaging unit).
        """
        check_channel_id(channel, f"Plane {self.plane_id}")
        if channel not in self.channels:
            raise ValueError(
                f"Plane {self.plane_id} has channel(s) {', '.join(self.channels)}; channel "
                f"{channel} is absent from it (its imaging unit does not list it), so it has no "
                "histogram. An absent channel is not measured and is never taken as zero."
            )
        return self.histograms[self.channels.index(channel)]

    @property
    def image_file(self) -> str:
        """File name of the source image."""
        return self.source_relative_path.rsplit("/", 1)[-1]

    @property
    def plane_id(self) -> str:
        """Deterministic plane identifier: path plus the plane's coordinates."""
        return make_plane_id(self.source_relative_path, z=self.z, t=self.t)

    def group_key(self, level: str) -> tuple[str, ...]:
        """Labels identifying this plane's group at ``level``.

        ``level`` is one of ``"condition"``, ``"biological_repeat"``,
        ``"technical_repeat"`` or ``"image"`` (the source file).
        """
        if level == "image":
            return (*self.group_key("technical_repeat"), self.source_relative_path)
        if level not in LEVELS:
            raise ValueError(f"Unknown level {level!r}; expected one of {(*LEVELS, 'image')}.")
        labels = (self.condition, self.biological_repeat, self.technical_repeat)
        return labels[: LEVELS.index(level) + 1]

    def _check_position_id(self) -> None:
        # Never guessed or filled in: the position is part of the identity of the image.
        if not isinstance(self.position_id, str) or not self.position_id.strip():
            raise ValueError(
                f"Plane of {self.source_relative_path}: it has no position_id (got "
                f"{self.position_id!r}). Every plane needs the XY position of its image within "
                "its technical repeat, as the preprocessing brick records it; give "
                "position_id, or run the preprocessing brick again on this experiment."
            )

    def _check_channels(self) -> None:
        """Refuse no histogram, one ID per histogram missing, or channels that differ."""
        if not self.histograms:
            raise ValueError(f"Plane {self.plane_id} has no channels; give one histogram each.")
        if not all(isinstance(h, PlaneHistogram) for h in self.histograms):
            raise TypeError(
                f"Plane {self.plane_id}: the histograms must be PlaneHistogram objects, one "
                "per channel."
            )
        if len(self.channels) != len(self.histograms):
            raise ValueError(
                f"Plane {self.plane_id} has {len(self.histograms)} histogram(s) but "
                f"{len(self.channels)} channel ID(s) ({', '.join(self.channels) or 'none'}). "
                "Give one channel ID per histogram, in source order: the channels list of the "
                "image's imaging unit in the experiment description."
            )
        first = self.histograms[0]
        for channel, hist in zip(self.channels[1:], self.histograms[1:], strict=True):
            for what, a, b in (
                ("pixel type", first.dtype, hist.dtype),
                ("bin count", first.bin_count, hist.bin_count),
                ("number of pixels", first.total_pixel_count, hist.total_pixel_count),
            ):
                if a != b:
                    raise ValueError(
                        f"Plane {self.plane_id}: channel {channel} has {what} {b}, but channel "
                        f"{self.channels[0]} has {a}. All channels of a plane have the same "
                        f"{what}."
                    )


@dataclass(frozen=True)
class ChannelMeasurementFailure:
    """A channel of a plane whose histogram could not be measured (ADR 0003 HV3).

    The calling brick records one when computing a channel's histogram fails (a read
    error, an unsupported value); the plane-histogram table then cannot be written
    (:func:`~morphobricks_normalization.csv_io.write_plane_histograms`). It is not an
    absent channel (a channel the imaging unit does not list), and not an empty
    histogram (a measured channel without finite pixels, which is a measurement).

    Attributes
    ----------
    plane_id
        The plane, as :func:`make_plane_id` gives it.
    channel
        The canonical channel ID of the channel that failed.
    reason
        What went wrong, in plain language, for the run report.
    """

    plane_id: str
    channel: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.plane_id, str) or not self.plane_id:
            raise ValueError(
                f"A failed measurement names its plane (plane_id); got {self.plane_id!r}."
            )
        check_channel_id(self.channel, f"Failed measurement of plane {self.plane_id}")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError(
                f"The failed measurement of plane {self.plane_id}, channel {self.channel}, "
                "needs a reason for the run report."
            )


class ChannelMeasurementError(ValueError):
    """A channel measurement failed, so the plane-histogram table is not written (HV3).

    Attributes
    ----------
    failures
        Every failed measurement, in the order given.
    removed
        The table of an earlier run that the writer removed, so that no stale table is
        left that looks current; None when there was none.
    """

    def __init__(
        self, failures: Sequence[ChannelMeasurementFailure], removed: str | None = None
    ) -> None:
        self.failures = tuple(failures)
        self.removed = removed
        listed = "; ".join(
            f"plane {f.plane_id}, channel {f.channel}: {f.reason}"
            for f in self.failures[:_MAX_LISTED]
        )
        more = (
            f"; and {len(self.failures) - _MAX_LISTED} more"
            if len(self.failures) > _MAX_LISTED
            else ""
        )
        stale = (
            f" The table of an earlier run, {removed}, was removed, so that no out-of-date "
            "table is left that looks current."
            if removed
            else ""
        )
        super().__init__(
            f"{len(self.failures)} channel measurement(s) failed ({listed}{more}), so "
            "plane_histograms.csv was not written: the table holds a measured histogram for "
            f"every channel a plane has, and never a partial table.{stale} Fix the cause (read "
            "the failing image and channel), then run the preprocessing brick again."
        )


def check_measurements(failures: Iterable[ChannelMeasurementFailure]) -> None:
    """Refuse a run with any failed channel measurement (ADR 0003 HV3).

    Parameters
    ----------
    failures
        The run's failed measurements; empty when every channel was measured.

    Raises
    ------
    ChannelMeasurementError
        Naming every failing plane and channel, when ``failures`` is not empty.
    TypeError
        For an entry that is not a :class:`ChannelMeasurementFailure`.
    """
    failures = list(failures)
    if not all(isinstance(f, ChannelMeasurementFailure) for f in failures):
        raise TypeError("Give the failed measurements as ChannelMeasurementFailure objects.")
    if failures:
        raise ChannelMeasurementError(failures)


def _coordinate(value: object, name: str, record: PlaneRecord) -> int:
    """Return a plane coordinate as an int: None is 0, other values must be whole and >= 0."""
    if value is None:
        return 0
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(
            f"Plane of {record.source_relative_path}: its {name} coordinate must be a whole number "
            f"0 or above (coordinates count from 0); got {value!r}."
        )
    return int(value)


def duplicate_planes(records: Iterable[PlaneRecord], subject: str) -> None:
    """Refuse records that share a ``plane_id``: each plane's pixels must count once.

    Parameters
    ----------
    records
        The records of one table or group.
    subject
        Starts the message, e.g. ``"The planes to write"``.

    Raises
    ------
    ValueError
        Naming every repeated ``plane_id`` and how often it appears.
    """
    seen: dict[str, int] = {}
    for record in records:
        seen[record.plane_id] = seen.get(record.plane_id, 0) + 1
    repeated = [f"{plane_id} ({n} times)" for plane_id, n in seen.items() if n > 1]
    if repeated:
        shown = ", ".join(repeated[:_MAX_LISTED])
        more = f" and {len(repeated) - _MAX_LISTED} more" if len(repeated) > _MAX_LISTED else ""
        raise ValueError(
            f"{subject} hold some planes more than once: {shown}{more}. Every plane (an image "
            "at one Z and T position) must appear once, or its pixels count twice; remove the "
            "repeated rows, or run the preprocessing brick again on this experiment."
        )


def check_universe(records: list[PlaneRecord], universe: set[str]) -> None:
    """Refuse records with a channel the experiment's list does not have."""
    outside: dict[str, list[str]] = {}
    for record in records:
        for channel in record.channels:
            if channel not in universe:
                outside.setdefault(channel, []).append(record.plane_id)
    if outside:
        parts = "; ".join(
            f"{channel} (in {', '.join(ids[:3])}{' ...' if len(ids) > 3 else ''})"
            for channel, ids in sorted(outside.items())
        )
        raise ValueError(
            f"Some planes have channels that are not in the experiment's channel IDs "
            f"({', '.join(sorted(universe))}): {parts}. Give channel_ids every channel ID of the "
            "experiment description."
        )


def same_channels_per_image(records: Iterable[PlaneRecord], subject: str) -> None:
    """Refuse an image whose planes list different channels.

    One source file has one channel layout: every Z and T plane of it has the same
    channels in the same source order. Planes that differ would pool one stain under
    another's ID or drop a channel for some planes, so they are refused, never merged.

    Parameters
    ----------
    records
        The records of one table or group.
    subject
        Starts the message, e.g. ``"The planes to write"``.

    Raises
    ------
    ValueError
        Naming each such image and the channel lists of its planes.
    """
    layouts: dict[str, dict[tuple[str, ...], list[str]]] = {}
    for record in records:
        planes = layouts.setdefault(record.source_relative_path, {})
        planes.setdefault(record.channels, []).append(record.plane_id)
    mixed = {path: found for path, found in layouts.items() if len(found) > 1}
    if not mixed:
        return
    parts = []
    for path, found in list(mixed.items())[:_MAX_LISTED]:
        lists = "; ".join(
            f"{';'.join(channels)} in {', '.join(ids[:2])}{' ...' if len(ids) > 2 else ''}"
            for channels, ids in found.items()
        )
        parts.append(f"{path} ({lists})")
    more = f" and {len(mixed) - _MAX_LISTED} more" if len(mixed) > _MAX_LISTED else ""
    raise ValueError(
        f"{subject} hold images whose planes list different channels: {', '.join(parts)}{more}. "
        "Every plane of one image has the channels of its imaging unit, in the same order; "
        "run the preprocessing brick again on this experiment."
    )


def make_plane_id(source_relative_path: str, *, z: int | None = 0, t: int | None = 0) -> str:
    """Build a plane identifier such as ``"cond/rep1/img.tif::z=3,t=0"``.

    Both coordinates are always written; None (an axis the image does not
    have) is written as 0, so a 2D image's identifier is ``"<path>::z=0,t=0"``.
    A plane holds every channel, so the identifier has no channel coordinate.
    """
    coords = ",".join(
        f"{name}={0 if value is None else value}" for name, value in (("z", z), ("t", t))
    )
    return f"{source_relative_path}::{coords}"


def group_records(
    records: list[PlaneRecord], level: str
) -> dict[tuple[str, ...], list[PlaneRecord]]:
    """Group records by their labels at ``level``, in sorted key order."""
    groups: dict[tuple[str, ...], list[PlaneRecord]] = {}
    for record in records:
        groups.setdefault(record.group_key(level), []).append(record)
    return dict(sorted(groups.items()))
