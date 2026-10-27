"""One row of the output table: an XYC plane's labels, coordinates and histograms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .histogram import PlaneHistogram

# Label used for hierarchy levels a branch of the folder tree does not have.
SINGLE = "single"

# Hierarchy levels from the top down. A group at one level is identified by the
# labels of that level and every level above it, because labels are only unique
# among siblings (bio_rep1 under two conditions are different samples).
LEVELS = ("condition", "biological_repeat", "technical_repeat")

# Images named per channel count in one message before the rest are counted.
_MAX_LISTED = 5


@dataclass(frozen=True, eq=False)
class PlaneRecord:
    """An XYC plane's identity, experimental labels and per-channel histograms.

    One plane is every channel of one image at one Z and T position (contract
    §1.3, N41), so a record holds one histogram per channel, in the image's
    channel order.

    Attributes
    ----------
    source_relative_path
        Path of the source file relative to the experiment root, with forward
        slashes.
    condition, biological_repeat, technical_repeat
        Folder-derived labels; ``"single"`` when the level is absent.
    histograms
        One :class:`~morphobricks_normalization.histogram.PlaneHistogram` per
        channel, channel 0 first. Every channel has the same pixel type, bin
        count and size. A single histogram is accepted for a one-channel plane.
    z, t
        Zero-based plane coordinates. An axis the image does not have, or has
        with length 1, has coordinate 0; None is stored as 0.
    position_id
        The XY position within its technical repeat (MorphoBricks decision D02):
        from reliable metadata, or the file stem when one file is one position; None
        only for records written before the field was filled.
    """

    source_relative_path: str
    condition: str
    biological_repeat: str
    technical_repeat: str
    histograms: Sequence[PlaneHistogram]
    z: int = 0
    t: int = 0
    position_id: str | None = None

    def __post_init__(self) -> None:
        # Every plane has both coordinates (contract §6.2), so a plane is
        # identified the same way whether or not its file has the axis.
        for name in ("z", "t"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, 0)
        histograms = self.histograms
        if isinstance(histograms, PlaneHistogram):
            histograms = (histograms,)
        object.__setattr__(self, "histograms", tuple(histograms))
        self._check_channels()

    @property
    def channel_count(self) -> int:
        """Number of channels of the plane."""
        return len(self.histograms)

    def channel_histogram(self, channel: int) -> PlaneHistogram:
        """Histogram of channel ``channel`` (zero-based).

        Raises
        ------
        ValueError
            If the plane has no such channel.
        """
        if not 0 <= channel < self.channel_count:
            raise ValueError(
                f"Plane {self.plane_id} has {self.channel_count} channel(s), numbered from 0, "
                f"so it has no channel {channel}."
            )
        return self.histograms[channel]

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

    def _check_channels(self) -> None:
        if not self.histograms:
            raise ValueError(f"Plane {self.plane_id} has no channels; give one histogram each.")
        if not all(isinstance(h, PlaneHistogram) for h in self.histograms):
            raise TypeError(
                f"Plane {self.plane_id}: the histograms must be PlaneHistogram objects, one "
                "per channel."
            )
        first = self.histograms[0]
        for channel, hist in enumerate(self.histograms[1:], 1):
            for what, a, b in (
                ("pixel type", first.dtype, hist.dtype),
                ("bin count", first.bin_count, hist.bin_count),
                ("number of pixels", first.total_pixel_count, hist.total_pixel_count),
            ):
                if a != b:
                    raise ValueError(
                        f"Plane {self.plane_id}: channel {channel} has {what} {b}, but channel "
                        f"0 has {a}. All channels of a plane have the same {what}."
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


def different_channel_counts(images: list[tuple[str, int]], subject: str) -> str:
    """Message for images with different numbers of channels (N43).

    ``images`` lists ``(source_relative_path, channel_count)``, repeats allowed;
    ``subject`` starts the sentence, e.g. ``"These images"``.
    """
    by_count: dict[int, dict[str, None]] = {}
    for image, count in images:
        by_count.setdefault(count, {})[image] = None  # each image once, in order
    parts = []
    for count, found in sorted(by_count.items()):
        names = list(found)
        shown = ", ".join(names[:_MAX_LISTED])
        more = f" and {len(names) - _MAX_LISTED} more" if len(names) > _MAX_LISTED else ""
        parts.append(f"{count} channel(s): {shown}{more}")
    return (
        f"{subject} have different numbers of channels ({'; '.join(parts)}). Every image of an "
        "experiment must have the same channels; split or exclude the other images, then run "
        "the preprocessing brick again."
    )
