"""One row of the output table: a plane's labels, coordinates and histogram."""

from __future__ import annotations

from dataclasses import dataclass

from .histogram import PlaneHistogram

# Label used for hierarchy levels a branch of the folder tree does not have.
SINGLE = "single"

# Hierarchy levels from the top down. A group at one level is identified by the
# labels of that level and every level above it, because labels are only unique
# among siblings (bio_rep1 under two conditions are different samples).
LEVELS = ("condition", "biological_repeat", "technical_repeat")


@dataclass(frozen=True, eq=False)
class PlaneRecord:
    """A plane's identity, experimental labels and histogram.

    Attributes
    ----------
    source_relative_path
        Path of the source file relative to the experiment root, with forward
        slashes.
    condition, biological_repeat, technical_repeat
        Folder-derived labels; ``"single"`` when the level is absent.
    histogram
        The plane's :class:`~morphobricks_normalization.histogram.PlaneHistogram`.
    z, c, t
        Zero-based plane coordinates. An axis the image does not have, or has
        with length 1, has coordinate 0; None is stored as 0.
    position_id
        Position identifier from reliable metadata, or None.
    """

    source_relative_path: str
    condition: str
    biological_repeat: str
    technical_repeat: str
    histogram: PlaneHistogram
    z: int = 0
    c: int = 0
    t: int = 0
    position_id: str | None = None

    def __post_init__(self) -> None:
        # Every plane has all three coordinates (contract §2), so a plane is
        # identified the same way whether or not its file has the axis.
        for name in ("z", "c", "t"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, 0)

    @property
    def image_file(self) -> str:
        """File name of the source image."""
        return self.source_relative_path.rsplit("/", 1)[-1]

    @property
    def plane_id(self) -> str:
        """Deterministic plane identifier: path plus the plane's coordinates."""
        return make_plane_id(self.source_relative_path, z=self.z, c=self.c, t=self.t)

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


def make_plane_id(
    source_relative_path: str, *, z: int | None = 0, c: int | None = 0, t: int | None = 0
) -> str:
    """Build a plane identifier such as ``"cond/rep1/img.tif::z=3,c=0,t=0"``.

    All three coordinates are always written; None (an axis the image does not
    have) is written as 0, so a 2D image's identifier is
    ``"<path>::z=0,c=0,t=0"``.
    """
    coords = ",".join(
        f"{name}={0 if value is None else value}" for name, value in (("z", z), ("c", c), ("t", t))
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
