"""Image trees built from experiment descriptions, for the Stage 3 tests (ADR 0003).

A description is read from the vendored MorphoBricks-Spec fixtures (``tests/spec_fixtures``,
pinned in ``PIN.toml``) with ``tomllib``, or written here as the same TOML text (the owner's
case is not a Spec fixture). Only what the statistics need is taken from it: every imaging
unit (repeat, condition, technical repeat) with its folder labels and its ordered ``channels``
list (experiment-v1 §4.4), and the experiment's channel IDs (``used_ids.channels``).

Each image is simulated as the calling brick holds it, a ``(C, Y, X)`` array in the unit's
source channel order; its histograms are computed per C index, exactly as the preprocessing
brick does, and the record is labelled with the unit's list. The pixels of a channel depend on
the image, the time point and the channel *ID* only, never on its index, so two layouts of the
same stains (``v04``) hold the same pixels per ID.
"""

import tomllib
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from morphobricks_normalization import SINGLE, PlaneRecord, compute_plane_histogram

FIXTURES = Path(__file__).parent / "spec_fixtures" / "fixtures" / "valid"
# Brightness per stain, so pooling one stain under another's ID changes every number.
STAIN_LEVEL = {"DNA": 1.0, "ACTIN": 2.5, "TUBULIN": 0.6, "PAXILLIN": 1.7, "PHASE": 4.0}


@dataclass(frozen=True)
class Unit:
    """One imaging unit of a description: where its images are and which channels they have."""

    repeat: str
    condition: str
    source_repeat: str
    technical_repeat: str
    channels: tuple[str, ...]

    def folder(self) -> str:
        parts = [self.condition, self.source_repeat]
        if self.technical_repeat != SINGLE:
            parts.append(self.technical_repeat)
        return "/".join(parts)


@dataclass(frozen=True)
class Description:
    """The units and channel IDs of a description."""

    units: tuple[Unit, ...]
    channel_ids: tuple[str, ...]

    def repeat_groups(self) -> dict[tuple[str, str], str]:
        """``(condition, source repeat)`` to the canonical repeat ID (contract §4.2)."""
        return {(u.condition, u.source_repeat): u.repeat for u in self.units}


def parse(text: str) -> Description:
    data = tomllib.loads(text)
    units = []
    for repeat, entry in sorted(data["repeats"].items()):
        for condition, member in sorted(entry["members"].items()):
            folder = data["conditions"][condition]["source_folder"]
            for _, tech in sorted(member["technical_repeats"].items()):
                units.append(
                    Unit(repeat, folder, member["source_repeat"], tech["source_folder"],
                         tuple(tech["channels"]))
                )  # fmt: skip
    return Description(tuple(units), tuple(data["used_ids"]["channels"]))


def fixture(name: str) -> Description:
    """The description of a vendored Spec fixture, e.g. ``"v04_swapped_channel_order"``."""
    return parse((FIXTURES / f"{name}.toml").read_text(encoding="utf-8"))


def renamed(description: Description, old: str, new: str) -> Description:
    """The same description with channel ID ``old`` renamed ``new`` (no fixture has a ``_``)."""

    def swap(ids):
        return tuple(new if c == old else c for c in ids)

    units = tuple(
        Unit(u.repeat, u.condition, u.source_repeat, u.technical_repeat, swap(u.channels))
        for u in description.units
    )
    return Description(units, swap(description.channel_ids))


def channel_pixels(path: str, t: int, channel: str, repeat: str, shape=(12, 12)) -> np.ndarray:
    """The pixels of one channel plane: a function of the image, time point and channel ID."""
    rng = np.random.default_rng(zlib.crc32(f"{path}|{t}|{channel}".encode()))
    level = STAIN_LEVEL.get(channel, 1.0 + zlib.crc32(channel.encode()) % 7 / 3)
    level *= 1.0 + 0.25 * (zlib.crc32(repeat.encode()) % 4)
    return np.clip(rng.normal(40 * level, 6 * level, shape), 0, 255).astype(np.uint8)


def images(description: Description, n_images: int = 2, n_times: int = 2):
    """Yield ``(unit, path, t, pixels)``, ``pixels`` a ``(C, Y, X)`` array in source order."""
    for unit in description.units:
        for i in range(n_images):
            path = f"{unit.folder()}/img_{i}.tif"
            for t in range(n_times):
                planes = [channel_pixels(path, t, c, unit.repeat) for c in unit.channels]
                yield unit, path, t, np.stack(planes)


def record(unit: Unit, path: str, t: int, pixels: np.ndarray, channels=None) -> PlaneRecord:
    """The record the preprocessing brick writes: histograms by C index, the unit's IDs."""
    return PlaneRecord(
        path,
        unit.condition,
        unit.source_repeat,
        unit.technical_repeat,
        [compute_plane_histogram(plane) for plane in pixels],
        t=t,
        position_id=path.rsplit("/", 1)[-1].removesuffix(".tif"),
        channels=unit.channels if channels is None else channels,
    )


def records(description: Description, **kwargs) -> list[PlaneRecord]:
    return [record(*item) for item in images(description, **kwargs)]


def pixels_of(description: Description, channel: str, where=lambda unit, path, t: True, **kwargs):
    """Every pixel of ``channel`` (by ID) in the images ``where`` selects, as float64."""
    found = [
        pixels[unit.channels.index(channel)].ravel()
        for unit, path, t, pixels in images(description, **kwargs)
        if channel in unit.channels and where(unit, path, t)
    ]
    return np.concatenate(found).astype(np.float64)
