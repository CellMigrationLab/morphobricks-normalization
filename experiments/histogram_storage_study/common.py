"""Datasets, plane records and histogram variants of the storage study (not for merge).

Every histogram is computed with this repository's own code
(``compute_plane_histogram`` / ``histogram_with_range_check``); nothing here bins a pixel.

Datasets (folder layout made by the commands in README.md, under ``DATA``):

``study``  preprocessing's ``tools/benchmark_run_workers.py`` data at scale 1 (PR #318): 14 seeded
           uint16 images (TCYX, 2 channels), 76 channel planes, values 0..3499.
``cmu1``   OpenSlide test data CMU-1 (Aperio, CC0): 8 level-0 crops of 2048 x 2048 with the most
           tissue, and CMU-1-Small-Region (2967 x 2220); uint8 RGB.
``hpmec``  MorphoRelate ``data/hpmec_perfusion``: 2 fields, 4 channels, 2048 x 2048, uint16.
``jump``   JUMP cpg0016 source_4 (CC0), the plates of MorphoRelate ``data/jump_target2``: 3 plates
           (biological repeats) x wells E08 (DMSO) and D19 (FK-866) x sites 1-3, 5 channels,
           1080 x 1080 uint16 (checked against MorphoRelate's ``sources.csv`` SHA-256).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from morphobricks_normalization import (
    HistogramSettings,
    PlaneRecord,
    compute_plane_histogram,
    histogram_with_range_check,
)

DATA = Path(os.environ.get("D9_DATA", "data"))
WORK = Path(os.environ.get("D9_WORK", "work"))

# Histogram variants: name -> (bin_count or None for the default, histogram_range)
VARIANTS = {
    "exact": (None, "minmax"),
    "d4096": (4096, "minmax"),
    "d4096r": (4096, "robust"),
    "d1024": (1024, "minmax"),
    "d1024r": (1024, "robust"),
}
JUMP_CHANNELS = {"ch5": "DNA", "ch4": "ER", "ch3": "RNA", "ch2": "AGP", "ch1": "Mito"}
JUMP_WELLS = {"r05c08": ("DMSO", "E08"), "r04c19": ("FK-866", "D19")}


@dataclass(frozen=True)
class Image:
    """One source image: its labels and how to read it as T, C, Y, X."""

    rel: str
    condition: str
    repeat: str
    techrep: str
    position_id: str
    channels: tuple[str, ...]
    files: tuple[Path, ...]  # one file (all channels) or one file per channel
    kind: str  # tiff_tcyx | zarr_tcyx | tiff_cyx | tiff_yx_per_channel | tiff_rgb | svs_rgb

    def read(self) -> np.ndarray:
        """Return the pixels as (T, C, Y, X)."""
        import tifffile

        if self.kind == "tiff_tcyx":
            return tifffile.imread(self.files[0])
        if self.kind == "zarr_tcyx":
            import zarr

            return np.asarray(zarr.open_array(str(self.files[0] / "0"), mode="r")[...])
        if self.kind == "tiff_cyx":
            return tifffile.imread(self.files[0])[None]
        if self.kind == "tiff_yx_per_channel":
            return np.stack([tifffile.imread(f) for f in self.files])[None]
        if self.kind in ("tiff_rgb", "svs_rgb"):
            with tifffile.TiffFile(self.files[0]) as tif:
                rgb = tif.pages[0].asarray()
            return np.moveaxis(rgb, -1, 0)[None]
        raise ValueError(self.kind)


def images(dataset: str) -> list[Image]:
    """The images of a dataset, in a fixed order."""
    root = DATA / dataset
    out = []
    if dataset == "study":
        for path in sorted((root / "experiment").glob("*/rep1/*")):
            condition = path.parts[-3]
            kind = "zarr_tcyx" if path.name.endswith(".zarr") else "tiff_tcyx"
            stem = path.name.split(".")[0]
            rel = f"{condition}/rep1/{path.name}"
            out.append(
                Image(rel, condition, "rep1", "single", stem, ("ch1", "ch2"), (path,), kind)
            )
    elif dataset == "cmu1":
        files = sorted((root / "crops").glob("*.tif")) + [root / "CMU-1-Small-Region.svs"]
        for path in files:
            kind = "svs_rgb" if path.suffix == ".svs" else "tiff_rgb"
            rel = f"slide/single/{path.name}"
            out.append(
                Image(rel, "slide", "single", "single", path.stem, ("R", "G", "B"), (path,), kind)
            )
    elif dataset == "hpmec":
        for name, condition in (("Position_1", "HPMEC_ctrl"), ("Position_2", "HPMEC_cond2")):
            path = root / f"{name}.tif"
            rel = f"{condition}/single/{name}.tif"
            chans = ("ch0", "ch1", "ch2", "ch3")
            out.append(Image(rel, condition, "single", "single", name, chans, (path,), "tiff_cyx"))
    elif dataset == "jump":
        pattern = re.compile(r"(\d{4}_\d\d_\d\d)_Batch\d+__(r\d\dc\d\d)(f\d\d)p01-ch(\d)")
        fields: dict[tuple[str, str, str], dict[str, Path]] = {}
        for path in sorted(root.glob("*.tiff")):
            date, well, site, ch = pattern.match(path.name).groups()
            fields.setdefault((date, well, site), {})[f"ch{ch}"] = path
        for (date, well, site), by_ch in sorted(fields.items()):
            condition, well_name = JUMP_WELLS[well]
            files = tuple(by_ch[ch] for ch in JUMP_CHANNELS)
            chans = tuple(JUMP_CHANNELS.values())
            pos = f"{date}_{well_name}_{site}"
            rel = f"{condition}/{date}/{well_name}/{pos}.tiff"
            kind = "tiff_yx_per_channel"
            out.append(Image(rel, condition, date, well_name, pos, chans, files, kind))
    else:
        raise ValueError(dataset)
    return out


def image_type(dataset: str) -> str:
    return "rgb" if dataset == "cmu1" else "fluorescence"


def plane_histogram(pixels: np.ndarray, variant: str):
    """The histogram of one channel plane under a variant, and its range check (or None)."""
    bins, histogram_range = VARIANTS[variant]
    if variant == "exact":
        return compute_plane_histogram(pixels), None
    top = int(np.iinfo(pixels.dtype).max) + 1 if pixels.dtype.kind in "ui" else bins
    bins = min(bins, top)  # uint8 allows at most 256 bins: 1,024 and 4,096 do not apply
    settings = HistogramSettings(histogram_range=histogram_range)
    return histogram_with_range_check(
        lambda: (pixels,), pixels.dtype, bins, settings=settings
    )


def records_of(image: Image, data: np.ndarray, variant: str) -> tuple[list[PlaneRecord], list]:
    """One record per T plane of an image (Z is 1 in every dataset here), and range checks."""
    records, checks = [], []
    for t in range(data.shape[0]):
        hists = []
        for c, channel in enumerate(image.channels):
            hist, check = plane_histogram(np.ascontiguousarray(data[t, c]), variant)
            hists.append(hist)
            checks.append((f"{image.rel}::t={t}", channel, check))
        records.append(
            PlaneRecord(
                source_relative_path=image.rel,
                condition=image.condition,
                biological_repeat=image.repeat,
                technical_repeat=image.techrep,
                histograms=hists,
                z=0,
                t=t,
                position_id=image.position_id,
                channels=image.channels,
            )
        )
    return records, checks


def iter_images(dataset: str) -> Iterator[tuple[Image, np.ndarray]]:
    for image in images(dataset):
        yield image, image.read()


def channel_ids(records: list[PlaneRecord]) -> tuple[str, ...]:
    return tuple(sorted({c for r in records for c in r.channels}))
