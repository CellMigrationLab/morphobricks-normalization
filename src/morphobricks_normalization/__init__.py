"""MorphoBricks normalization brick.

Defines and computes the per-plane statistics that normalization needs, reads
and writes their files, and turns pooled statistics into normalization
parameters. Other bricks (preprocessing, patch extraction) call it; it never
reads folders or modifies source images.
"""

from importlib.metadata import PackageNotFoundError, version

from .csv_io import OUTPUT_FILENAME, read_plane_histograms, write_plane_histograms
from .dtypes import UnsupportedDtypeError, default_bin_count
from .histogram import PlaneHistogram, compute_block_histogram, compute_plane_histogram
from .normalization import (
    NormalizationParameters,
    min_max,
    percentile_range,
    robust_z_score,
    z_score,
)
from .pooling import NoValidPixelsError, PooledHistogram, PooledStatistics, pooled_statistics
from .records import SINGLE, PlaneRecord, group_records, make_plane_id

try:
    __version__ = version("morphobricks-normalization")
except PackageNotFoundError:  # running from a source checkout without installing
    __version__ = "0.0.0+unknown"

__all__ = [
    "OUTPUT_FILENAME",
    "SINGLE",
    "NoValidPixelsError",
    "NormalizationParameters",
    "PlaneHistogram",
    "PlaneRecord",
    "PooledHistogram",
    "PooledStatistics",
    "UnsupportedDtypeError",
    "compute_block_histogram",
    "compute_plane_histogram",
    "default_bin_count",
    "group_records",
    "make_plane_id",
    "min_max",
    "percentile_range",
    "pooled_statistics",
    "read_plane_histograms",
    "robust_z_score",
    "write_plane_histograms",
    "z_score",
]
