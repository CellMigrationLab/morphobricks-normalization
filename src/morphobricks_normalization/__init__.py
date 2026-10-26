"""MorphoBricks normalization brick.

Defines and computes the per-plane statistics that normalization needs, reads
and writes their files, and turns pooled statistics into normalization
parameters. Other bricks (preprocessing, patch extraction) call it; it never
reads folders or modifies source images.
"""

from ._version import __version__
from .csv_io import OUTPUT_FILENAME, read_plane_histograms, write_plane_histograms
from .dtypes import UnsupportedDtypeError, default_bin_count
from .fitting import Parameters, fit
from .histogram import PlaneHistogram, compute_block_histogram, compute_plane_histogram
from .issues import Issue
from .normalization import (
    NormalizationParameters,
    min_max,
    percentile_range,
    robust_z_score,
    z_score,
)
from .pooling import (
    NoValidPixelsError,
    PooledHistogram,
    PooledStatistics,
    WeightedHistogram,
    pooled_statistics,
)
from .records import SINGLE, PlaneRecord, group_records, make_plane_id
from .scopes import group_key, group_name, scope_groups
from .settings import MethodNotAllowedError, NormalizationSettings, check_settings

__all__ = [
    "OUTPUT_FILENAME",
    "SINGLE",
    "Issue",
    "MethodNotAllowedError",
    "NoValidPixelsError",
    "NormalizationParameters",
    "NormalizationSettings",
    "Parameters",
    "PlaneHistogram",
    "PlaneRecord",
    "PooledHistogram",
    "PooledStatistics",
    "UnsupportedDtypeError",
    "WeightedHistogram",
    "__version__",
    "check_settings",
    "compute_block_histogram",
    "compute_plane_histogram",
    "default_bin_count",
    "fit",
    "group_key",
    "group_name",
    "group_records",
    "make_plane_id",
    "min_max",
    "percentile_range",
    "pooled_statistics",
    "read_plane_histograms",
    "robust_z_score",
    "scope_groups",
    "write_plane_histograms",
    "z_score",
]
