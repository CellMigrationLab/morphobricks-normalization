"""MorphoBricks normalization brick.

Defines and computes the per-plane statistics that normalization needs, reads
and writes their files, turns pooled statistics into normalization
parameters, and checks the images' intensities from those statistics. Other
bricks (preprocessing, patch extraction) call it; it never reads folders or
modifies source images.
"""

from ._version import __version__
from .channels import check_channel_id, check_channel_ids
from .checks import IntensityCheckSettings, intensity_checks, unchecked_channels
from .csv_io import (
    OUTPUT_FILENAME,
    CsvFormatError,
    PlaneHistogramTable,
    read_plane_histogram_table,
    read_plane_histograms,
    write_plane_histograms,
)
from .dtypes import UnsupportedDtypeError, default_bin_count
from .fitting import ChannelScopeIncompleteError, Parameters, fit
from .histogram import (
    HistogramTail,
    PlaneHistogram,
    compute_block_histogram,
    compute_plane_histogram,
    histogram_with_range_check,
)
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
from .ranges import (
    HistogramSettings,
    IncompatibleRangeChecksError,
    RangeCheck,
    range_check_issues,
)
from .records import (
    SINGLE,
    ChannelMeasurementError,
    ChannelMeasurementFailure,
    PlaneRecord,
    check_measurements,
    group_records,
    make_plane_id,
)
from .rgb import apply_rgb, rgb_percentile
from .scopes import group_key, group_name, scope_groups
from .settings import MethodNotAllowedError, NormalizationSettings, check_settings
from .summaries import (
    ImageSummary,
    IncompatibleSummariesError,
    image_summaries,
    iter_image_summaries,
    summary_columns,
)

__all__ = [
    "OUTPUT_FILENAME",
    "SINGLE",
    "ChannelMeasurementError",
    "ChannelMeasurementFailure",
    "ChannelScopeIncompleteError",
    "CsvFormatError",
    "HistogramSettings",
    "HistogramTail",
    "ImageSummary",
    "IncompatibleRangeChecksError",
    "IncompatibleSummariesError",
    "IntensityCheckSettings",
    "Issue",
    "MethodNotAllowedError",
    "NoValidPixelsError",
    "NormalizationParameters",
    "NormalizationSettings",
    "Parameters",
    "PlaneHistogram",
    "PlaneHistogramTable",
    "PlaneRecord",
    "PooledHistogram",
    "PooledStatistics",
    "RangeCheck",
    "UnsupportedDtypeError",
    "WeightedHistogram",
    "__version__",
    "apply_rgb",
    "check_channel_id",
    "check_channel_ids",
    "check_measurements",
    "check_settings",
    "compute_block_histogram",
    "compute_plane_histogram",
    "default_bin_count",
    "fit",
    "group_key",
    "group_name",
    "group_records",
    "histogram_with_range_check",
    "image_summaries",
    "iter_image_summaries",
    "intensity_checks",
    "make_plane_id",
    "min_max",
    "percentile_range",
    "pooled_statistics",
    "range_check_issues",
    "read_plane_histogram_table",
    "read_plane_histograms",
    "rgb_percentile",
    "robust_z_score",
    "scope_groups",
    "summary_columns",
    "unchecked_channels",
    "write_plane_histograms",
    "z_score",
]
