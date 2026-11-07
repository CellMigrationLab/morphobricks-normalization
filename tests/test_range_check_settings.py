"""range_check_issues refuses malformed channels and mixed settings (issue #59)."""

import numpy as np
import pytest

from morphobricks_normalization import (
    RangeCheck,
    range_check_issues,
)


def _narrow(**changes):
    values = {
        "histogram_range": "minmax",
        "central_percentiles": (1.0, 99.8),
        "central_bins": 5,
        "robust_quantiles": (0.01, 99.99),
        "robust_limits": (0, 1),
        "outside_robust_count": 3,
        "robust_central_bins": 900,
        "min_central_bins": 100,
    }
    return RangeCheck(**{**values, **changes})


@pytest.mark.parametrize(
    "change",
    [
        {"histogram_range": "robust"},
        {"robust_quantiles": (1.0, 99.0)},
        {"central_percentiles": (2.0, 99.0)},
        {"min_central_bins": 200},
    ],
)
def test_range_check_issues_refuse_mixed_settings_in_either_order(change):
    pair = [("A", 0, _narrow()), ("B", 0, _narrow(**change))]
    for order in (pair, pair[::-1]):
        with pytest.raises(ValueError, match="channel C0.*A.*B|channel C0.*B.*A") as refused:
            range_check_issues(order)
        assert type(refused.value).__name__ == "IncompatibleRangeChecksError"
    # the same settings in another channel are not mixed
    assert len(range_check_issues([("A", 0, _narrow()), ("B", 1, _narrow(**change))])) == 2


@pytest.mark.parametrize("channel", [True, 1.0, 1.5, "1", -1, None])
def test_range_check_issues_refuse_malformed_channels(channel):
    with pytest.raises(ValueError, match="whole number counted from 0"):
        range_check_issues([("A", channel, _narrow())])


def test_range_check_issues_accept_numpy_integer_channels():
    (issue,) = range_check_issues([("A", np.int64(2), _narrow())])
    assert issue.channel == 2 and type(issue.channel) is int
