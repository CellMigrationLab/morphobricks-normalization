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
    pair = [("A", "DNA", _narrow()), ("B", "DNA", _narrow(**change))]
    for order in (pair, pair[::-1]):
        with pytest.raises(ValueError, match="channel DNA.*A.*B|channel DNA.*B.*A") as refused:
            range_check_issues(order)
        assert type(refused.value).__name__ == "IncompatibleRangeChecksError"
    # the same settings in another channel are not mixed
    pairs = [("A", "DNA", _narrow()), ("B", "ACTIN", _narrow(**change))]
    assert [i.channel for i in range_check_issues(pairs)] == ["ACTIN", "DNA"]


@pytest.mark.parametrize("channel", [True, 1.0, 1.5, 0, np.int64(2), None])
def test_range_check_issues_refuse_channel_indices(channel):
    with pytest.raises(ValueError, match="never by its position"):
        range_check_issues([("A", channel, _narrow())])


@pytest.mark.parametrize("channel", ["1", "", "DNA "])
def test_range_check_issues_refuse_text_that_is_not_an_id(channel):
    with pytest.raises(ValueError, match="is not a canonical channel ID"):
        range_check_issues([("A", channel, _narrow())])


def test_range_check_issues_name_the_channel_id():
    (issue,) = range_check_issues([("A", "PAX_488", _narrow())])
    assert issue.channel == "PAX_488" and "channel PAX_488 " in issue.message
