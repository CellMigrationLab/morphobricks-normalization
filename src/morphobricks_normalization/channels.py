"""Canonical channel IDs: the channel token of every per-channel column (ADR 0003 HV2).

A channel is named by its canonical channel ID from the confirmed experiment description
(MorphoBricks experiment-v1 §3.1, §4.3), never by its index in the image: the same index can
name different stains in two imaging units (Spec fixture ``v04``), and a unit can lack a
channel another unit has (``v05``, ``v17``). An ID follows the Spec's Token rule
(experiment-v1 §3.2), ``^[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*$``; it may contain ``_``
(``PAX_488``), so column names are never split on ``_`` (:mod:`~morphobricks_normalization.csv_io`).

IDs are compared exactly: nothing is trimmed, re-cased or normalized (experiment-v1 §3.3).
Which IDs exist and which channel of a file each one names is the calling brick's knowledge
(the description); this module only checks that a value is a well-formed ID.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# experiment-v1 §3.2, the Token rule. It also keeps ``;`` (the separator of the ``channels``
# cell) and ``,`` out of an ID.
CHANNEL_ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*")


def check_channel_id(channel: object, where: str = "") -> str:
    """Return ``channel`` if it is a canonical channel ID, else refuse it.

    Parameters
    ----------
    channel
        The value to check.
    where
        Starts the message, e.g. ``"Plane a.tif::z=0,t=0"``; may be empty.

    Returns
    -------
    str
        ``channel``, unchanged.

    Raises
    ------
    ValueError
        For an index (an int), a value that is not text, or text that is not a Token.
    """
    prefix = f"{where}: " if where else ""
    if isinstance(channel, str) and CHANNEL_ID_PATTERN.fullmatch(channel):
        return channel
    if not isinstance(channel, str):
        raise ValueError(
            f"{prefix}a channel is named by its canonical channel ID from the experiment "
            f"description (for example 'DNA'), never by its position in the image; got "
            f"{channel!r}. Give the channel IDs of the imaging unit the image belongs to."
        )
    raise ValueError(
        f"{prefix}{channel!r} is not a canonical channel ID: an ID starts with a letter and "
        "holds letters and digits, with single '_' between parts (for example 'DNA' or "
        "'PAX_488'). Use the channel IDs of the confirmed experiment description."
    )


def check_channel_ids(channels: Iterable[object], where: str = "") -> tuple[str, ...]:
    """Return channel IDs as a tuple, in the given order, refusing bad or repeated IDs.

    Parameters
    ----------
    channels
        The IDs to check; text, not a single string.
    where
        Starts the message; may be empty.

    Returns
    -------
    tuple of str
        The IDs, unchanged and in order.

    Raises
    ------
    ValueError
        For a single string instead of a list, an ID that :func:`check_channel_id`
        refuses, or an ID given twice.
    """
    prefix = f"{where}: " if where else ""
    if isinstance(channels, str):
        raise ValueError(
            f"{prefix}expected a list of channel IDs, got the text {channels!r}; give "
            f"[{channels!r}] for one channel."
        )
    ids = tuple(check_channel_id(channel, where) for channel in channels)
    repeated = sorted({channel for channel in ids if ids.count(channel) > 1})
    if repeated:
        raise ValueError(
            f"{prefix}channel(s) {', '.join(repeated)} are listed more than once; each "
            "channel of an image has one ID, and each ID names one channel."
        )
    return ids
