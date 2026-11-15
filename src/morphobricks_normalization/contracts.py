"""Shared refusals of the MorphoBricks contracts that every brick can import (ADR 0005, BM10).

``RESOLVER_VERSION_UNSUPPORTED`` is built here, in the lowest brick, because a consumer of
preprocessing's request resolver must be able to refuse a preprocessing release that predates
the resolver contract it needs: normalization imports nothing of preprocessing, so the
constructor works whatever preprocessing is installed. Each consumer holds its own
``SUPPORTED_RESOLVER_VERSIONS``, checks ``provenance.resolver_version`` against it right after
``resolve_request`` returns, and raises what :func:`refuse_resolver_version` returns (or its own
exception type with the code and message copied verbatim). The constructor builds the error; it
does not decide whether to raise.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = [
    "RESOLVER_VERSION_UNSUPPORTED",
    "ResolverVersionError",
    "refuse_resolver_version",
]

RESOLVER_VERSION_UNSUPPORTED = "RESOLVER_VERSION_UNSUPPORTED"
"""Runtime code of a resolver contract version a consumer does not support."""


class ResolverVersionError(ValueError):
    """A resolver contract version the consumer does not support.

    Attributes
    ----------
    code
        Always ``"RESOLVER_VERSION_UNSUPPORTED"``.
    found
        The version the installed resolver reports.
    supported
        The consumer's supported versions, sorted.
    """

    code = RESOLVER_VERSION_UNSUPPORTED

    def __init__(self, message: str, *, found: object, supported: tuple[int, ...]) -> None:
        super().__init__(message)
        self.found = found
        self.supported = supported


def refuse_resolver_version(found: object, supported: Iterable[int]) -> ResolverVersionError:
    """Return the ``RESOLVER_VERSION_UNSUPPORTED`` error: the only constructor of that code.

    Parameters
    ----------
    found
        ``provenance.resolver_version`` of the resolver that ran (written as given).
    supported
        The consumer's ``SUPPORTED_RESOLVER_VERSIONS``: a non-empty collection of integers,
        written as a sorted list without repeats.

    Returns
    -------
    ResolverVersionError
        Not raised: the caller raises it (or its own type, code and message verbatim).

    Raises
    ------
    ValueError
        ``supported`` is empty or holds something other than integers (a programming error,
        no code).
    """
    versions = tuple(supported)
    if not versions or not all(type(v) is int for v in versions):
        raise ValueError(
            "supported must be a non-empty collection of integer resolver contract versions; "
            f"got {supported!r}."
        )
    ordered = sorted(set(versions))
    return ResolverVersionError(
        f"{RESOLVER_VERSION_UNSUPPORTED}: the installed morphobricks-preprocessing resolver has "
        f"contract version {found}; this consumer supports exactly {ordered}. Install the "
        "morphobricks-preprocessing release it pins; nothing was read or written.",
        found=found,
        supported=tuple(ordered),
    )
