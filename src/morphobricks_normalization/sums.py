"""Sums whose result never depends on the CPU thread count (issue #67).

``np.dot`` hands a long float64 dot product to the BLAS library, which splits it across its
threads and adds the partial sums in an order that depends on how many threads it has
(OpenBLAS threads a dot product of more than 10,000 elements). A pooled mean or standard
deviation, and so a z-score fit, then changed in its last bits with ``OMP_NUM_THREADS`` or
``OPENBLAS_NUM_THREADS``. :func:`weighted_sum` multiplies element by element (each product
rounded once, as in ``np.dot``) and adds the products with :func:`math.fsum`, which returns
their correctly rounded sum: the same bits for any thread count, BLAS library, NumPy version or
platform, and never less accurate than the dot product it replaces.
"""

from __future__ import annotations

import math

import numpy as np


def weighted_sum(weights: np.ndarray, values: np.ndarray) -> float:
    """Return ``sum(weights * values)``, correctly rounded and independent of threads.

    Parameters
    ----------
    weights, values
        One-dimensional arrays of the same length (counts or weights, and the values they
        weigh); converted to float64 before multiplying, as ``np.dot`` does.

    Returns
    -------
    total : float
        The correctly rounded sum of the float64 products.
    """
    products = np.multiply(weights, values, dtype=np.float64)
    return math.fsum(products.tolist())
