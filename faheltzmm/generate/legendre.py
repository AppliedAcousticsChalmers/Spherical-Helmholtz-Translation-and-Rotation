"""Implementations for associated legendre polynomials."""

import numpy as np


def scipy_norm(orders, modes):
    """Calculate normalization factors to convert from orthonormal to the scipy format.

    Multiplying an orthonormal associated Legendre polynomial with this value
    yields results with the same scale as the scipy implementations.
    """
    from scipy.special import factorial
    return (2 * factorial(orders + modes) / (2 * orders + 1) / factorial(orders - modes))**0.5


def complement_norm(x, modes):
    """Calculate normalization factor to convert from complement normalized to orthonormal.

    Multiplying a complement normalized associated Legendre polynomial with this
    values yields the orthonormal equivalent.

    Note
    ----
    This transformation is not invertible at :math:`|x|=1`, e.g. at the poles of
    a spherical surface. The complement normalizations have non-zero values at
    these points, while other normalizations do not.
    """
    modes = np.asarray(modes)
    return (1 - x**2) ** (modes.reshape(modes.shape + (1,) * np.ndim(x)) / 2)


def sectorial(max_order, x, normalization='orthonormal', out=None):
    x = np.asarray(x)
    legendre = out[:max_order + 1] if out is not None else np.zeros((max_order + 1,) + x.shape)

    legendre[0] = 2**-0.5
    for order in range(1, max_order + 1):
        legendre[order] = - ((2 * order + 1) / (2 * order))**0.5 * legendre[order - 1]

    if 'complement' in normalization.lower():
        return legendre
    if 'orthonormal' in normalization.lower():
        legendre *= complement_norm(x, np.arange(max_order + 1))
    elif 'scipy' in normalization.lower():
        legendre *= complement_norm(x, np.arange(max_order + 1)) * scipy_norm(np.arange(max_order + 1), np.arange(max_order + 1)).reshape([-1] + [1] * x.ndim)
    else:
        raise ValueError('Unknown normalization option: `{}`'.format(normalization))
    return legendre
