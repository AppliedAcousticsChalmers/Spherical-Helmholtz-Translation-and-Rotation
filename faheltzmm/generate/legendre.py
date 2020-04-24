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


def order_expansion(sectorial_coefficient, x, mode, max_order, normalization='orthonormal', out=None):
    """Expand the sectorial coefficient of a given mode to higher orders."""
    x = np.asarray(x)
    sectorial_coefficient = np.asarray(sectorial_coefficient)
    n_orders = max_order - mode + 1
    legendre = out[:n_orders] if out is not None else np.zeros((n_orders, ) + sectorial_coefficient.shape)
    legendre[-1] = 0  # The loop wil access the -1 element in the first iteration. If this is a view the value might not be zero.
    legendre[0] = sectorial_coefficient

    for idx, order in enumerate(range(mode + 1, max_order + 1), 1):
        legendre[idx] = (
            ((2 * order - 1) * (2 * order + 1) / (order + mode) / (order - mode))**0.5
            * x * legendre[idx - 1]
            - ((2 * order + 1) * (order - mode - 1) * (order + mode - 1) / (2 * order - 3) / (order + mode) / (order - mode))**0.5
            * legendre[idx - 2]
        )

    if 'complement' in normalization.lower():
        return legendre
    mode_scale = (1 - x**2) ** (mode / 2)
    if 'orthonormal' in normalization.lower():
        legendre *= mode_scale
    elif 'scipy' in normalization.lower():
        legendre *= scipy_norm(np.arange(mode, max_order + 1).reshape([-1] + [1] * np.ndim(x)), mode) * mode_scale
    else:
        raise ValueError('Unknown normalization option: `{}`'.format(normalization))
    return legendre


def mode_expansion(sectorial_coefficient, x, order, normalization='orthonormal', out=None):
    """Expand the sectorial coefficient of a given order to lower modes."""
    x = np.asarray(x)
    x_complement = (1 - x**2)**0.5
    sectorial_coefficient = np.asarray(sectorial_coefficient)
    legendre = out[:order + 1] if out is not None else np.zeros((order + 1, ) + sectorial_coefficient.shape)

    legendre[order] = sectorial_coefficient
    if order > 0:
        legendre[order - 1] = - legendre[order] * (2 * order) ** 0.5 * x  # Cannot rely on clever indexing to get rid of legendre[mode+2] in the first iteration.
    for mode in reversed(range(order - 1)):
        legendre[mode] = - (
            ((order + mode + 2) * (order - mode - 1) / (order - mode) / (order + mode + 1)) ** 0.5
            * legendre[mode + 2] * x_complement**2
            + 2 * (mode + 1) / ((order + mode + 1) * (order - mode))**0.5
            * legendre[mode + 1] * x
        )

    if 'complement' in normalization.lower():
        return legendre
    modes = np.arange(order + 1).reshape([order + 1] + [1] * np.ndim(x))
    mode_scale = x_complement**modes
    if 'orthonormal' in normalization.lower():
        legendre *= mode_scale
    elif 'scipy' in normalization.lower():
        legendre *= scipy_norm(order, modes) * mode_scale
    else:
        raise ValueError('Unknown normalization option: `{}`'.format(normalization))
    return legendre
