"""Implementations for spherical harmonics."""

import numpy as np
from . import legendre


def spherical_harmonics(max_order, colatitude=None, azimuth=None, cosine_colatitude=None, return_negative_m=True):
    if return_negative_m is True:
        return spherical_harmonics(max_order, colatitude=colatitude, azimuth=azimuth, cosine_colatitude=cosine_colatitude, return_negative_m='full')
    if return_negative_m is False:
        return spherical_harmonics(max_order, colatitude=colatitude, azimuth=azimuth, cosine_colatitude=cosine_colatitude, return_negative_m='positive')
    if 'half' in return_negative_m.lower():
        return spherical_harmonics(max_order, colatitude=colatitude, azimuth=azimuth, cosine_colatitude=cosine_colatitude, return_negative_m='positive')
    if 'full' in return_negative_m.lower():
        return positive_to_full(spherical_harmonics(max_order, colatitude=colatitude, azimuth=azimuth, cosine_colatitude=cosine_colatitude, return_negative_m='positive'))
    if 'compact' in return_negative_m.lower():
        return positive_to_compact(spherical_harmonics(max_order, colatitude=colatitude, azimuth=azimuth, cosine_colatitude=cosine_colatitude, return_negative_m='positive'))

    cosine_colatitude = cosine_colatitude if cosine_colatitude is not None else np.cos(colatitude)
    azimuth = azimuth if np.iscomplexobj(azimuth) else np.exp(1j * azimuth)
    angles = np.broadcast(cosine_colatitude, azimuth)

    legendre_values = legendre(max_order, cosine_colatitude, normalization='orthonormal').reshape((max_order + 1, max_order + 1) + (1,) * (angles.ndim - np.ndim(cosine_colatitude)) + np.shape(cosine_colatitude))
    azimuth = azimuth ** np.arange(max_order + 1).reshape([-1] + [1] * angles.ndim)
    harmonics = np.zeros((max_order + 1, max_order + 1) + angles.shape, dtype=complex)
    harmonics[:, :max_order + 1] = legendre_values * azimuth

    return harmonics * (2 * np.pi)**-0.5


def positive_to_full(positive, is_harmonics=True):
    max_mode = np.shape(positive)[1] - 1
    if is_harmonics:
        minus_one_to_m = (-1) ** np.arange(max_mode, 0, -1).reshape([-1] + [1] * (np.ndim(positive) - 2))
        negative = np.conj(positive[:, -1:0:-1]) * minus_one_to_m
    else:
        negative = positive[:, -1:0:-1]
    return np.concatenate([positive, negative], axis=1)


def positive_to_compact(harmonics, is_harmonics=True, inplace=False):
    max_order = harmonics.shape[0] - 1
    out = harmonics if inplace is True else harmonics.copy()
    if is_harmonics:
        minus_one_to_m = np.fromiter(((-1)**m for m in range(1, max_order + 1) for n in range(max_order - m + 1)), dtype=int).reshape([-1] + [1] * (np.ndim(harmonics) - 2))
        negative = minus_one_to_m * np.conj(np.swapaxes(out, 0, 1)[1:, 1:][np.triu_indices(max_order, 0)])
    else:
        negative = np.swapaxes(out, 0, 1)[1:, 1:][np.triu_indices(max_order, 0)]

    out[np.triu_indices(max_order + 1, 1)] = negative
    return out


def compact_to_positive(compact, inplace=False):
    positive = compact if inplace is True else compact.copy()
    positive[np.triu_indices(compact.shape[0], 1)] = 0
    return positive


def full_to_positive(full):
    max_order = full.shape[0] - 1
    positive = full.copy()[:, :max_order + 1]
    positive[np.triu_indices(full.shape[0], 1)] = 0
    return positive


def full_to_compact(full):
    max_order = full.shape[0] - 1
    max_mode = (full.shape[1] - 1) // 2
    if max_order != max_mode:
        raise ValueError("Compact mode only supported for full sets, input has max order {} and max mode {}".format(max_order, max_mode))
    compact = np.zeros((max_order + 1, max_order + 1) + full.shape[2:], dtype=full.dtype)
    compact[np.tril_indices(max_order + 1)] = full[np.tril_indices(max_order + 1)]
    compact[np.triu_indices(max_order + 1, 1)] = np.swapaxes(full[1:, -1:max_order:-1], 0, 1)[np.triu_indices(max_order)]
    return compact


def compact_to_full(compact):
    max_order = compact.shape[0] - 1
    full = np.zeros((max_order + 1, 2 * max_order + 1) + compact.shape[2:], dtype=compact.dtype)
    full[np.tril_indices(max_order + 1)] = compact[np.tril_indices(max_order + 1)]
    np.swapaxes(full[1:, -1:max_order:-1], 0, 1)[np.triu_indices(max_order)] = compact[np.triu_indices(max_order + 1, 1)]
    return full


def show_index_scheme(max_order, return_negative_m=True):
    """Show the (order, mode) pairs in an indexing scheme.

    There are multiple indexing schemes available for plain spherical harmonics
    or expansion coefficients. Below is a description of how the (order, mode)
    is mapped to indices in a matrix A[].

    - Positive
        This scheme only include 0<=n<=N, 0<=m<=n, and is indexed with A[n,m].
        Uses a (N+1, N+1) matrix to store the (N+1)(N+2)/2 values, i.e. the sparsity goes to 50%.
    - Full
        This scheme include all n and m, and is indexed with A[n,m] for all indices.
        This relies on python reverse indexing for m<0, i.e. A[n,-|m|].
        Uses a (N+1, 2N+1) matrix to store the (N+1)^2 values, i.e the sparsity goes to 50%.
    - Compact
        This scheme includes all n and m, and is indexed with A[n,m] for m>=0,
        and A[|m|-1, n] for m<0 (or A[-(|m|+1), n]), i.e. the negative coefficients for a given order
        is stored in a single column.


    Parameters
    ----------
    max_order : int
        The max order up to which to generate the pairs.
    return_negative_m : string (or bool), default True
        Choose an indexing scheme to show. One of:
            'full' (True) - Show the full scheme
            'positive' (False) - Show the positive scheme
            'compact' - show the compact scheme.

    Returns
    -------
    np.ndarray
        Object array comtaining index tuples (order, mode) at the corresponding positions.
    """
    if return_negative_m is True:
        return show_index_scheme(max_order=max_order, return_negative_m='full')
    if return_negative_m is False:
        return show_index_scheme(max_order=max_order, return_negative_m='positive')
    if 'half' in return_negative_m.lower():
        return show_index_scheme(max_order=max_order, return_negative_m='positive')
    if 'positive' in return_negative_m.lower():
        indices = np.full((max_order + 1, max_order + 1), None)
        for n in range(max_order + 1):
            for m in range(n + 1):
                indices[n, m] = (n, m)
        return indices
    if 'full' in return_negative_m.lower():
        indices = np.full((max_order + 1, 2 * max_order + 1), None)
        for n in range(max_order + 1):
            for m in range(n + 1):
                indices[n, m] = (n, m)
                indices[n, -m] = (n, -m)
        return indices
    if 'compact' in return_negative_m.lower():
        indices = np.full((max_order + 1, max_order + 1), None)
        for n in range(max_order + 1):
            for m in range(n + 1):
                indices[n, m] = (n, m)
                indices[m - 1, n] = (n, -m)
        return indices
