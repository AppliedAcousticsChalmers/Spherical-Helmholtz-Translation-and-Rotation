"""Implementations for spherical harmonics."""

import numpy as np
from ._legendre import legendre_all
from .. import indexing


def spherical_harmonics_all(max_order, colatitude=None, azimuth=None, cosine_colatitude=None, indexing_scheme='natural'):
    """Calculate all spherical harmonics up to a given max order.

    Parameters
    ----------
    max_order : int
        The maximum order to calculate, inclusive.
    colatitude : ndarray
        The colatitude angles at which to calculate the spherical harmonics. Have to be broadcastable with azimuth.
    azimuth : ndarray
        The azimuth angles at which to calculate the spherical harmonics. Have to be broadcastable with colatitude.
        This can optionally be given as a complex value `exp(1j * azimuth)`.
    cosine_colatitude : ndarray, optional
        Can be given instead of the colatitude.
    indexing_scheme : str
        Chooses the indexing scheme of the output. See `indexing.expansions` for more details.

    Returns
    -------
    Y_n^m, complex ndarray
        The calculated spherical harmonics.

    Note
    ----
    This function is optimized for calculation of all spherical harmonics, and will not have good performace
    for e.g. zonal or sectorial harmonics only.
    """
    cosine_colatitude = cosine_colatitude if cosine_colatitude is not None else np.cos(colatitude)
    azimuth = azimuth if np.iscomplexobj(azimuth) else np.exp(1j * azimuth)
    angles = np.broadcast(cosine_colatitude, azimuth)

    legendre_values = legendre_all(max_order, cosine_colatitude, normalization='orthonormal').reshape((max_order + 1, max_order + 1) + (1,) * (angles.ndim - np.ndim(cosine_colatitude)) + np.shape(cosine_colatitude))
    azimuth = azimuth ** np.arange(max_order + 1).reshape([-1] + [1] * angles.ndim)

    if 'non' in indexing_scheme.lower() and 'negative' in indexing_scheme.lower():
        harmonics = (legendre_values * azimuth)[indexing.expansions(max_order, 'natural', indexing_scheme)]
    else:
        harmonics = np.zeros((max_order + 1, 2 * max_order + 1) + angles.shape, dtype=complex)
        harmonics[:, :max_order + 1] = legendre_values * azimuth
        negative_indices = indexing.expansions(max_order, 'natural', 'negative')
        positive_indices = indexing.expansions(max_order, 'natural', 'positive')
        minus_one_to_m = (-1) ** positive_indices[1].reshape([-1] + [1] * (angles.ndim))
        harmonics[negative_indices] = harmonics[positive_indices].conj() * minus_one_to_m
        harmonics = indexing.expansions(harmonics, 'natural', indexing_scheme)
    return harmonics * (2 * np.pi)**-0.5
