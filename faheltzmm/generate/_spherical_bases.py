"""Implementations for the spherical basis functions."""

import numpy as np
from ._spherical_harmonics import spherical_harmonics_all
from ._bessel import spherical_jn_all, spherical_hn_all, spherical_yn_all
from ..coordinates import cartesian_2_trigonometric
from .. import indexing


def spherical_base_all(max_order, position, wavenumber, domain, indexing_scheme='natural'):
    r"""Calculate all spherical basis functions up to a given max order.

    This calculates the regular :math:`(R_n^m)` and/or singular :math:`(S_n^m)`spherical basis functions,
    defined as

    .. math::

        R_n^m(\vec r) = j_n(kr) Y_n^m(\theta, \phi)
        S_n^m(\vec r) = h_n(kr) Y_n^m(\theta, \phi)

    at the cartesian positions `position`, for the wavenumbers in `waveumber`, up to and including order `n <= max_order`.
    Selection of the regular basis function is done using `domain='regular'` or `domain='interior'`, and selection
    of the singular basis function is done using `domain='singular'` or `domain='exterior'`.
    Simultaneous calculation of both can be done using `domain='both'` or `domain='all'`,
    in which case the output is `(regular, singular).

    Parameters
    ----------
    max_order : int
        The maximum order to calculate, inclusive.
    position : ndarray
        The cartesian positions at which to calculate. The first axis corresponds to x, y, z.
    wavenumber: ndarray
        The wavenumbers at which to calculate. A single value or an ndarray can be given.
    domain : str
        The domain for which to calculate the bases, see above.
    indexing_scheme : str
        Chooses the indexing scheme of the output. See `indexing.expansions` for more details.

    Returns
    -------
    F_n^m, complex ndarray
        The calculated basis functions harmonics.

    Note
    ----
    This function is optimized for calculation of all spherical bases, and will not have good performace
    for e.g. zonal or sectorial components only.
    """
    r, cos_theta, _, cos_phi, sin_phi = cartesian_2_trigonometric(position)
    kr = r * np.reshape(wavenumber, np.shape(wavenumber) + (1,) * np.ndim(r))

    radial_indices = indexing.expansions(max_order, 'natural', indexing_scheme)[0]

    if 'all' in domain.lower() or 'both' in domain.lower():
        regular = spherical_jn_all(max_order=max_order, z=kr)[radial_indices]
        singular = regular + 1j * spherical_yn_all(max_order=max_order, z=kr)[radial_indices]
        radial = np.stack([regular, singular], axis=0)
    elif 'singular' in domain.lower() or 'exterior' in domain.lower():
        radial = spherical_hn_all(max_order=max_order, z=kr)[radial_indices]
    elif 'regular' in domain.lower() or 'interior' in domain.lower():
        radial = spherical_jn_all(max_order=max_order, z=kr)[radial_indices]
    else:
        raise ValueError(f'Unknown domain `{domain}`')

    angular = spherical_harmonics_all(
        max_order=max_order, indexing_scheme=indexing_scheme,
        cosine_colatitude=cos_theta, azimuth=cos_phi + 1j * sin_phi,
    )
    return angular.reshape(angular.shape[:np.ndim(radial_indices)] + (1,) * np.ndim(wavenumber) + angular.shape[np.ndim(radial_indices):]) * radial


def singular_base_all(max_order, position, wavenumber, indexing_scheme='natural'):
    return spherical_base_all(max_order=max_order, position=position, wavenumber=wavenumber, indexing_scheme=indexing_scheme, domain='singular')


def regular_base_all(max_order, position, wavenumber, indexing_scheme='fulnaturall'):
    return spherical_base_all(max_order=max_order, position=position, wavenumber=wavenumber, indexing_scheme=indexing_scheme, domain='regular')
