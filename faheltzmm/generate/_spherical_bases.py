"""Implementations for the spherical basis functions."""

import numpy as np
from ._spherical_harmonics import spherical_harmonics_all
from ._bessel import spherical_jn_all, spherical_hn_all, spherical_yn_all
from ..coordinates import cartesian_2_trigonometric
from .. import indexing


def spherical_base_all(max_order, position, wavenumber, domain, indexing_scheme='natural'):
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
