"""Implementations for the spherical basis functions."""

import numpy as np
from ._spherical_harmonics import spherical_harmonics_all
from ._bessel import spherical_jn_all, spherical_hn_all
from ..coordinates import cartesian_2_trigonometric


def spherical_base_all(max_order, position, wavenumber, domain, return_negative_m=True):
    r, cos_theta, _, cos_phi, sin_phi = cartesian_2_trigonometric(position)
    kr = r * np.reshape(wavenumber, np.shape(wavenumber) + (1,) * np.ndim(r))

    if 'singular' in domain.lower() or 'exterior' in domain.lower():
        radial = spherical_hn_all(max_order=max_order, z=kr)
    elif 'regular' in domain.lower() or 'interior' in domain.lower():
        radial = spherical_jn_all(max_order=max_order, z=kr)
    else:
        raise ValueError(f'Unknown domain `{domain}`')

    angular = spherical_harmonics_all(
        max_order=max_order, return_negative_m=True,
        cosine_colatitude=cos_theta, azimuth=cos_phi + 1j * sin_phi,
    )
    # TODO: This will give incorrect values if the compact output form is used!
    return angular.reshape(angular.shape[:2] + (1,) * np.ndim(wavenumber) + angular.shape[2:]) * radial.reshape((-1, 1) + kr.shape)


def singular_base_all(max_order, position, wavenumber, return_negative_m=True):
    return spherical_base_all(max_order=max_order, position=position, wavenumber=wavenumber, return_negative_m=return_negative_m, domain='singular')


def regular_base_all(max_order, position, wavenumber, return_negative_m=True):
    return spherical_base_all(max_order=max_order, position=position, wavenumber=wavenumber, return_negative_m=return_negative_m, domain='regular')
