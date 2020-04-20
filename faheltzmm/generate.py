"""Module to generate various basic sets of coefficients."""

import numpy as np
from scipy.special import sph_harm, spherical_jn, spherical_yn
from .coordinates import cartesian_2_spherical
from .indexing import SphericalHarmonicsIndexer


def spherical_hn(n, z, derivative=False):
    """Calculate the spherical Hankel function of the first kind.

    See `scipy.special.spherical_jn` and `scipy.special.spherical_yn` for details on the arguments.
    """
    return spherical_jn(n, z, derivative=derivative) + 1j * spherical_yn(n, z, derivative=derivative)


def base_mesh(orders, modes, position, wavenumber, base_type):
    # DOCS: Write up the shape of the inputs and output for base_mesh.
    # We output of dimensions (orders and modes, wavenumbers, positions)
    r, theta, phi = cartesian_2_spherical(position)
    orders, modes, wavenumber = np.asarray(orders), np.asarray(modes), np.asarray(wavenumber)

    kr = wavenumber.reshape(wavenumber.shape + (1,) * r.ndim) * r
    unique_orders, unique_indices = np.unique(orders, return_inverse=True)
    unique_orders.shape = unique_orders.shape + (1,) * kr.ndim
    if 'regular' in base_type.lower():
        radial_function = spherical_jn(unique_orders, kr)
    elif 'singular' in base_type.lower():
        radial_function = spherical_hn(unique_orders, kr)
    else:
        raise ValueError('Unknown base type: `' + base_type + '`')

    radial_function = radial_function[unique_indices].reshape(orders.shape + kr.shape)
    angular_function = sph_harm(
        modes.reshape(modes.shape + (1,) * (r.ndim + wavenumber.ndim)),
        orders.reshape(orders.shape + (1,) * (r.ndim + wavenumber.ndim)),
        theta, phi
    )

    return radial_function * angular_function


def singular_base(order, mode, position, wavenumber):
    r"""Calculate the singular basis function for sound fields.

    The singular basis function is defined as

    .. math:: S_n^m(\vec r) = h_n(kr) Y_n^m(\theta, \phi)

    where :math:`h_n = j_n + i y_n` is the spherical Hankel function of the first kind.
    """
    return base_mesh(order, mode, position, wavenumber, base_type='singular')


def singular_modes(order, position, wavenumber):
    """Calculate the singular basis functions of a given order for all modes."""
    _, modes = zip(*SphericalHarmonicsIndexer(order, order))
    return base_mesh(order, modes, position, wavenumber, base_type='singular')


def singular_base_set(max_order, position, wavenumber):
    """Calculate the full set of singular basis functions up to a given order."""
    orders, modes = zip(*SphericalHarmonicsIndexer(max_order))
    return base_mesh(orders, modes, position, wavenumber, base_type='singular')


def regular_base(order, mode, position, wavenumber):
    r"""Calculate the regular basis function for sound fields.

    The regular basis function is defined as

    .. math:: R_n^m(\vec r) = j_n(kr) Y_n^m(\theta, \phi)

    where :math:`j_n` is the spherical Bessel function.
    """
    return base_mesh(order, mode, position, wavenumber, base_type='regular')


def regular_modes(order, position, wavenumber):
    """Calculate the regular basis functions of a given order for all modes."""
    _, modes = zip(*SphericalHarmonicsIndexer(order, order))
    return base_mesh(order, modes, position, wavenumber, base_type='regular')


def regular_base_set(max_order, position, wavenumber):
    """Calculate the full set of regular basis functions up to a given order."""
    orders, modes = zip(*SphericalHarmonicsIndexer(max_order))
    return base_mesh(orders, modes, position, wavenumber, base_type='regular')
