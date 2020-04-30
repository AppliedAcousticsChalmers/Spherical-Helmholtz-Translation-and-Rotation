import pytest
import numpy as np
import scipy.special

import faheltzmm.generate._spherical_bases
import faheltzmm.generate._spherical_harmonics
import faheltzmm.generate._bessel
import faheltzmm.coordinates

np.random.seed(0)
max_order = 12
positions = np.random.normal(size=(3, 11, 17))
r, theta, phi = faheltzmm.coordinates.cartesian_2_spherical(positions)
wavenumbers = np.random.uniform(low=0.1, high=2 * max_order, size=(5))


@pytest.mark.parametrize('base', ['regular', 'singular'])
def test_bases_generation(base):
    if base == 'regular':
        radial_func = faheltzmm.generate._bessel.spherical_jn
        bases = faheltzmm.generate._spherical_bases.regular_base_all
    elif base == 'singular':
        radial_func = faheltzmm.generate._bessel.spherical_hn
        bases = faheltzmm.generate._spherical_bases.singular_base_all

    kr = np.reshape(wavenumbers, np.shape(wavenumbers) + np.ndim(r) * (1,)) * r
    manual = np.zeros((max_order + 1, 2 * max_order + 1) + np.shape(kr), dtype=complex)
    all_orders = bases(max_order, positions, wavenumbers)

    for order in range(max_order + 1):
        manual[order, 0] = radial_func(order, kr) * scipy.special.sph_harm(0, order, phi, theta)[None]
        for mode in range(1, order + 1):
            manual[order, mode] = radial_func(order, kr) * scipy.special.sph_harm(mode, order, phi, theta)
            manual[order, -mode] = radial_func(order, kr) * scipy.special.sph_harm(-mode, order, phi, theta)

    np.testing.assert_allclose(manual, all_orders)
