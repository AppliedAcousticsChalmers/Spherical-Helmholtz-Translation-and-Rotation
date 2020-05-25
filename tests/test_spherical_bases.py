import pytest
import numpy as np
import scipy.special

import faheltzmm.generate._spherical_bases
import faheltzmm.generate._spherical_harmonics
import faheltzmm.generate._bessel
import faheltzmm.coordinates
import faheltzmm.indexing

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


@pytest.mark.parametrize('base', ['regular', 'singular'])
@pytest.mark.parametrize('indexing_scheme', [
    'natural', 'compact', 'linear', 'natural non-negative', 'compact non-negative',
    'zonal', 'positive', 'negative', 'non-positive', 'non-negative',
    'sectorial', 'sectorial positive', 'sectorial negative', 'sectorial non-positive', 'sectorial non-negative',
    'tesseral', 'positive tesseral', 'negative tesseral',
])
def test_indexing_schemes(base, indexing_scheme):
    natural = faheltzmm.generate.spherical_base_all(max_order, positions, wavenumbers, base, 'natural')
    scheme = faheltzmm.generate.spherical_base_all(max_order, positions, wavenumbers, base, indexing_scheme)
    natural_2_scheme = faheltzmm.indexing.expansions(natural, 'natural', indexing_scheme)

    np.testing.assert_allclose(scheme, natural_2_scheme)


@pytest.mark.parametrize('order', [0, 1, 3, 7])
@pytest.mark.parametrize('indexing_scheme', ['natural', 'compact', 'linear'])
@pytest.mark.parametrize('position, wavenumber', [
    (np.random.normal(size=3), np.random.uniform()),
    (np.random.normal(size=(3, 5)), np.random.uniform()),
    (np.random.normal(size=(3, 5, 7)), np.random.uniform()),
    (np.random.normal(size=3), np.random.uniform(size=(5))),
    (np.random.normal(size=3), np.random.uniform(size=(5, 7))),
    (np.random.normal(size=(3, 5)), np.random.uniform(size=(11))),
    (np.random.normal(size=(3, 2, 3)), np.random.uniform(size=(4, 5))),
])
def test_dual_domain(order, position, wavenumber, indexing_scheme):
    regular_singular = faheltzmm.generate.spherical_base_all(order, position, wavenumber, 'both', indexing_scheme)
    regular = faheltzmm.generate.spherical_base_all(order, position, wavenumber, 'regular', indexing_scheme)
    singular = faheltzmm.generate.spherical_base_all(order, position, wavenumber, 'singular', indexing_scheme)
    np.testing.assert_allclose(regular, regular_singular[0])
    np.testing.assert_allclose(singular, regular_singular[1])
