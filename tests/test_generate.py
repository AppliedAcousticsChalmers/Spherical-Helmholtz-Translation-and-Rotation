import pytest
import numpy as np
import faheltzmm.generate
import faheltzmm.indexing
import faheltzmm.coordinates

np.random.seed(0)
max_order = 12
positions = np.random.normal(size=(3, 11, 17))
r, theta, phi = faheltzmm.coordinates.cartesian_2_spherical(positions)
wavenumbers = np.random.uniform(low=0.1, high=2 * max_order, size=(5))


@pytest.mark.parametrize('base', ['regular', 'singular'])
def test_bases_generation(base):
    if base == 'regular':
        radial_func = faheltzmm.generate.spherical_jn
        bases = faheltzmm.generate.regular_base_set
        modes = faheltzmm.generate.regular_modes
        base = faheltzmm.generate.regular_base
    elif base == 'singular':
        radial_func = faheltzmm.generate.spherical_hn
        bases = faheltzmm.generate.singular_base_set
        modes = faheltzmm.generate.singular_modes
        base = faheltzmm.generate.singular_base

    manual = []
    order_mode = []
    order_all_modes = []
    all_orders = bases(max_order, positions, wavenumbers)

    sph_idx = faheltzmm.indexing.SphericalHarmonicsIndexer(max_order)
    for order in sph_idx.orders:
        order_all_modes.append(modes(order, positions, wavenumbers))
        for mode in sph_idx.modes:
            order_mode.append(base(order, mode, positions, wavenumbers))
            manual.append([])
            for wavenumber in wavenumbers:
                manual[-1].append(radial_func(order, wavenumber * r) * faheltzmm.generate.sph_harm(mode, order, theta, phi))
    order_all_modes = np.concatenate(order_all_modes, axis=0)
    order_mode = np.stack(order_mode, axis=0)
    manual = np.stack(manual, axis=0)

    np.testing.assert_allclose(manual, all_orders)
    np.testing.assert_allclose(manual, order_mode)
    np.testing.assert_allclose(manual, order_all_modes)
