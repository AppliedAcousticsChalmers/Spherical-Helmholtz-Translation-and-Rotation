import faheltzmm.generate._legendre
import scipy.special
import numpy as np
import pytest

nice_x = [np.linspace(-0.9, 0.9, 50), np.random.uniform(low=-0.9, high=0.9, size=(200, 200))]
difficult_x = [-1, 0, 1]

@pytest.mark.parametrize('x', nice_x + difficult_x)
@pytest.mark.parametrize('max_order', [8, 12, 60])
def test_sectorial(x, max_order):
    implemented = faheltzmm.generate._legendre.sectorial(max_order, x)
    implemented_scipy_norm = faheltzmm.generate._legendre.sectorial(max_order, x, normalization='scipy')
    out_arr = np.zeros(implemented.shape)
    out_arr_scipy = np.zeros(implemented.shape)
    faheltzmm.generate._legendre.sectorial(max_order, x, out=out_arr)
    faheltzmm.generate._legendre.sectorial(max_order, x, out=out_arr_scipy, normalization='scipy')

    x = np.asarray(x)
    orders = np.arange(max_order + 1).reshape([max_order + 1] + [1] * x.ndim)
    applied_norm = (2 * scipy.special.factorial(2 * orders) / (2 * orders + 1))**0.5
    scipy_lpmv = scipy.special.lpmv(orders, orders, x)

    np.testing.assert_allclose(scipy_lpmv, implemented_scipy_norm)
    np.testing.assert_allclose(implemented * applied_norm, implemented_scipy_norm)
    np.testing.assert_allclose(implemented, out_arr)
    np.testing.assert_allclose(scipy_lpmv, out_arr_scipy)


@pytest.mark.parametrize('x', nice_x + difficult_x)
@pytest.mark.parametrize('max_order', [8, 12, 60])
@pytest.mark.parametrize('mode', [0, 4, 8])
def test_order_expansion(x, mode, max_order):
    sectorial = faheltzmm.generate._legendre.sectorial(mode, x, normalization='complement')[-1]

    implemented_orthonormal = faheltzmm.generate._legendre.order_expansion(sectorial, x, mode, max_order, normalization='orthonormal')
    implemented_complement = faheltzmm.generate._legendre.order_expansion(sectorial, x, mode, max_order, normalization='complement')
    implemented_scipy = faheltzmm.generate._legendre.order_expansion(sectorial, x, mode, max_order, normalization='scipy')

    out_arr_complement = np.zeros(implemented_complement.shape)
    out_arr_orthonormal = np.zeros(implemented_orthonormal.shape)
    out_arr_scipy = np.zeros(implemented_scipy.shape)
    faheltzmm.generate._legendre.order_expansion(sectorial, x, mode, max_order, out=out_arr_complement, normalization='complement')
    faheltzmm.generate._legendre.order_expansion(sectorial, x, mode, max_order, out=out_arr_orthonormal, normalization='orthonormal')
    faheltzmm.generate._legendre.order_expansion(sectorial, x, mode, max_order, out=out_arr_scipy, normalization='scipy')

    orders = np.arange(mode, max_order + 1).reshape([-1] + [1] * np.ndim(x))
    scipy_norm = (2 * scipy.special.factorial(mode + orders) / (2 * orders + 1) / scipy.special.factorial(orders - mode))**0.5
    complementary_norm = ((1 - x**2) ** 0.5) ** mode
    scipy_lpmv = scipy.special.lpmv(mode, orders, x)

    np.testing.assert_allclose(scipy_lpmv, implemented_scipy)
    np.testing.assert_allclose(implemented_orthonormal * scipy_norm, implemented_scipy)
    np.testing.assert_allclose(implemented_orthonormal, implemented_complement * complementary_norm)
    np.testing.assert_allclose(implemented_complement, out_arr_complement)
    np.testing.assert_allclose(implemented_orthonormal, out_arr_orthonormal)
    np.testing.assert_allclose(implemented_scipy, out_arr_scipy)


@pytest.mark.parametrize('x', nice_x + difficult_x)
@pytest.mark.parametrize('order', [8, 12, 60])
def test_mode_expansion(x, order):
    sectorial = faheltzmm.generate._legendre.sectorial(order, x, normalization='complementary')[-1]

    implemented_orthonormal = faheltzmm.generate._legendre.mode_expansion(sectorial, x, order, normalization='orthonormal')
    implemented_complement = faheltzmm.generate._legendre.mode_expansion(sectorial, x, order, normalization='complementary')
    implemented_scipy = faheltzmm.generate._legendre.mode_expansion(sectorial, x, order, normalization='scipy')

    out_arr_complement = np.zeros(implemented_orthonormal.shape)
    out_arr_orthonormal = np.zeros(implemented_orthonormal.shape)
    out_arr_scipy = np.zeros(implemented_orthonormal.shape)
    faheltzmm.generate._legendre.mode_expansion(sectorial, x, order, out=out_arr_complement, normalization='complement')
    faheltzmm.generate._legendre.mode_expansion(sectorial, x, order, out=out_arr_orthonormal, normalization='orthonormal')
    faheltzmm.generate._legendre.mode_expansion(sectorial, x, order, out=out_arr_scipy, normalization='scipy')

    modes = np.arange(order + 1).reshape([-1] + [1] * np.ndim(x))
    complementary_norm = ((1 - x**2) ** 0.5) ** modes
    scipy_norm = (2 * scipy.special.factorial(modes + order) / (2 * order + 1) / scipy.special.factorial(order - modes))**0.5
    scipy_lpmv = scipy.special.lpmv(modes, order, x)

    np.testing.assert_allclose(scipy_lpmv, implemented_scipy)
    np.testing.assert_allclose(implemented_orthonormal * scipy_norm, implemented_scipy)
    np.testing.assert_allclose(implemented_orthonormal, implemented_complement * complementary_norm)
    np.testing.assert_allclose(implemented_complement, out_arr_complement)
    np.testing.assert_allclose(implemented_orthonormal, out_arr_orthonormal)
    np.testing.assert_allclose(implemented_scipy, out_arr_scipy)


@pytest.mark.parametrize('max_order', [0, 1, 5, 12])
@pytest.mark.parametrize('x', nice_x + difficult_x)
@pytest.mark.parametrize('direction', ['orders', 'modes'])
def test_legendre_all(max_order, x, direction):
    implemented_complement = faheltzmm.generate._legendre.legendre_all(max_order, x, normalization='complement', direction=direction)
    implemented_orthonormal = faheltzmm.generate._legendre.legendre_all(max_order, x, normalization='orthonormal', direction=direction)
    implemented_scipy = faheltzmm.generate._legendre.legendre_all(max_order, x, normalization='scipy', direction=direction)

    out_arr_complement = np.zeros(implemented_orthonormal.shape)
    out_arr_orthonormal = np.zeros(implemented_orthonormal.shape)
    out_arr_scipy = np.zeros(implemented_orthonormal.shape)
    faheltzmm.generate._legendre.legendre_all(max_order, x, out=out_arr_complement, normalization='complement', direction=direction)
    faheltzmm.generate._legendre.legendre_all(max_order, x, out=out_arr_orthonormal, normalization='orthonormal', direction=direction)
    faheltzmm.generate._legendre.legendre_all(max_order, x, out=out_arr_scipy, normalization='scipy', direction=direction)

    orders = np.arange(max_order + 1).reshape([max_order + 1, 1] + [1] * np.ndim(x))
    modes = np.arange(max_order + 1).reshape([1, max_order + 1] + [1] * np.ndim(x))
    complementary_norm = ((1 - x**2) ** 0.5) ** modes
    scipy_norm = (2 * scipy.special.factorial(modes + orders) / (2 * orders + 1) / np.where(orders < modes, 1, scipy.special.factorial(orders - modes)))**0.5
    scipy_lpmv = scipy.special.lpmv(modes, orders, x)

    np.testing.assert_allclose(scipy_lpmv, implemented_scipy)
    np.testing.assert_allclose(implemented_orthonormal * scipy_norm, implemented_scipy)
    np.testing.assert_allclose(implemented_orthonormal, implemented_complement * complementary_norm)
    np.testing.assert_allclose(implemented_complement, out_arr_complement)
    np.testing.assert_allclose(implemented_orthonormal, out_arr_orthonormal)
    np.testing.assert_allclose(implemented_scipy, out_arr_scipy)
