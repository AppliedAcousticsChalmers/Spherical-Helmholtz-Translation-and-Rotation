import faheltzmm.generate.legendre
import scipy.special
import numpy as np
import pytest

nice_x = [np.linspace(-0.9, 0.9, 50), np.random.uniform(low=-0.9, high=0.9, size=(200, 200))]
difficult_x = [-1, 0, 1]

@pytest.mark.parametrize('x', nice_x + difficult_x)
@pytest.mark.parametrize('max_order', [8, 12, 60])
def test_sectorial(x, max_order):
    implemented = faheltzmm.generate.legendre.sectorial(max_order, x)
    implemented_scipy_norm = faheltzmm.generate.legendre.sectorial(max_order, x, normalization='scipy')
    out_arr = np.zeros(implemented.shape)
    out_arr_scipy = np.zeros(implemented.shape)
    faheltzmm.generate.legendre.sectorial(max_order, x, out=out_arr)
    faheltzmm.generate.legendre.sectorial(max_order, x, out=out_arr_scipy, normalization='scipy')

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
    sectorial = faheltzmm.generate.legendre.sectorial(mode, x, normalization='complement')[-1]

    implemented_orthonormal = faheltzmm.generate.legendre.order_expansion(sectorial, x, mode, max_order, normalization='orthonormal')
    implemented_complement = faheltzmm.generate.legendre.order_expansion(sectorial, x, mode, max_order, normalization='complement')
    implemented_scipy = faheltzmm.generate.legendre.order_expansion(sectorial, x, mode, max_order, normalization='scipy')

    out_arr_complement = np.zeros(implemented_complement.shape)
    out_arr_orthonormal = np.zeros(implemented_orthonormal.shape)
    out_arr_scipy = np.zeros(implemented_scipy.shape)
    faheltzmm.generate.legendre.order_expansion(sectorial, x, mode, max_order, out=out_arr_complement, normalization='complement')
    faheltzmm.generate.legendre.order_expansion(sectorial, x, mode, max_order, out=out_arr_orthonormal, normalization='orthonormal')
    faheltzmm.generate.legendre.order_expansion(sectorial, x, mode, max_order, out=out_arr_scipy, normalization='scipy')

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
