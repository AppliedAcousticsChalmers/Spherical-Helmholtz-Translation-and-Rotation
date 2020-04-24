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
