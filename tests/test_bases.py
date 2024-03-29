import shetar.bases
import scipy.special
import numpy as np
import pytest


@pytest.fixture(scope='module', params=[
    0, 1, 2, 3, 5, 8, 13,
    pytest.param(21, marks=pytest.mark.slow),
    pytest.param(34, marks=pytest.mark.slow)
])
def order(request):
    return request.param


@pytest.fixture(scope='module', params=[
    0, np.pi / 2, np.pi, np.linspace(0.1, 0.9, 25) * np.pi,
    pytest.param(np.random.uniform(low=0.1, high=0.9, size=(7, 11)) * np.pi, marks=pytest.mark.slow)
])
def colatitude(request):
    return request.param


@pytest.fixture(scope='module')
def cosine_colatitude(colatitude):
    # The scipy implementation is quantized for small x, so we have to make sure that any x close to zero is identivally zero.
    cos = np.cos(colatitude)
    return np.where(np.abs(cos) < 1e-6, 0, cos)


@pytest.fixture(scope='module', params=[
    0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi,
    np.linspace(0, 2, 51) * np.pi,
    pytest.param(np.random.uniform(low=0, high=2, size=(7, 11)) * np.pi, marks=pytest.mark.slow)
])
def azimuth(request):
    return request.param


@pytest.fixture(scope='module', params=[
    1, np.linspace(1e-3, 10, 25),
    pytest.param(np.random.uniform(low=1e-3, high=1, size=(7, 11)), marks=pytest.mark.slow)
])
def radius(request):
    return request.param


@pytest.fixture(scope='module', params=[
    np.random.normal(size=3), np.random.normal(size=(25, 3)),
    pytest.param(np.random.normal(size=(7, 11, 3)), marks=pytest.mark.slow)
])
def position(request):
    return request.param


@pytest.fixture(scope='module')
def wavenumber(position):
    return np.random.uniform(low=1e-3, high=1, size=position.shape[:-1])


def test_legendre_values(order, cosine_colatitude):
    associated_legendre = shetar.bases.AssociatedLegendrePolynomials(order=order, x=cosine_colatitude)
    legendre = shetar.bases.LegendrePolynomials(order=order, x=cosine_colatitude)

    indices = np.asarray([(n, m) for n in range(order + 1) for m in range(-n, n + 1)])
    n, m = indices.T

    scipy_value = scipy.special.lpmv(m, n, cosine_colatitude[..., None])
    scipy_norm = (2 * scipy.special.factorial(n + m) / (2 * n + 1) / scipy.special.factorial(n - m))**0.5
    implemented_values = associated_legendre[n, m]
    np.testing.assert_allclose(scipy_value, implemented_values * scipy_norm, err_msg=f'Implemended AssociatedLegendrePolynomials does not match scipy after scaling at {order = }')
    np.testing.assert_allclose(implemented_values, associated_legendre[indices], err_msg=f'Indexing AssociatedLegendrePolynomials with [n, m] or [index] gives different results at {order = }')
    np.testing.assert_allclose(associated_legendre[np.arange(order + 1), 0], legendre[np.arange(order + 1)], err_msg=f'Indexing AssociatedLegendrePolynomials[n, 0] differs from LegendrePolynomials[n] at {order = }')

    for idx, (n, m) in enumerate(indices):
        np.testing.assert_allclose(implemented_values[..., idx], associated_legendre[n, m], err_msg=f'Indexing AssociatedLegendrePolynomials with vector of single value gives different reslts at {(n, m) = }')


def test_spherical_harmonics_values(order, colatitude, azimuth):
    try:
        colatitude, azimuth = np.broadcast_arrays(colatitude, azimuth)
    except ValueError:
        colatitude = colatitude.reshape(colatitude.shape + (1,) * np.ndim(azimuth))
    sh = shetar.bases.SphericalHarmonics(order=order, colatitude=colatitude, azimuth=azimuth)

    indices = np.asarray([(n, m) for n in range(order + 1) for m in range(-n, n + 1)])
    n, m = indices.T
    scipy_value = scipy.special.sph_harm(m, n, azimuth[..., None], colatitude[..., None])
    implemented_values = sh[n, m]
    np.testing.assert_allclose(
        scipy_value, implemented_values,
        err_msg=f'Implemented SphericalHarmonics differs from scipy implemented values at {order = }',
        atol=10 * np.finfo(float).eps  # Since the scipy implementation has more floating point errors internally.
    )
    np.testing.assert_allclose(implemented_values, sh[indices], err_msg=f'Indexing SphericalHarmonics with [n, m] or [index] gives different results at {order = }')
 
    for idx, (n, m) in enumerate(indices):
        np.testing.assert_allclose(implemented_values[..., idx], sh[n, m], err_msg=f'Indexing SphericalHarmonics with vector of single value gives different reslts at {(n, m) = }')


def test_regular_base_values(order, position, wavenumber):
    radius = np.sum(position**2, axis=-1) ** 0.5
    kr = radius * wavenumber
    colatitude = np.arccos(position[..., 2] / radius)
    azimuth = np.arctan2(position[..., 1], position[..., 0])

    implemented = shetar.bases.RegularBase(order=order, position=position, wavenumber=wavenumber)
    for n in range(order + 1):
        radial = scipy.special.spherical_jn(n, kr)
        for m in range(-n, n + 1):
            expected = radial * scipy.special.sph_harm(m, n, azimuth, colatitude)
            np.testing.assert_allclose(
                expected, implemented[n, m],
                err_msg=f'RegularBase does not give expected value at (n, m) = {(n, m)}'
            )


def test_singular_base_values(order, position, wavenumber):
    radius = np.sum(position**2, axis=-1) ** 0.5
    kr = radius * wavenumber
    colatitude = np.arccos(position[..., 2] / radius)
    azimuth = np.arctan2(position[..., 1], position[..., 0])

    implemented = shetar.bases.SingularBase(order=order, position=position, wavenumber=wavenumber)
    for n in range(order + 1):
        radial = scipy.special.spherical_jn(n, kr) + 1j * scipy.special.spherical_yn(n, kr)
        for m in range(-n, n + 1):
            expected = radial * scipy.special.sph_harm(m, n, azimuth, colatitude)
            np.testing.assert_allclose(
                expected, implemented[n, m],
                err_msg=f'RegularBase does not give expected value at (n, m) = {(n, m)}'
            )


def test_dual_base(order, position, wavenumber):
    regular = shetar.bases.RegularBase(order=order, position=position, wavenumber=wavenumber)
    singular = shetar.bases.SingularBase(order=order, position=position, wavenumber=wavenumber)
    dual = shetar.bases.DualBase(order=order, position=position, wavenumber=wavenumber)

    for n in range(order + 1):
        for m in range(-n, n + 1):
            np.testing.assert_allclose(regular[n, m], dual.regular[n, m])
            np.testing.assert_allclose(singular[n, m], dual.singular[n, m])
