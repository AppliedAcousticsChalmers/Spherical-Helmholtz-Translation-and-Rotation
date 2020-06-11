import numpy as np
import faheltzmm.rotations
import faheltzmm.generate
import faheltzmm.coordinates
import faheltzmm.indexing
import pytest


# ===================== Test input parameters =====================
@pytest.fixture(params=[0, 1, 2, 3, 4, pytest.param(15, marks=pytest.mark.slow), pytest.param(40, marks=pytest.mark.slow)], scope='module')
def order(request):
    return request.param


@pytest.fixture(params=[1 for x in range(2)], scope='module')
def axis_angles(request):
    scale = request.param
    return (
        scale * np.random.uniform(0, np.pi),
        scale * np.random.uniform(0, 2 * np.pi),
        scale * np.random.uniform(0, 2 * np.pi)
    )


@pytest.fixture(params=[tuple(), (5,), (5, 7)], scope='module')
def old_positions(request):
    size = request.param
    return np.random.normal(size=(3,) + size)


# ===================== Actual tests ============================
def test_value_equality(old_values, new_values):
    np.testing.assert_allclose(new_values, old_values)


def test_inverse_rotations(old_coefficients, new_coefficients, inverse_rotation_coefficients):
    np.testing.assert_allclose(old_coefficients, np.einsum('mnp, nm -> np', inverse_rotation_coefficients, new_coefficients))


def test_angle_specification_rotation_coefficients(rotation_coefficients, axis_angles, order):
    beta, alpha, gamma = axis_angles
    manual_coefficients = faheltzmm.rotations.rotation_coefficients(max_order=order, colatitude=beta, primary_azimuth=alpha, secondary_azimuth=np.pi - gamma)
    np.testing.assert_allclose(manual_coefficients, rotation_coefficients)


def test_inverse_rotation_coefficients(inverse_rotation_coefficients, axis_angles, order):
    beta, alpha, gamma = axis_angles
    manual_inverse_coefficients = faheltzmm.rotations.rotation_coefficients(max_order=order, colatitude=beta, primary_azimuth=gamma, secondary_azimuth=np.pi - alpha)
    np.testing.assert_allclose(manual_inverse_coefficients, inverse_rotation_coefficients)


def test_rotate_func(old_coefficients, new_z, old_z, new_coefficients, rotation_coefficients):
    np.testing.assert_allclose(new_coefficients, faheltzmm.rotations.rotate(old_coefficients, new_z_axis=new_z, old_z_axis=old_z))
    np.testing.assert_allclose(new_coefficients, faheltzmm.rotations.rotate(old_coefficients, rotation_coefficients=rotation_coefficients))
    np.testing.assert_allclose(old_coefficients, faheltzmm.rotations.rotate(new_coefficients, rotation_coefficients=rotation_coefficients, inverse=True))


# ===================== Geometry and positions  =====================
@pytest.fixture(scope="module")
def new_z(axis_angles):
    beta, alpha, gamma = axis_angles
    return faheltzmm.coordinates.spherical_2_cartesian(1, beta, alpha)


@pytest.fixture(scope="module")
def old_z(axis_angles):
    beta, alpha, gamma = axis_angles
    return faheltzmm.coordinates.spherical_2_cartesian(1, beta, gamma)


@pytest.fixture(scope='module')
def new_positions(new_z, old_z, old_positions):
    return np.einsum('ij, j...-> i...', faheltzmm.coordinates.z_axes_rotation_matrix(new_axis=new_z, old_axis=old_z), old_positions)


# ===================== Expansion coefficients  =====================
@pytest.fixture(scope='module')
def old_coefficients(order):
    n_coeffs = (order + 1)**2
    linear_coeffs = np.random.normal(size=n_coeffs) + 1j * np.random.normal(size=n_coeffs)
    return faheltzmm.indexing.expansions(linear_coeffs, 'linear', 'natural')


@pytest.fixture(scope='module')
def rotation_coefficients(order, new_z, old_z):
    return faheltzmm.rotations.rotation_coefficients(max_order=order, new_z_axis=new_z, old_z_axis=old_z)


@pytest.fixture(scope='module')
def inverse_rotation_coefficients(order, new_z, old_z):
    return faheltzmm.rotations.rotation_coefficients(max_order=order, new_z_axis=old_z, old_z_axis=new_z)


@pytest.fixture(scope='module')
def new_coefficients(old_coefficients, rotation_coefficients):
    return np.einsum('mnp, nm...->np...', rotation_coefficients, old_coefficients)


@pytest.fixture(scope='module')
def old_bases(order, old_positions):
    _, colatitude, azimuth = faheltzmm.coordinates.cartesian_2_spherical(old_positions)
    return faheltzmm.generate.spherical_harmonics_all(max_order=order, colatitude=colatitude, azimuth=azimuth)


@pytest.fixture(scope='module')
def new_bases(order, new_positions):
    _, colatitude, azimuth = faheltzmm.coordinates.cartesian_2_spherical(new_positions)
    return faheltzmm.generate.spherical_harmonics_all(max_order=order, colatitude=colatitude, azimuth=azimuth)


@pytest.fixture(scope='module')
def old_values(old_coefficients, old_bases):
    return np.einsum('nm, nm...', old_coefficients, old_bases)


@pytest.fixture(scope='module')
def new_values(new_coefficients, new_bases):
    return np.einsum('nm, nm...', new_coefficients, new_bases)
