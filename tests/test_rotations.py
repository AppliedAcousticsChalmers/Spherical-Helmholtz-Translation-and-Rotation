import numpy as np
import shetar.rotations
import shetar.coordinates
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


def test_inverse_rotations(old_coefficients, new_coefficients, inverse_rotation_coefficients, rotation_coefficients):
    recalc_old_indirect = inverse_rotation_coefficients.apply(new_coefficients, inverse=True)
    np.testing.assert_allclose(recalc_old_indirect._data, old_coefficients._data)
    recalc_old_direct = rotation_coefficients.apply(new_coefficients)
    np.testing.assert_allclose(recalc_old_direct._data, old_coefficients._data)


def test_angle_specification_rotation_coefficients(rotation_coefficients, axis_angles, order):
    beta, alpha, gamma = axis_angles
    manual_coefficients = shetar.rotations.Rotation(order=order, colatitude=beta, azimuth=alpha, secondary_azimuth=np.pi - gamma)
    np.testing.assert_allclose(manual_coefficients._data, rotation_coefficients._data)


def test_inverse_rotation_coefficients(inverse_rotation_coefficients, axis_angles, order):
    beta, alpha, gamma = axis_angles
    manual_inverse_coefficients = shetar.rotations.Rotation(order=order, colatitude=beta, azimuth=gamma, secondary_azimuth=np.pi - alpha)
    np.testing.assert_allclose(manual_inverse_coefficients._data, inverse_rotation_coefficients._data)


# ===================== Geometry and positions  =====================
@pytest.fixture(scope="module")
def new_z(axis_angles):
    beta, alpha, gamma = axis_angles
    return shetar.coordinates.spherical_2_cartesian(1, beta, alpha)


@pytest.fixture(scope="module")
def old_z(axis_angles):
    beta, alpha, gamma = axis_angles
    return shetar.coordinates.spherical_2_cartesian(1, beta, gamma)


@pytest.fixture(scope='module')
def new_positions(new_z, old_z, old_positions):
    return np.einsum('ij, j...-> i...', shetar.coordinates.z_axes_rotation_matrix(new_axis=new_z, old_axis=old_z), old_positions)


# ===================== Expansion coefficients  =====================
@pytest.fixture(scope='module')
def old_coefficients(order):
    coeffs = shetar.expansions.Expansion(order=order)
    coeffs._data = np.random.normal(size=len(coeffs._data)) + 1j * np.random.normal(size=len(coeffs._data))
    return coeffs


@pytest.fixture(scope='module')
def rotation_coefficients(order, new_z, old_z):
    return shetar.rotations.Rotation(order=order, new_z_axis=new_z, old_z_axis=old_z)


@pytest.fixture(scope='module')
def inverse_rotation_coefficients(order, new_z, old_z):
    return shetar.rotations.Rotation(order=order, new_z_axis=old_z, old_z_axis=new_z)


@pytest.fixture(scope='module')
def new_coefficients(old_coefficients, rotation_coefficients):
    return rotation_coefficients.apply(old_coefficients, inverse=True)


@pytest.fixture(scope='module')
def old_bases(order, old_positions):
    _, colatitude, azimuth = shetar.coordinates.cartesian_2_spherical(old_positions)
    return shetar.bases.SphericalHarmonics(order=order, colatitude=colatitude, azimuth=azimuth)


@pytest.fixture(scope='module')
def new_bases(order, new_positions):
    _, colatitude, azimuth = shetar.coordinates.cartesian_2_spherical(new_positions)
    return shetar.bases.SphericalHarmonics(order=order, colatitude=colatitude, azimuth=azimuth)


@pytest.fixture(scope='module')
def old_values(old_coefficients, old_bases):
    return old_bases.apply(old_coefficients)


@pytest.fixture(scope='module')
def new_values(new_coefficients, new_bases):
    return new_bases.apply(new_coefficients)
