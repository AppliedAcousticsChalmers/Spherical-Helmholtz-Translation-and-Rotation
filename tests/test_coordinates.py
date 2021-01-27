import pytest
import shetar.coordinates
import numpy as np

test_potisions = [
    [0, 0, 0],
    [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5],
    [-0.5, 0, 0], [0, -0.5, 0], [0, 0, -0.5],
    np.random.normal(size=(3, 5)), np.random.normal(size=(3, 5, 7)),
]
twice_random = (v for _ in iter(int, 1) for v in [np.random.normal()] * 2)


@pytest.mark.parametrize("position", test_potisions)
def test_spherical_invertability(position):
    spherical_pos = shetar.coordinates.cartesian_2_spherical(position)
    np.testing.assert_allclose(position, shetar.coordinates.spherical_2_cartesian(*spherical_pos), rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("position", test_potisions)
def test_trigonometric_invertability(position):
    r, cos_theta, sin_theta, cos_phi, sin_phi = shetar.coordinates.cartesian_2_trigonometric(position)
    x = r * sin_theta * cos_phi
    y = r * sin_theta * sin_phi
    z = r * cos_theta
    np.testing.assert_allclose(position, np.stack([x, y, z], axis=0), rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("original, target, colatitude, azimuth, secondary_azimuth", [
    # Rotations of positive x-axis
    ([1, 0, 0], [1, 0, 0], 0, -next(twice_random), next(twice_random)), ([1, 0, 0], [1, 0, 0], 0, 2 * np.pi - next(twice_random), next(twice_random)),
    ([1, 0, 0], [-1, 0, 0], 0, np.pi - next(twice_random), next(twice_random)), ([1, 0, 0], [-1, 0, 0], 0, next(twice_random), np.pi - next(twice_random)),
    ([1, 0, 0], [0, 1, 0], 0, np.pi / 2 - next(twice_random), next(twice_random)), ([1, 0, 0], [0, 1, 0], 0, next(twice_random), np.pi / 2 - next(twice_random)),
    ([1, 0, 0], [0, -1, 0], 0, -np.pi / 2 - next(twice_random), next(twice_random)), ([1, 0, 0], [0, -1, 0], 0, next(twice_random), -np.pi / 2 - next(twice_random)),
    ([1, 0, 0], [0, 0, 1], np.pi / 2, np.random.normal(), np.pi), ([1, 0, 0], [0, 0, -1], np.pi / 2, np.random.normal(), 0),
    # Rotations of negative x-axis
    ([-1, 0, 0], [1, 0, 0], 0, np.pi - next(twice_random), next(twice_random)), ([-1, 0, 0], [1, 0, 0], 0, next(twice_random), -np.pi - next(twice_random)),
    ([-1, 0, 0], [-1, 0, 0], 0, next(twice_random), -next(twice_random)), ([-1, 0, 0], [-1, 0, 0], 0, next(twice_random), 2 * np.pi - next(twice_random)),
    ([-1, 0, 0], [0, 1, 0], 0, -np.pi / 2 - next(twice_random), next(twice_random)), ([-1, 0, 0], [0, 1, 0], 0, next(twice_random), -np.pi / 2 - next(twice_random)),
    ([-1, 0, 0], [0, -1, 0], 0, np.pi / 2 - next(twice_random), next(twice_random)), ([-1, 0, 0], [0, -1, 0], 0, next(twice_random), np.pi / 2 - next(twice_random)),
    ([-1, 0, 0], [0, 0, 1], np.pi / 2, np.random.normal(), 0), ([-1, 0, 0], [0, 0, -1], np.pi / 2, np.random.normal(), np.pi),
    # Rotations of positive y-axis
    ([0, 1, 0], [1, 0, 0], 0, -np.pi / 2 - next(twice_random), next(twice_random)), ([0, 1, 0], [1, 0, 0], 0, next(twice_random), -np.pi / 2 - next(twice_random)),
    ([0, 1, 0], [-1, 0, 0], 0, np.pi / 2 - next(twice_random), next(twice_random)), ([0, 1, 0], [-1, 0, 0], 0, next(twice_random), np.pi / 2 - next(twice_random)),
    ([0, 1, 0], [0, 1, 0], 0, next(twice_random), -next(twice_random)), ([0, 1, 0], [0, 1, 0], 0, 2 * np.pi - next(twice_random), next(twice_random)),
    ([0, 1, 0], [0, -1, 0], 0, np.pi / 2 - next(twice_random), np.pi / 2 + next(twice_random)), ([0, 1, 0], [0, -1, 0], 0, next(twice_random), np.pi - next(twice_random)),
    ([0, 1, 0], [0, 0, 1], np.pi / 2, np.random.normal(), np.pi / 2), ([0, 1, 0], [0, 0, -1], np.pi / 2, np.random.normal(), -np.pi / 2),
    # Rotations of negative y-axis
    ([0, -1, 0], [1, 0, 0], 0, np.pi / 2 - next(twice_random), next(twice_random)), ([0, -1, 0], [1, 0, 0], 0, next(twice_random), np.pi / 2 - next(twice_random)),
    ([0, -1, 0], [-1, 0, 0], 0, - np.pi / 2 - next(twice_random), next(twice_random)), ([0, -1, 0], [-1, 0, 0], 0, next(twice_random), - np.pi / 2 - next(twice_random)),
    ([0, -1, 0], [0, 1, 0], 0, np.pi + next(twice_random), -next(twice_random)), ([0, -1, 0], [0, 1, 0], 0, next(twice_random), np.pi - next(twice_random)),
    ([0, -1, 0], [0, -1, 0], 0, next(twice_random), -next(twice_random)), ([0, -1, 0], [0, -1, 0], 0, -np.pi + next(twice_random), np.pi - next(twice_random)),
    ([0, -1, 0], [0, 0, 1], np.pi / 2, np.random.normal(), -np.pi / 2), ([0, -1, 0], [0, 0, -1], np.pi / 2, np.random.normal(), np.pi / 2),
    # Rotations of positive z-axis
    ([0, 0, 1], [1, 0, 0], np.pi / 2, 0, np.random.normal()), ([0, 0, 1], [1, 0, 0], np.pi / 2, 2 * np. pi, np.random.normal()),
    ([0, 0, 1], [-1, 0, 0], np.pi / 2, np.pi, np.random.normal()), ([0, 0, 1], [-1, 0, 0], np.pi / 2, -np.pi, np.random.normal()),
    ([0, 0, 1], [0, 1, 0], np.pi / 2, np.pi / 2, np.random.normal()), ([0, 0, 1], [0, 1, 0], np.pi / 2, -1.5 * np.pi, np.random.normal()),
    ([0, 0, 1], [0, -1, 0], np.pi / 2, -np.pi / 2, np.random.normal()), ([0, 0, 1], [0, -1, 0], np.pi / 2, 1.5 * np.pi, np.random.normal()),
    ([0, 0, 1], [0, 0, 1], 0, np.random.normal(), np.random.normal()), ([0, 0, 1], [0, 0, -1], np.pi, np.random.normal(), np.random.normal()),
    # Rotations of negative z-axis
    ([0, 0, -1], [1, 0, 0], np.pi / 2, np.pi, np.random.normal()), ([0, 0, -1], [1, 0, 0], np.pi / 2, -np.pi, np.random.normal()),
    ([0, 0, -1], [-1, 0, 0], np.pi / 2, 0, np.random.normal()), ([0, 0, -1], [-1, 0, 0], np.pi / 2, 2 * np.pi, np.random.normal()),
    ([0, 0, -1], [0, 1, 0], np.pi / 2, -np.pi / 2, np.random.normal()), ([0, 0, -1], [0, 1, 0], np.pi / 2, 1.5 * np.pi, np.random.normal()),
    ([0, 0, -1], [0, -1, 0], np.pi / 2, np.pi / 2, np.random.normal()), ([0, 0, -1], [0, -1, 0], np.pi / 2, -1.5 * np.pi, np.random.normal()),
    ([0, 0, -1], [0, 0, 1], np.pi, np.random.normal(), np.random.normal()), ([0, 0, -1], [0, 0, -1], 0, np.random.normal(), np.random.normal()),
])
def test_axis_rotations(original, target, colatitude, azimuth, secondary_azimuth):
    rotation_matrix = shetar.coordinates.rotation_matrix(colatitude=colatitude, azimuth=azimuth, secondary_azimuth=secondary_azimuth)
    np.testing.assert_allclose(rotation_matrix.T @ original, target, atol=1e-15)


@pytest.mark.parametrize('beta', np.random.uniform(0, np.pi, 3))
@pytest.mark.parametrize('alpha', np.random.uniform(0, 2 * np.pi, 3))
@pytest.mark.parametrize('mu', np.random.uniform(0, 2 * np.pi, 3))
def test_z_axis_rotations(beta, alpha, mu):
    Q = shetar.coordinates.rotation_matrix(beta, alpha, mu)
    old_axis = Q[:, 2]  # The coordinates of the old axis expressed in the new system.
    new_axis = Q[2]  # The coordinates of the new axis expressed in the old system

    Q_from_axes = shetar.coordinates.z_axes_rotation_matrix(new_axis=new_axis, old_axis=old_axis)
    np.testing.assert_allclose(Q, Q_from_axes)
    beta, alpha, mu = shetar.coordinates.z_axes_rotation_angles(new_axis=new_axis, old_axis=old_axis)
    beta_inv, alpha_inv, mu_inv = shetar.coordinates.z_axes_rotation_angles(new_axis=old_axis, old_axis=new_axis)
    np.testing.assert_allclose(beta, beta_inv)
    np.testing.assert_allclose(alpha_inv, np.pi - mu)
    np.testing.assert_allclose(mu_inv, np.pi - alpha)
