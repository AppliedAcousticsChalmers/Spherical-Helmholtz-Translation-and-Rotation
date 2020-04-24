import pytest
import faheltzmm.coordinates
import numpy as np

np.random.seed(0)
test_potisions = [
    [0, 0, 0],
    [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5],
    [-0.5, 0, 0], [0, -0.5, 0], [0, 0, -0.5],
    np.random.normal(size=(3, 5)), np.random.normal(size=(3, 5, 7)),
]


@pytest.mark.parametrize("position", test_potisions)
def test_spherical_invertability(position):
    spherical_pos = faheltzmm.coordinates.cartesian_2_spherical(position)
    np.testing.assert_allclose(position, faheltzmm.coordinates.spherical_2_cartesian(*spherical_pos), rtol=1e-6, atol=1e-12)

@pytest.mark.parametrize("position", test_potisions)
def test_trigonometric_invertability(position):
    r, cos_theta, sin_theta, cos_phi, sin_phi = faheltzmm.coordinates.cartesian_2_trigonometric(position)
    x = r * sin_theta * cos_phi
    y = r * sin_theta * sin_phi
    z = r * cos_theta
    np.testing.assert_allclose(position, np.stack([x, y, z], axis=0), rtol=1e-6, atol=1e-12)