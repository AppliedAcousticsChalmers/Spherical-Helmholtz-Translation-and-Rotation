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
def test_invertability(position):
    spherical_pos = faheltzmm.coordinates.cartesian_2_spherical(position)
    np.testing.assert_allclose(position, faheltzmm.coordinates.spherical_2_cartesian(*spherical_pos), rtol=1e-6, atol=1e-12)
