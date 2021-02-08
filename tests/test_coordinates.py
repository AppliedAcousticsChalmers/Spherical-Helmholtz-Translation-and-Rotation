import pytest
import shetar.coordinates
import numpy as np

test_potisions = [
    [0, 0, 0],
    [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5],
    [-0.5, 0, 0], [0, -0.5, 0], [0, 0, -0.5],
    np.random.normal(size=(5, 3)), np.random.normal(size=(5, 7, 3)),
]
twice_random = (v for _ in iter(int, 1) for v in [np.random.normal()] * 2)


@pytest.fixture(scope='module', params=[
    [0, 0, 0],
    [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5],
    [-0.5, 0, 0], [0, -0.5, 0], [0, 0, -0.5],
    np.random.normal(size=(5, 3)), np.random.normal(size=(5, 7, 3)),
])
def cartesian_mesh(request):
    return np.asarray(request.param)


@pytest.fixture(scope='module')
def x(cartesian_mesh):
    return cartesian_mesh[..., 0]


@pytest.fixture(scope='module')
def y(cartesian_mesh):
    return cartesian_mesh[..., 1]


@pytest.fixture(scope='module')
def z(cartesian_mesh):
    return cartesian_mesh[..., 2]


@pytest.fixture(scope='module')
def radius(x, y, z):
    return (x**2 + y**2 + z**2)**0.5


@pytest.fixture(scope='module')
def colatitude(z, radius):
    clipped = np.clip(z / radius, -1, 1)
    replaced = np.nan_to_num(clipped, nan=1, neginf=1, posinf=1)
    return np.arccos(replaced)


@pytest.fixture(scope='module')
def azimuth(x, y):
    return np.arctan2(y, x)


@pytest.fixture(scope='module')
def spherical_mesh(radius, colatitude, azimuth):
    return np.stack([radius, colatitude, azimuth], axis=-1)


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
    rotation = shetar.coordinates.Rotation.parse_args(colatitude=colatitude, azimuth=azimuth, secondary_azimuth=secondary_azimuth)
    calculated = rotation.apply(original).cartesian_mesh
    np.testing.assert_allclose(calculated, target, atol=1e-15)


def compare_coordinate(coord, cartesian_mesh, x, y, z, radius, colatitude, azimuth, spherical_mesh):
    np.testing.assert_allclose(cartesian_mesh, coord.cartesian_mesh, atol=1e-16)
    np.testing.assert_allclose(x, coord.x, atol=1e-16)
    np.testing.assert_allclose(y, coord.y, atol=1e-16)
    np.testing.assert_allclose(z, coord.z, atol=1e-16)
    np.testing.assert_allclose(radius, coord.radius, atol=1e-16)
    np.testing.assert_allclose(colatitude, coord.colatitude, atol=1e-16)
    np.testing.assert_allclose(azimuth, coord.azimuth, atol=1e-16)
    np.testing.assert_allclose(spherical_mesh, coord.spherical_mesh, atol=1e-16)


def test_CartesianMesh(cartesian_mesh, x, y, z, radius, colatitude, azimuth, spherical_mesh):
    coord = shetar.coordinates.SpatialCoordinate.parse_args(cartesian_mesh=cartesian_mesh)
    assert type(coord) is shetar.coordinates.CartesianMesh
    compare_coordinate(coord, cartesian_mesh, x, y, z, radius, colatitude, azimuth, spherical_mesh)


def test_Cartesian(cartesian_mesh, x, y, z, radius, colatitude, azimuth, spherical_mesh):
    coord = shetar.coordinates.SpatialCoordinate.parse_args(x=x, y=y, z=z)
    assert type(coord) is shetar.coordinates.Cartesian
    compare_coordinate(coord, cartesian_mesh, x, y, z, radius, colatitude, azimuth, spherical_mesh)


def test_Spherical(cartesian_mesh, x, y, z, radius, colatitude, azimuth, spherical_mesh):
    coord = shetar.coordinates.SpatialCoordinate.parse_args(radius=radius, colatitude=colatitude, azimuth=azimuth)
    assert type(coord) is shetar.coordinates.Spherical
    compare_coordinate(coord, cartesian_mesh, x, y, z, radius, colatitude, azimuth, spherical_mesh)


def test_SphericalMesh(cartesian_mesh, x, y, z, radius, colatitude, azimuth, spherical_mesh):
    coord = shetar.coordinates.SpatialCoordinate.parse_args(spherical_mesh=spherical_mesh)
    assert type(coord) is shetar.coordinates.SphericalMesh
    compare_coordinate(coord, cartesian_mesh, x, y, z, radius, colatitude, azimuth, spherical_mesh)