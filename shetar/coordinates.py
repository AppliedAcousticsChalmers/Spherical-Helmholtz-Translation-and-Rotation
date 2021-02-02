"""Module to handle coordinate transforms."""

import numpy as np
import abc


class Coordinate(abc.ABC):
    @classmethod
    def parse_args(cls, coordinate):
        if isinstance(coordinate, Coordinate):
            return cls.from_coordinate(coordinate)

    @classmethod
    @abc.abstractmethod
    def from_coordinate(cls, coordinate):
        return coordinate

    def __init__(self):
        self.shapes = type(self)._ShapeClass(self)

    @property
    def shape(self):
        return self.shapes.shape

    @property
    def ndim(self):
        return self.shapes.ndim

    class _ShapeClass(abc.ABC):
        @staticmethod
        def broadcast_shapes(*shapes, output='new'):
            ndim = max(len(s) for s in shapes)
            padded_shapes = [(1,) * (ndim - len(s)) + s for s in shapes]
            out_shape = [max(s) for s in zip(*padded_shapes)]
            if not all([dim == 1 or dim == out_dim for dims, out_dim in zip(zip(*padded_shapes), out_shape) for dim in dims]):
                raise ValueError(f"Shapes {shapes} cannot be broadcast together")
            if output == 'new':
                return tuple(out_shape)
            elif output == 'reshape':
                return padded_shapes

        def __init__(self, owner):
            self.owner = owner

        @property
        @abc.abstractmethod
        def shape(self):
            pass

        def ndim(self):
            return len(self.shape)


class NonspatialCoordinate(Coordinate):
    def __init__(self, x, **kwargs):
        super().__init__(**kwargs)
        self._x = np.asarray(x)

    @classmethod
    def from_coordinate(cls, coordinate):
        if type(coordinate) is not cls:
            raise ValueError(f'Cannot cast a `{coordinate.__class__.__name__}` to {cls.__name__}')
        return coordinate

    class _ShapeClass(Coordinate._ShapeClass):
        @property
        def shape(self):
            return np.shape(self.owner._x)


class SpatialCoordinate(Coordinate):
    @classmethod
    def parse_args(cls, position=None, *,
                   x=None, y=None, z=None, cartesian_mesh=None,
                   radius=None, colatitude=None, azimuth=None, spherical_mesh=None,
                   **kwargs):
        obj = super().parse_args(position)
        if obj is not None:
            return obj
        if (x is not None) or (y is not None) or (z is not None):
            return Cartesian(x=x, y=y, z=z, **kwargs)
        if (radius is not None) or (colatitude is not None) or (azimuth is not None):
            return Spherical(radius=radius, colatitude=colatitude, azimuth=azimuth, **kwargs)
        if position is not None:
            return CartesianMesh(mesh=position, **kwargs)
        if cartesian_mesh is not None:
            return CartesianMesh(mesh=cartesian_mesh, **kwargs)
        if spherical_mesh is not None:
            return SphericalMesh(mesh=spherical_mesh, **kwargs)

    def __init__(self, *args, automesh=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.automesh = automesh

    @property
    @abc.abstractmethod
    def x(self):
        pass

    @property
    @abc.abstractmethod
    def y(self):
        pass

    @property
    @abc.abstractmethod
    def z(self):
        pass

    @property
    def cartesian_mesh(self):
        return np.stack(self.xyz, axis=-1)

    @property
    def xyz(self):
        if self.automesh:
            return np.meshgrid(self.x, self.y, self.z, indexing='ij')
        else:
            return self.x, self.y, self.z

    @property
    @abc.abstractmethod
    def radius(self):
        pass

    @property
    @abc.abstractmethod
    def colatitude(self):
        pass

    @property
    @abc.abstractmethod
    def azimuth(self):
        pass

    @property
    def radius_colatitude_azimuth(self):
        if self.automesh:
            return np.meshgrid(self.radius, self.colatitude, self.azimuth, indexing='ij')
        else:
            return self.radius, self.colatitude, self.azimuth

    @property
    def spherical_mesh(self):
        return np.stack(self.radius_colatitude_azimuth, axis=-1)

    class _ShapeClass(Coordinate._ShapeClass):
        @property
        @abc.abstractmethod
        def x(self):
            pass

        @property
        @abc.abstractmethod
        def y(self):
            pass

        @property
        @abc.abstractmethod
        def z(self):
            pass

        @property
        def cartesian_mesh(self):
            return self.broadcast_shapes(self.x, self.y, self.z)

        @property
        @abc.abstractmethod
        def radius(self):
            pass

        @property
        @abc.abstractmethod
        def colatitude(self):
            pass

        @property
        @abc.abstractmethod
        def azimuth(self):
            pass

        @property
        def spherical_mesh(self):
            return self.broadcast_shapes(self.radius, self.colatitude, self.azimuth)


def cartesian_2_spherical(cartesian_positions):
    """Convert a cartesian position to the spherical variables.

    Parameters
    ----------
    cartesian_positions: array_like
        A 3x... shape array with the cartesian positions [x, y, z] in the
        first axis.

    Returns
    -------
    spherical_coordinates : np.ndarray
        A 3x... shape array with the first axis corresponding to the radius,
        the colatitude angle, and the azimuth angle.
    """
    cartesian_positions = np.asarray(cartesian_positions)
    r = np.sum(cartesian_positions**2, axis=0)**0.5
    with np.errstate(divide='ignore', invalid='ignore'):
        theta = np.where(r == 0, 0, np.arccos(np.clip(cartesian_positions[2] / r, -1., 1.)))
    phi = np.arctan2(cartesian_positions[1], cartesian_positions[0])
    return np.stack([r, theta, phi], axis=0)


def cartesian_2_trigonometric(cartesian_positions):
    r"""Convert a cartesian position to the spherical variables.

    This defines the spherical variable transform

    .. math::
        x = r \sin\theta\cos\phi
        y = r \sin\theta\sin\phi
        z = r \cos\theta

    but instead of calculating the angles, values of the trigonometric functions
    are returned. In many applications is is the trigonometric values that are of
    interest, and it is more efficient to compute them directly.

    Parameters
    ----------
    cartesian_positions: array_like
        A 3x... shape array with the cartesian positions [x, y, z] in the
        first axis.

    Returns
    -------
    spherical_coordinates : np.ndarray
        A 5x... shape array with the first axis corresponding to the radius,
        :math:`\cos\theta`, :math:`\sin\theta`, :math:`\cos\phi`, and :math:`\sin\phi`
    """
    cartesian_positions = np.asarray(cartesian_positions)
    r = np.sum(cartesian_positions**2, axis=0)**0.5
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = np.where(r == 0, 0, cartesian_positions / r)
    cos_theta = np.clip(normalized[2], -1, 1)
    sin_theta = (1 - cos_theta**2)**0.5
    with np.errstate(divide='ignore', invalid='ignore'):
        xy_projected = np.where(sin_theta * r == 0, np.reshape([1, 0], [2] + [1] * np.ndim(sin_theta)), normalized[:2] / sin_theta)
    cos_phi = np.clip(xy_projected[0], -1, 1)
    sin_phi = np.clip(xy_projected[1], -1, 1)
    return np.stack([r, cos_theta, sin_theta, cos_phi, sin_phi], axis=0)


def spherical_2_cartesian(radius, colatitude, azimuth):
    """Convert spherical coordinates to cartesian positions.

    Parameters
    ----------
    spherical_coordinates : array_like
        A 3x... shape array with the first axis corresponding to the radius,
        the colatitude angle, and the azimuth angle.

    Returns
    -------
    cartesian_positions: np.ndarray
        A 3x... shape array with the cartesian positions [x, y, z] in the
        first axis.
    """
    radius, colatitude, azimuth = np.broadcast_arrays(radius, colatitude, azimuth)
    x = np.sin(colatitude) * np.cos(azimuth)
    y = np.sin(colatitude) * np.sin(azimuth)
    z = np.cos(colatitude)
    return radius * np.stack([x, y, z], axis=0)


def rotation_matrix(colatitude, azimuth, secondary_azimuth):
    return np.array([
        [
            np.cos(colatitude) * np.cos(secondary_azimuth) * np.cos(azimuth) - np.sin(secondary_azimuth) * np.sin(azimuth),
            np.cos(colatitude) * np.cos(secondary_azimuth) * np.sin(azimuth) + np.sin(secondary_azimuth) * np.cos(azimuth),
            -np.sin(colatitude) * np.cos(secondary_azimuth),
        ],
        [
            -np.cos(colatitude) * np.sin(secondary_azimuth) * np.cos(azimuth) - np.cos(secondary_azimuth) * np.sin(azimuth),
            np.cos(secondary_azimuth) * np.cos(azimuth) - np.cos(colatitude) * np.sin(secondary_azimuth) * np.sin(azimuth),
            np.sin(colatitude) * np.sin(secondary_azimuth),
        ],
        [
            np.sin(colatitude) * np.cos(azimuth),
            np.sin(colatitude) * np.sin(azimuth),
            np.cos(colatitude)
        ],
    ])


def z_axes_rotation_angles(new_axis=None, old_axis=None):
    if new_axis is None and old_axis is None:
        raise ValueError('Must define at least the new z-axis or the old z-axis')

    if new_axis is None:
        _, colatitude, azimuth_old = cartesian_2_spherical(old_axis)
        azimuth_new = 0
    elif old_axis is None:
        _, colatitude, azimuth_new = cartesian_2_spherical(new_axis)
        azimuth_old = np.pi
    else:
        _, colatitude, azimuth_new = cartesian_2_spherical(new_axis)
        _, colatitude_old, azimuth_old = cartesian_2_spherical(old_axis)
        if not np.allclose(colatitude, colatitude_old):
            raise ValueError('New z-axis and old z-axis does not have the same angle between them, check the inputs!')
    return colatitude, azimuth_new, np.pi - azimuth_old


def z_axes_rotation_matrix(new_axis=None, old_axis=None):
    beta, alpha, mu = z_axes_rotation_angles(new_axis=new_axis, old_axis=old_axis)
    return rotation_matrix(beta, alpha, mu)
