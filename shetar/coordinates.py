"""Module to handle coordinate transforms."""

import numpy as np
import abc


class OwnerMixin:
    @property
    def shape(self):
        return self.coordinate.shape

    @property
    def ndim(self):
        return len(self.shape)

    def copy(self, deep=False):
        new_obj = type(self).__new__(type(self))
        new_obj.coordinate = self.coordinate.copy(deep=deep)
        return new_obj


class Coordinate(abc.ABC):
    @classmethod
    def parse_args(cls, coordinate):
        if isinstance(coordinate, Coordinate):
            return cls.from_coordinate(coordinate)

    @classmethod
    def from_coordinate(cls, coordinate):
        if not isinstance(coordinate, Coordinate):
            raise ValueError(f'Cannot cast a {coordinate.__class__.__name__} to {cls.__name__}')
        return coordinate

    def __init__(self):
        self.shapes = type(self)._ShapeClass(self)

    @abc.abstractmethod
    def copy(self, deep=False):
        pass

    @property
    def shape(self):
        return self.shapes.shape

    @property
    def ndim(self):
        return self.shapes.ndim

    def apply(self, transform, *args, **kwargs):
        return transform.apply(self, *args, **kwargs)

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

    @property
    def x(self):
        return self._x

    def copy(self, deep=False):
        if deep:
            return type(self)(x=self.x.copy())
        else:
            return type(self)(x=self.x)

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
        xyz = np.meshgrid(self.x, self.y, self.z, indexing='ij', sparse=False)
        return np.stack(xyz, axis=-1)

    @property
    def xyz(self):
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
        return self.radius, self.colatitude, self.azimuth

    def rotate(self, *args, **kwargs):
        return Rotation.parse_args(*args, **kwargs).apply(self)

    def translate(self, *args, **kwargs):
        return Translation.parse_args(*args, **kwargs).apply(self)

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


class _CartesianConverter(SpatialCoordinate):
    @property
    def radius(self):
        return (self.x**2 + self.y**2 + self.z**2)**0.5

    @property
    def colatitude(self):
        with np.errstate(invalid='ignore'):
            r = self.radius
            return np.arccos(np.where(r == 0, 0, self.z / r))

    @property
    def azimuth(self):
        return np.arctan2(self.y, self.x)

    @property
    def spherical_mesh(self):
        return np.stack(np.broadcast_arrays(self.radius, self.colatitude, self.azimuth), axis=-1).squeeze()

    class _ShapeClass(SpatialCoordinate._ShapeClass):
        @property
        def radius(self):
            return self.broadcast_shapes(self.x, self.y, self.z)

        @property
        def colatitude(self):
            return self.broadcast_shapes(self.radius, self.z)

        @property
        def azimuth(self):
            return self.broadcast_shapes(self.x, self.y)


class _SphericalConverter(SpatialCoordinate):
    @property
    def x(self):
        return self.radius * np.sin(self.colatitude) * np.cos(self.azimuth)

    @property
    def y(self):
        return self.radius * np.sin(self.colatitude) * np.sin(self.azimuth)

    @property
    def z(self):
        return self.radius * np.cos(self.colatitude)

    @property
    def cartesian_mesh(self):
        return np.stack(np.broadcast_arrays(self.x, self.y, self.z), axis=-1).squeeze()

    class _ShapeClass(SpatialCoordinate._ShapeClass):
        @property
        def x(self):
            return self.broadcast_shapes(self.radius, self.colatitude, self.azimuth)

        @property
        def y(self):
            return self.broadcast_shapes(self.radius, self.colatitude, self.azimuth)

        @property
        def z(self):
            return self.broadcast_shapes(self.radius, self.colatitude)


class Cartesian(_CartesianConverter):

    @classmethod
    def from_coordinate(cls, coordinate):
        return cls(x=coordinate.x, y=coordinate.y, z=coordinate.z)

    def __init__(self, x=None, y=None, z=None, **kwargs):
        super().__init__(**kwargs)
        self._x = np.asarray(x if x is not None else 0)
        self._y = np.asarray(y if y is not None else 0)
        self._z = np.asarray(z if z is not None else 0)

    def copy(self, deep=False):
        if deep:
            return type(self)(x=self._x.copy(), y=self._y.copy(), z=self._z.copy(), automesh=self.automesh)
        else:
            return type(self)(x=self._x, y=self._y, z=self._z, automesh=self.automesh)

    @property
    def x(self):
        return np.reshape(self._x, self.shapes.x)

    @property
    def y(self):
        return np.reshape(self._y, self.shapes.y)

    @property
    def z(self):
        return np.reshape(self._z, self.shapes.z)

    @property
    def cartesian_mesh(self):
        return np.stack(np.meshgrid(self.x, self.y, self.z, indexing='ij', sparse=False), axis=-1).squeeze()

    class _ShapeClass(_CartesianConverter._ShapeClass):
        @property
        def x(self):
            if self.owner.automesh:
                return np.shape(self.owner._x) + (1,) * (np.ndim(self.owner._y) + np.ndim(self.owner._z))
            else:
                return np.shape(self.owner._x)

        @property
        def y(self):
            if self.owner.automesh:
                return (1,) * np.ndim(self.owner._y) + np.shape(self.owner._y) + (1,) * np.ndim(self.owner._z)
            else:
                return np.shape(self.owner._y)

        @property
        def z(self):
            if self.owner.automesh:
                return (1,) * (np.ndim(self.owner._x) + np.ndim(self.owner._y)) + np.shape(self.owner._z)
            else:
                return np.shape(self.owner._z)

        @property
        def shape(self):
            return self.broadcast_shapes(self.x, self.y, self.z)


class CartesianMesh(_CartesianConverter):
    @classmethod
    def from_coordinate(cls, coordinate):
        return cls(mesh=coordinate.cartesian_mesh)

    def __init__(self, mesh, **kwargs):
        kwargs['automesh'] = False  # Should not apply meshing to a mesh again.
        super().__init__(**kwargs)
        self._cartesian_mesh = np.asarray(mesh)

    def copy(self, deep=False):
        if deep:
            return type(self)(mesh=self._cartesian_mesh.copy(), automesh=self.automesh)
        else:
            return type(self)(mesh=self._cartesian_mesh, automesh=self.automesh)

    @property
    def x(self):
        return self._cartesian_mesh[..., 0]

    @property
    def y(self):
        return self._cartesian_mesh[..., 1]

    @property
    def z(self):
        return self._cartesian_mesh[..., 2]

    @property
    def cartesian_mesh(self):
        return self._cartesian_mesh

    class _ShapeClass(_CartesianConverter._ShapeClass):
        def shape(self):
            return np.shape(self.owner._cartesian_mesh)[:-1]
        x = property(shape)
        y = property(shape)
        z = property(shape)
        shape = property(shape)


class Spherical(_SphericalConverter):
    @classmethod
    def from_coordinate(cls, coordinate):
        return cls(radius=coordinate.radius, colatitude=coordinate.colatitude, azimuth=coordinate.azimuth)

    def __init__(self, radius=None, colatitude=None, azimuth=None, **kwargs):
        super().__init__(**kwargs)
        self._radius = np.asarray(radius if radius is not None else 1)
        self._colatitude = np.asarray(colatitude if colatitude is not None else 0)
        self._azimuth = np.asarray(azimuth if azimuth is not None else 0)

    def copy(self, deep=False):
        if deep:
            return type(self)(radius=self._radius.copy(), colatitude=self._colatitude.copy(), azimuth=self._azimuth.copy(), automesh=self.automesh)
        else:
            return type(self)(radius=self._radius, colatitude=self._colatitude, azimuth=self._azimuth, automesh=self.automesh)

    @property
    def radius(self):
        return np.reshape(self._radius, self.shapes.radius)

    @property
    def colatitude(self):
        return np.reshape(self._colatitude, self.shapes.colatitude)

    @property
    def azimuth(self):
        return np.reshape(self._azimuth, self.shapes.azimuth)

    @property
    def spherical_mesh(self):
        return np.stack(np.meshgrid(self.radius, self.colatitude, self.azimuth, indexing='ij', sparse=False), axis=-1).squeeze()

    class _ShapeClass(_SphericalConverter._ShapeClass):
        @property
        def radius(self):
            if self.owner.automesh:
                return np.shape(self.owner._radius) + (1,) * (np.ndim(self.owner._colatitude) + np.ndim(self.owner._azimuth))
            else:
                return np.shape(self.owner._radius)

        @property
        def colatitude(self):
            if self.owner.automesh:
                return (1,) * np.ndim(self.owner._radius) + np.shape(self.owner._colatitude) + (1,) * np.ndim(self.owner._azimuth)
            else:
                return np.shape(self.owner._colatitude)

        @property
        def azimuth(self):
            if self.owner.automesh:
                return (1,) * (np.ndim(self.owner._radius) + np.ndim(self.owner._colatitude)) + np.shape(self.owner._azimuth)
            else:
                return np.shape(self.owner._azimuth)

        @property
        def shape(self):
            return self.broadcast_shapes(self.radius, self.colatitude, self.azimuth)


class SphericalMesh(_SphericalConverter):
    @classmethod
    def from_coordinate(cls, coordinate):
        return cls(mesh=coordinate.spherical_mesh)

    def __init__(self, mesh, **kwargs):
        kwargs['automesh'] = False  # Should not apply meshing to a mesh again.
        super().__init__(**kwargs)
        self._spherical_mesh = np.asarray(mesh)

    def copy(self, deep=False):
        if deep:
            return type(self)(mesh=self._spherical_mesh.copy(), automesh=self.automesh)
        else:
            return type(self)(mesh=self._spherical_mesh, automesh=self.automesh)

    @property
    def radius(self):
        return self._spherical_mesh[..., 0]

    @property
    def colatitude(self):
        return self._spherical_mesh[..., 1]

    @property
    def azimuth(self):
        return self._spherical_mesh[..., 2]

    @property
    def spherial_mesh(self):
        return self._spherial_mesh

    @property
    def shape(self):
        return np.shape(self._spherical_mesh)[:-1]

    class _ShapeClass(_SphericalConverter._ShapeClass):
        def shape(self):
            return np.shape(self.owner._spherical_mesh)[:-1]
        radius = property(shape)
        colatitude = property(shape)
        azimuth = property(shape)
        shape = property(shape)


class Rotation(Coordinate):
    @staticmethod
    def z_axes_rotation_angles(new_axis=None, old_axis=None):
        if new_axis is None and old_axis is None:
            raise ValueError('Must define at least the new z-axis or the old z-axis')

        if new_axis is None:
            _, colatitude, azimuth_old = CartesianMesh(old_axis).radius_colatitude_azimuth
            azimuth_new = 0
        elif old_axis is None:
            _, colatitude, azimuth_new = CartesianMesh(new_axis).radius_colatitude_azimuth
            azimuth_old = np.pi
        else:
            _, colatitude, azimuth_new = CartesianMesh(new_axis).radius_colatitude_azimuth
            _, colatitude_old, azimuth_old = CartesianMesh(old_axis).radius_colatitude_azimuth
            if not np.allclose(colatitude, colatitude_old):
                raise ValueError('New z-axis and old z-axis does not have the same angle between them, check the inputs!')
        return colatitude, azimuth_new, np.pi - azimuth_old

    @classmethod
    def parse_args(cls, position=None, *, colatitude=None, azimuth=None, secondary_azimuth=None, new_z_axis=None, old_z_axis=None):
        obj = super().parse_args(position)
        if obj is not None:
            return obj
        if new_z_axis is not None or old_z_axis is not None:
            colatitude, azimuth, secondary_azimuth = cls.z_axes_rotation_angles(new_axis=new_z_axis, old_axis=old_z_axis)
        return Rotation(colatitude=colatitude, azimuth=azimuth, secondary_azimuth=secondary_azimuth)

    def __init__(self, colatitude=None, azimuth=None, secondary_azimuth=None, automesh=False, **kwargs):
        super().__init__(**kwargs)
        self.automesh = automesh
        self._colatitude = np.asarray(colatitude if colatitude is not None else 0)
        self._azimuth = np.asarray(azimuth if azimuth is not None else 0)
        self._secondary_azimuth = np.asarray(secondary_azimuth if secondary_azimuth is not None else 0)

    @property
    def colatitude(self):
        return np.reshape(self._colatitude, self.shapes.colatitude)

    @property
    def azimuth(self):
        return np.reshape(self._azimuth, self.shapes.azimuth)

    @property
    def secondary_azimuth(self):
        return np.reshape(self._secondary_azimuth, self.shapes.secondary_azimuth)

    def copy(self, deep=False):
        if deep:
            return type(self)(colatitude=self._colatitude.copy(), azimuth=self._azimuth.copy(), secondary_azimuth=self._secondary_azimuth.copy(), automesh=self.automesh)
        else:
            return type(self)(colatitude=self._colatitude, azimuth=self._azimuth, secondary_azimuth=self._secondary_azimuth, automesh=self.automesh)

    @property
    def rotation_matrix(self):
        import scipy.spatial.transform
        return scipy.spatial.transform.Rotation.from_euler('zyz', [self.secondary_azimuth, self.colatitude, self.azimuth]).as_matrix()

    def apply(self, position=None, inverse=False, **kwargs):
        coordinate = SpatialCoordinate.parse_args(position=position, **kwargs)
        # As a result of that the mesh is of shape (..., 3), we have to do the matrix multiplication
        # from the right. The "normal" way is to do it from the left. This means that we have to
        # transpose the matrix to apply it in the forward way. Since a rotation matix is orthogonal, the
        # inverse rotation is described by the tranpose of the rotation matrix, which in this case ends up
        # as the rotation matrix, since we apply it from the right.
        if inverse:
            R = self.rotation_matrix
        else:
            R = self.rotation_matrix.T
        return CartesianMesh(coordinate.cartesian_mesh @ R)

    class _ShapeClass(Coordinate._ShapeClass):
        @property
        def colatitude(self):
            if self.owner.automesh:
                return np.shape(self.owner._colatitude) + (1,) * (np.ndim(self.owner._azimuth) + np.ndim(self.owner._secondary_azimuth))
            else:
                return np.shape(self.owner._colatitude)

        @property
        def azimuth(self):
            if self.owner.automesh:
                return (1,) * np.ndim(self.owner._colatitude) + np.shape(self.owner._azimuth) + (1,) * np.ndim(self.owner._secondary_azimuth)
            else:
                return np.shape(self.owner._azimuth)

        @property
        def secondary_azimuth(self):
            if self.owner.automesh:
                return (1,) * (np.ndim(self.owner._colatitude) + np.ndim(self.owner._azimuth)) + np.shape(self.owner._secondary_azimuth)
            else:
                return np.shape(self.owner._secondary_azimuth)

        @property
        def shape(self):
            return self.broadcast_shapes(self.colatitude, self.azimuth, self.secondary_azimuth)


class Translation(Coordinate):
    @classmethod
    def parse_args(cls, position=None, *, automesh=False,
                   x=None, y=None, z=None, cartesian_mesh=None,
                   radius=None, colatitude=None, azimuth=None, spherical_mesh=None):
        obj = super().parse_args(position)
        if obj is not None:
            return obj
        coordinate = SpatialCoordinate.parse_args(
            position=position, automesh=automesh,
            x=x, y=y, z=z, cartesian_mesh=cartesian_mesh,
            radius=radius, colatitude=colatitude, azimuth=azimuth, spherical_mesh=spherical_mesh)
        return cls(coordinate=coordinate)

    def __init__(self, coordinate):
        super().__init__()
        self.coordinate = coordinate

    def copy(self, deep=False):
        return type(self)(coordinate=self.coordinate.copy(deep=deep))

    def apply(self, position=None, inverse=False, **kwargs):
        coordinate = SpatialCoordinate.parse_args(position=position, **kwargs)
        if inverse:
            return Cartesian(x=coordinate.x - self.x, y=coordinate.y - self.y, z=coordinate.z - self.z)
        else:
            return Cartesian(x=coordinate.x + self.x, y=coordinate.y + self.y, z=coordinate.z + self.z)

    def _coordinate_property(key):
        def wrapped(self):
            return getattr(self.coordinate, key)
        return property(wrapped)

    x = _coordinate_property('x')
    y = _coordinate_property('y')
    z = _coordinate_property('z')
    radius = _coordinate_property('radius')
    colatitude = _coordinate_property('colatitude')
    azimuth = _coordinate_property('azimuth')
    cartesian_mesh = _coordinate_property('cartesian_mesh')
    spherical_mesh = _coordinate_property('spherical_mesh')
    xyz = _coordinate_property('xyz')
    radius_colatitude_azimuth = _coordinate_property('radius_colatitude_azimuth')

    class _ShapeClass(Coordinate._ShapeClass):
        def _coordinate_shape(key):
            def wrapped(self):
                return getattr(self.owner.coordinate.shapes, key)
            return property(wrapped)

        x = _coordinate_shape('x')
        y = _coordinate_shape('y')
        z = _coordinate_shape('z')
        radius = _coordinate_shape('radius')
        colatitude = _coordinate_shape('colatitude')
        azimuth = _coordinate_shape('azimuth')
        cartesian_mesh = _coordinate_shape('cartesian_mesh')
        spherical_mesh = _coordinate_shape('spherical_mesh')
        shape = _coordinate_shape('shape')
