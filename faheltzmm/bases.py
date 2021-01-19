import numpy as np
import scipy.special
from . import coordinates
from . import _shape_utilities


class AssociatedLegendrePolynomials:
    def __init__(self, order, x=None, normalization='orthonormal'):
        self._order = order
        self.normalization = normalization
        num_unique = (self.order + 1) * (self.order + 2) // 2
        self._data = np.zeros((num_unique,) + np.shape(x), dtype=float)

        if _shape_utilities.is_value(x):
            self.evaluate(x)

    @property
    def order(self):
        return self._order

    @property
    def shape(self):
        return self._data.shape[1:]

    @property
    def ndim(self):
        return len(self.shape)

    def copy(self, deep=False):
        new_obj = type(self).__new__(type(self))
        new_obj.normalization = self.normalization
        new_obj._order = self.order
        if deep:
            new_obj._data = self._data.copy()
        else:
            new_obj._data = self._data

        if hasattr(self, '_x'):
            if deep:
                new_obj._x = self._x.copy()
            else:
                new_obj._x = self._x
        return new_obj

    def reshape(self, newshape, *args, **kwargs):
        # The *args and **kwargs are needed to work correctly with np.reshape as a function.
        new_obj = self.copy()
        new_obj._data = np.reshape(new_obj._data, new_obj._data.shape[:1] + tuple(newshape))
        if hasattr(new_obj, '_x'):
            new_obj._x = np.reshape(new_obj._x, newshape)
        return new_obj

    def evaluate(self, x):
        self._x = x = np.asarray(x)
        one_minus_x_square = 1 - x**2

        # Access the data directly to bypass the normalization code in the getter.
        self._data[self._idx(0, 0)] = 2**-0.5
        for order in range(1, self.order + 1):
            # Recurrence to higher orders
            self._data[self._idx(order, order)] = - ((2 * order + 1) / (2 * order))**0.5 * self._data[self._idx(order - 1, order - 1)]
            # Same recurrence as below, but excluding the mode+2 part explicitly.
            self._data[self._idx(order, order - 1)] = - (2 * order)**0.5 * self._data[self._idx(order, order)] * x
            for mode in reversed(range(order - 1)):
                # Recurrece to lower modes
                self._data[self._idx(order, mode)] = - (
                    ((order + mode + 2) * (order - mode - 1) / (order - mode) / (order + mode + 1)) ** 0.5
                    * self._data[self._idx(order, mode + 2)] * one_minus_x_square
                    + 2 * (mode + 1) / ((order + mode + 1) * (order - mode))**0.5
                    * self._data[self._idx(order, mode + 1)] * x
                )
        return self

    def _idx(self, order=None, mode=None, index=None):
        if index is None:
            # Default mode, getting the linear index from the order and mode
            if mode < 0:
                raise IndexError(f'Mode {mode} not stored in {self.__class__.__name__}. Use getter or index the object directly')
            if order < 0 or order > self.order:
                raise IndexError(f'Order {order} is out of bounds for {self.__class__.__name__} with max order {self.order}')
            if mode > order:
                raise IndexError(f'Mode {mode} is out of bounds for order {order}')
            return order * (order + 1) // 2 + mode
        else:
            order = int((8 * index + 1)**0.5 - 1) // 2
            mode = index - order * (order + 1) // 2
            return (order, mode)

    @property
    def _component_indices(self):
        out = []
        for order in range(self.order + 1):
            for mode in range(order + 1):
                out.append((order, mode))
        return out

    def __getitem__(self, key):
        order, mode = key
        if mode < 0:
            sign = (-1) ** mode
        else:
            sign = 1
        value = self._data[self._idx(order, abs(mode))]
        if 'complement' in self.normalization.lower():
            return value * sign
        if 'orthonormal' in self.normalization.lower():
            norm = (1 - self._x**2) ** (abs(mode) / 2)
            return value * norm * sign
        if 'scipy' in self.normalization.lower():
            numer = 2 * scipy.special.factorial(order + mode)
            denom = scipy.special.factorial(order - mode) * (2 * order + 1)
            norm = (1 - self._x**2) ** (abs(mode) / 2) * (numer / denom) ** 0.5
            return value * norm * sign

    def apply(self, expansion):
        value = 0
        for n in range(self.order + 1):
            for m in range(-n, n + 1):
                value += self[n, m] * expansion[n, m]
        return value


class _RadialBaseClass:
    def __init__(self, order, radius=None, wavenumber=None):
        self._order = order
        self._wavenumber = wavenumber

        if _shape_utilities.is_value(radius):
            self.evaluate(radius, wavenumber)
        else:
            self._data = np.broadcast(radius)

    def evaluate(self, radius, wavenumber=None):
        if wavenumber is not None:
            self._wavenumber = wavenumber
        if self.wavenumber is None:
            x = radius
        else:
            x = radius * np.reshape(wavenumber, np.shape(wavenumber) + (1,) * np.ndim(radius))

        order = np.arange(self.order + 1).reshape((-1,) + (1,) * (np.ndim(self.wavenumber) + len(np.shape(radius))))
        self._data = self._radial_func(order, x)
        return self

    @property
    def order(self):
        return self._order

    @property
    def shape(self):
        return np.shape(self._data)[1 + np.ndim(self.wavenumber):]

    @property
    def ndim(self):
        return len(self.shape)

    def copy(self, deep=False):
        new_obj = type(self).__new__(type(self))
        new_obj._order = self.order
        new_obj._wavenumber = self.wavenumber
        if deep:
            new_obj._data = self._data.copy()
        else:
            new_obj._data = self._data
        return new_obj

    def reshape(self, newshape, *args, **kwargs):
        new_obj = self.copy()
        non_shape_dims = self._data.ndim - self.ndim
        new_obj._data = np.reshape(new_obj._data, new_obj._data.shape[:non_shape_dims] + tuple(newshape))
        return new_obj

    def __getitem__(self, key):
        return self._data[key]

    @property
    def wavenumber(self):
        return self._wavenumber


class RegularRadialBase(_RadialBaseClass):
    def _radial_func(self, order, x):
        return scipy.special.spherical_jn(order, x, derivative=False)


class SingularRadialBase(_RadialBaseClass):
    def _radial_func(self, order, x):
        return scipy.special.spherical_jn(order, x, derivative=False) + 1j * scipy.special.spherical_yn(order, x, derivative=False)


class DualRadialBase(_RadialBaseClass):
    def _radial_func(self, order, x):
        bessel = scipy.special.spherical_jn(order, x, derivative=False)
        neumann = scipy.special.spherical_yn(order, x, derivative=False)
        return np.stack([bessel, bessel + 1j * neumann], axis=1)

    @property
    def shape(self):
        return np.shape(self._data)[2 + np.ndim(self.wavenumber):]


class SphericalHarmonics:
    def __init__(self, order, position=None, colatitude=None, azimuth=None, *args, **kwargs):
        if position is not None:
            colatitude_shape = azimuth_shape = np.broadcast(position[0])
        else:
            colatitude_shape = np.broadcast(colatitude)
            azimuth_shape = np.broadcast(azimuth)
        self._legendre = AssociatedLegendrePolynomials(order, x=colatitude_shape, normalization='orthonormal')
        self._azimuth = azimuth_shape
        if _shape_utilities.is_value(position):
            self.evaluate(position=position)
        elif _shape_utilities.is_value(azimuth) and _shape_utilities.is_value(colatitude):
            self.evaluate(colatitude=colatitude, azimuth=azimuth)

    def evaluate(self, position=None, colatitude=None, azimuth=None):
        if position is not None:
            _, colatitude, azimuth = coordinates.cartesian_2_spherical(position)
        cosine_colatitude = np.cos(colatitude)
        self._legendre.evaluate(cosine_colatitude)
        self._azimuth = np.asarray(azimuth if np.iscomplexobj(azimuth) else np.exp(1j * azimuth))
        return self

    def apply(self, expansion):
        value = 0
        for n in range(self.order + 1):
            for m in range(-n, n + 1):
                value += self[n, m] * expansion[n, m]
        return value

    def __getitem__(self, key):
        order, mode = key
        return self._legendre[order, mode] * self._azimuth ** mode / (2 * np.pi)**0.5

    @property
    def order(self):
        return self._legendre.order

    @property
    def shape(self):
        return _shape_utilities.broadcast_shapes(self._legendre.shape, self._azimuth.shape, output='new')

    @property
    def ndim(self):
        return len(self.shape)

    def copy(self, deep=False):
        new_obj = type(self).__new__(type(self))
        new_obj._legendre = self._legendre.copy(deep=deep)
        if deep:
            new_obj._azimuth = self._azimuth.copy()
        else:
            new_obj._azimuth = self._azimuth
        return new_obj

    def reshape(self, newshape, *args, **kwargs):
        new_obj = self.copy()
        legendre_newshape, azimuth_newshape = _shape_utilities.broadcast_reshape(self._legendre.shape, self._azimuth.shape, newshape=newshape)
        new_obj._legendre = new_obj._legendre.reshape(legendre_newshape)
        new_obj._azimuth = new_obj._azimuth.reshape(azimuth_newshape)
        return new_obj


class SphericalBase:
    def __init__(self, order, position=None, wavenumber=None,
                 radius=None, colatitude=None, azimuth=None):
        if position is not None:
            radial_shape = np.broadcast(position[0])
            colatitude_shape = np.broadcast(position[0])
            azimuth_shape = np.broadcast(position[0])
        else:
            radial_shape = np.broadcast(radius)
            colatitude_shape = np.broadcast(colatitude)
            azimuth_shape = np.broadcast(azimuth)

        self._angular = SphericalHarmonics(order=order, colatitude=colatitude_shape, azimuth=azimuth_shape)
        self._radial = self._radial_cls(order=order, radius=radial_shape)
        if _shape_utilities.is_value(position):
            self.evaluate(position=position, wavenumber=wavenumber)
        elif _shape_utilities.is_value(radius) and _shape_utilities.is_value(colatitude) and _shape_utilities.is_value(azimuth):
            self.evaluate(radius=radius, colatitude=colatitude, azimuth=azimuth, wavenumber=wavenumber)

    def evaluate(self, position=None, wavenumber=None, radius=None, colatitude=None, azimuth=None):
        if position is not None:
            radius, colatitude, azimuth = coordinates.cartesian_2_spherical(position)
        self._radial.evaluate(radius, wavenumber)
        self._angular.evaluate(colatitude=colatitude, azimuth=azimuth)
        return self

    @property
    def order(self):
        return self._angular.order

    @property
    def shape(self):
        radial = np.empty(self._radial.shape, dtype=[])
        angular = np.empty(self._angular.shape, dtype=[])
        return np.broadcast(radial, angular).shape

    @property
    def ndim(self):
        return len(self.shape)

    def copy(self, deep=False):
        new_obj = type(self).__new__(type(self))
        new_obj._angular = self._angular.copy(deep=deep)
        new_obj._radial = self._radial.copy(deep=deep)
        return new_obj

    def reshape(self, newshape, *args, **kwargs):
        new_obj = self.copy()
        angular_newshape, radial_newshape = _shape_utilities.broadcast_reshape(self._angular.shape, self._radial.shape, newshape=newshape)
        new_obj._angular = new_obj._angular.reshape(angular_newshape)
        new_obj._radial = new_obj._radial.reshape(radial_newshape)
        return new_obj

    @property
    def wavenumber(self):
        return self._radial.wavenumber

    def __getitem__(self, key):
        order, mode = key
        return self._radial[order] * self._angular[order, mode]

    def apply(self, expansion):
        wavenumber = getattr(expansion, 'wavenumber', None)
        if wavenumber is not None:
            if not np.allclose(wavenumber, self.wavenumber):
                raise ValueError('Cannot apply bases to expansion of different wavenuber')
        else:
            # An expansion can be defined without a wavenumber, but for the
            # reshaping below to work properly, `wavenumber` needs to have the
            # same number of dimentions as `self.wavenumber`.
            wavenumber = np.reshape(wavenumber, np.ndim(self.wavenumber) * [1])

        ndim = max(self.ndim, expansion.ndim)
        exp_shape = np.shape(wavenumber) + (ndim - expansion.ndim) * (1,) + expansion.shape
        self_shape = np.shape(self.wavenumber) + (ndim - self.ndim) * (1,) + self.shape

        value = 0
        for n in range(min(self.order, expansion.order) + 1):
            for m in range(-n, n + 1):
                value += np.reshape(self[n, m], self_shape) * np.reshape(expansion[n, m], exp_shape)
        return value


class RegularBase(SphericalBase):
    _radial_cls = RegularRadialBase


class SingularBase(SphericalBase):
    _radial_cls = SingularRadialBase


class DualBase(SphericalBase):
    _radial_cls = DualRadialBase

    class _Regular:
        def __init__(self, parent):
            self.parent = parent

        def __getitem__(self, key):
            return self.parent[key][0]

    class _Singular:
        def __init__(self, parent):
            self.parent = parent

        def __getitem__(self, key):
            return self.parent[key][1]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._regular = self._Regular(self)
        self._singular = self._Singular(self)

    @property
    def regular(self):
        return self._regular

    @property
    def singular(self):
        return self._singular
