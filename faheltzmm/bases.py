import numpy as np
import scipy.special
from . import coordinates


class AssociatedLegendrePolynomials:
    def __init__(self, order, x=None, shape=None, normalization='orthonormal'):
        self._order = order
        self.normalization = normalization
        self._shape = shape if shape is not None else np.shape(x)
        num_unique = (self.order + 1) * (self.order + 2) // 2
        self._data = np.zeros((num_unique,) + self.shape, dtype=float)

        if x is not None:
            self.evaluate(x)

    @property
    def order(self):
        return self._order

    @property
    def shape(self):
        return self._shape

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
    def __init__(self, order, radius=None, wavenumber=None, shape=None):
        self._order = order
        self._shape = shape if shape is not None else np.shape(radius)

        if radius is not None:
            self.evaluate(radius, wavenumber)

    def evaluate(self, radius, wavenumber=None):
        self._wavenumber = wavenumber
        if wavenumber is not None:
            x = radius * np.reshape(wavenumber, np.shape(wavenumber) + (1,) * np.ndim(radius))
        else:
            x = radius
        order = np.arange(self.order + 1).reshape((-1,) + (1,) * (np.ndim(self.wavenumber) + len(self.shape)))
        self._data = self._radial_func(order, x)
        return self

    @property
    def order(self):
        return self._order

    @property
    def shape(self):
        return self._shape

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


class SphericalHarmonics:
    def __init__(self, order, colatitude=None, azimuth=None, shape=None):
        shape = shape if shape is not None else np.broadcast(colatitude, azimuth).shape
        self._legendre = AssociatedLegendrePolynomials(order, shape=shape, normalization='orthonormal')
        if azimuth is not None:
            self.evaluate(colatitude=colatitude, azimuth=azimuth)

    def evaluate(self, colatitude=None, azimuth=None):
        cosine_colatitude = np.cos(colatitude)
        self._legendre.evaluate(cosine_colatitude)
        self._azimuth = azimuth if np.iscomplexobj(azimuth) else np.exp(1j * azimuth)
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
        return self._legendre.shape


class SphericalBase:
    def __init__(self, order, position=None, wavenumber=None, shape=None):
        shape = shape if shape is not None else np.shape(position)[1:]
        self._angular = SphericalHarmonics(order=order, shape=shape)
        self._radial = self._radial_cls(order=order, shape=shape)
        if position is not None:
            self.evaluate(position, wavenumber)

    def evaluate(self, position=None, wavenumber=None, radius=None, colatitude=None, azimuth=None):
        radius, colatitude, azimuth = coordinates.cartesian_2_spherical(position)
        self._radial.evaluate(radius, wavenumber)
        self._angular.evaluate(colatitude=colatitude, azimuth=azimuth)
        return self

    @property
    def order(self):
        return self._angular.order

    @property
    def shape(self):
        return self._angular.shape

    @property
    def wavenumber(self):
        return self._radial.wavenumber

    def __getitem__(self, key):
        order, mode = key
        return self._radial[order] * self._angular[order, mode]

    def apply(self, expansion):
        value = 0
        for n in range(min(self.order, expansion.order) + 1):
            for m in range(-n, n + 1):
                value += self[n, m] * expansion[n, m]
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
