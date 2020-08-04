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

    def _idx(self, order, mode):
        # TODO: Should these checks be in the getter? That would simplify this function for internal use. Thinking of cython with nogil.
        if order < 0 or order > self.order:
            raise IndexError(f'Order {order} is out of bounds for {self.__class__.__name__} with max order {self.order}')
        if abs(mode) > order:
            raise IndexError(f'Mode {mode} is out of bounds for order {order}')
        return order * (order + 1) // 2 + abs(mode)

    def __getitem__(self, key):
        order, mode = key
        value = self._data[self._idx(order, mode)]
        if 'complement' in self.normalization.lower():
            if mode < 0:
                return value * (-1) ** mode
            else:
                return value.copy()  # Copy to make sure that it's not possible to modify the internal array in place.
        if 'orthonormal' in self.normalization.lower():
            norm = (1 - self._x**2) ** (abs(mode) / 2)
            if mode < 0:
                return value * (-1)**mode * norm
            else:
                return value * norm
        if 'scipy' in self.normalization.lower():
            numer = 2 * scipy.special.factorial(order + mode)
            denom = scipy.special.factorial(order - mode) * (2 * order + 1)
            norm = (1 - self._x**2) ** (abs(mode) / 2) * (numer / denom) ** 0.5
            if mode < 0:
                return value * (-1)**mode * norm
            else:
                return value * norm


class SphericalBessel:
    def __init__(self, order, x=None, shape=None):
        self._order = order
        self._shape = shape if shape is not None else np.shape(x)

        if x is not None:
            self.evaluate(x)

    def evaluate(self, x):
        order = np.arange(self.order + 1).reshape([-1] + [1] * len(self.shape))
        self._data = scipy.special.spherical_jn(order, x, derivative=False)

    @property
    def order(self):
        return self._order

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, key):
        return self._data[key]


class SphericalHankel(SphericalBessel):
    def evaluate(self, x):
        order = np.arange(self.order + 1).reshape([-1] + [1] * len(self.shape))
        self._data = scipy.special.spherical_jn(order, x, derivative=False) + 1j * scipy.special.spherical_yn(order, x, derivative=False)


class DualSphericalBessel(SphericalBessel):
    def evaluate(self, x):
        order = np.arange(self.order + 1).reshape([-1] + [1] * len(self.shape))
        bessel = scipy.special.spherical_jn(order, x, derivative=False)
        hankel = scipy.special.spherical_yn(order, x, derivative=False)
        self._data = np.stack([bessel, bessel + 1j * hankel], axis=1)


class SphericalHarmonics:
    def __init__(self, order, colatitude=None, azimuth=None, cosine_colatitude=None, shape=None):
        shape = shape if shape is not None else np.broadcast(cosine_colatitude if cosine_colatitude is not None else colatitude, azimuth).shape
        self._legendre = AssociatedLegendrePolynomials(order, shape=shape, normalization='orthonormal')
        if azimuth is not None:
            self.evaluate(colatitude=colatitude, azimuth=azimuth, cosine_colatitude=cosine_colatitude)

    def evaluate(self, colatitude=None, azimuth=None, cosine_colatitude=None):
        cosine_colatitude = np.cos(colatitude) if cosine_colatitude is None else cosine_colatitude
        self._legendre.evaluate(cosine_colatitude)
        self._azimuth = azimuth if np.iscomplexobj(azimuth) else np.exp(1j * azimuth)

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
        shape = shape if shape is not None else np.shape(wavenumber) + np.shape(position)[1:]
        self._angular = SphericalHarmonics(order=order, shape=shape)
        self._radial = self._radial_cls(order=order, shape=shape)
        if position is not None:
            self.evaluate(position, wavenumber)

    def evaluate(self, position, wavenumber):
        r, cos_theta, _, cos_phi, sin_phi = coordinates.cartesian_2_trigonometric(position)
        kr = r * np.reshape(wavenumber, np.shape(wavenumber) + (1,) * np.ndim(r))
        self._radial.evaluate(kr)
        self._angular.evaluate(cosine_colatitude=cos_theta, azimuth=cos_phi + 1j * sin_phi)

    @property
    def order(self):
        return self._angular.order

    @property
    def shape(self):
        return self._angular.shape

    def __getitem__(self, key):
        order, mode = key
        return self._radial[order] * self._angular[order, mode]


class RegularBase(SphericalBase):
    _radial_cls = SphericalBessel


class SingularBase(SphericalBase):
    _radial_cls = SphericalHankel


class DualBase(SphericalBase):
    _radial_cls = DualSphericalBessel

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
