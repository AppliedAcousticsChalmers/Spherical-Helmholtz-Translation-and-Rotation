import numpy as np
import scipy.special
from .. import coordinates


class RadialBaseClass(coordinates.OwnerMixin):
    def __init__(self, order, position=None, radius=None, wavenumber=None, defer_evaluation=False):
        self.coordinate = coordinates.SpatialCoordinate.parse_args(position=position, radius=radius)
        self._order = order
        self._wavenumber = wavenumber
        if not defer_evaluation:
            self.evaluate(self.coordinate)

    def evaluate(self, position=None, radius=None, wavenumber=None):
        self.coordinate = coordinates.SpatialCoordinate.parse_args(position=position, radius=radius)
        if wavenumber is not None:
            self._wavenumber = wavenumber
        if self.wavenumber is None:
            x = self.coordinate.radius
        else:
            x = self.coordinate.radius * np.reshape(self.wavenumber, np.shape(self.wavenumber) + (1,) * self.ndim)

        order = np.arange(self.order + 1).reshape((-1,) + (1,) * (np.ndim(self.wavenumber) + self.ndim))
        self._data = self._radial_func(order, x)
        return self

    @property
    def order(self):
        return self._order

    @property
    def shape(self):
        return self.coordinate.shapes.radius

    def copy(self, deep=False):
        new_obj = super().copy(deep=deep)
        new_obj._order = self.order
        new_obj._wavenumber = self.wavenumber
        if deep:
            new_obj._data = self._data.copy()
        else:
            new_obj._data = self._data
        return new_obj

    def __getitem__(self, key):
        return self._data[key]

    @property
    def wavenumber(self):
        return self._wavenumber


class RegularRadialBase(RadialBaseClass):
    def _radial_func(self, order, x):
        return scipy.special.spherical_jn(order, x, derivative=False)


class SingularRadialBase(RadialBaseClass):
    def _radial_func(self, order, x):
        return scipy.special.spherical_jn(order, x, derivative=False) + 1j * scipy.special.spherical_yn(order, x, derivative=False)


class DualRadialBase(RadialBaseClass):
    def _radial_func(self, order, x):
        bessel = scipy.special.spherical_jn(order, x, derivative=False)
        neumann = scipy.special.spherical_yn(order, x, derivative=False)
        return np.stack([bessel, bessel + 1j * neumann], axis=1)
