import numpy as np
from .. import coordinates
from ._spherical_harmonics import SphericalHarmonics
from ._radial import RegularRadialBase, SingularRadialBase, DualRadialBase


class SphericalBase(coordinates.OwnerMixin):
    def __init__(self, order, position=None, wavenumber=None,
                 radius=None, colatitude=None, azimuth=None, defer_evaluation=False):
        self.coordinate = coordinates.SpatialCoordinate.parse_args(position=position, radius=radius, colatitude=colatitude, azimuth=azimuth)
        self._angular = SphericalHarmonics(order=order, position=self.coordinate, defer_evaluation=defer_evaluation)
        self._radial = self._radial_cls(order=order, position=self.coordinate, wavenumber=wavenumber, defer_evaluation=defer_evaluation)

    def evaluate(self, position=None, wavenumber=None, radius=None, colatitude=None, azimuth=None):
        self.coordinate = coordinates.SpatialCoordinate.parse_args(position=position, radius=radius, colatitude=colatitude, azimuth=azimuth)
        if (position is not None) or (radius is not None) or (wavenumber is not None):
            self._radial.evaluate(self.coordinate, wavenumber=wavenumber)
        if (position is not None) or (colatitude is not None) or (azimuth is not None):
            # TODO: We could in principle optimize the case where only a new azimuth angle is given.
            self._angular.evaluate(self.coordinate)
        return self

    @property
    def order(self):
        return self._angular.order

    def copy(self, deep=False):
        new_obj = super().copy(deep=deep)
        new_obj._angular = self._angular.copy(deep=deep)
        new_obj._radial = self._radial.copy(deep=deep)
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
