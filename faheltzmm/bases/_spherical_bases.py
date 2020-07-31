import numpy as np
from ._spherical_harmonics import SphericalHarmonics
from ..coordinates import cartesian_2_trigonometric


class SphericalBase:
    def __init__(self, order, position=None, wavenumber=None, shape=None):
        shape = shape if shape is not None else np.shape(wavenumber) + np.shape(position)[1:]
        self._angular = SphericalHarmonics(order=order, shape=shape)
        self._radial = self._radial_cls(order=order, shape=shape)
        if position is not None:
            self.evaluate(position, wavenumber)

    def evaluate(self, position, wavenumber):
        r, cos_theta, _, cos_phi, sin_phi = cartesian_2_trigonometric(position)
        kr = r * np.reshape(wavenumber, np.shape(wavenumber) + (1,) * np.ndim(r))
        self._radial.evaluate(kr)
        self._angular.evaluate(cosine_colatitude=cos_theta, azimuth=cos_phi + 1j * sin_phi)

    @property
    def order(self):
        return self._angular.order

    @property
    def shape(self):
        return self._angular.shape

    @property
    def num_unique(self):
        return self._angular.num_unique

    def __getitem__(self, key):
        order, mode = key
        return self._radial[order] * self._angular[order, mode]


class RegularBase(SphericalBase):
    from ._bessel import SphericalBessel as _radial_cls


class SingularBase(SphericalBase):
    from ._bessel import SphericalHankel as _radial_cls


class DualBase(SphericalBase):
    from ._bessel import DualSphericalBessel as _radial_cls

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
