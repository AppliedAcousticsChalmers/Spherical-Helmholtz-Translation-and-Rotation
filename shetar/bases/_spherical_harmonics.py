import numpy as np
from .. import coordinates
from ._legendre import AssociatedLegendrePolynomials


class SphericalHarmonics(coordinates.OwnerMixin):
    def __init__(self, order, position=None, colatitude=None, azimuth=None, defer_evaluation=False, *args, **kwargs):
        self.coordinate = coordinates.SpatialCoordinate.parse_args(position=position, colatitude=colatitude, azimuth=azimuth)
        self._legendre = AssociatedLegendrePolynomials(order, position=self.coordinate, normalization='orthonormal', defer_evaluation=True)
        if not defer_evaluation:
            self.evaluate(self.coordinate)
        else:
            self._phase = np.ones(self.shape, complex)

    def evaluate(self, position=None, colatitude=None, azimuth=None):
        self.coordinate = coordinates.SpatialCoordinate.parse_args(position=position, colatitude=colatitude, azimuth=azimuth)
        if (position is not None) or (colatitude is not None):
            self._legendre.evaluate(position=self.coordinate)
        if (position is not None) or (azimuth is not None):
            self._phase = np.asarray(np.exp(1j * self.coordinate.azimuth))
        return self

    def apply(self, expansion):
        value = 0
        for n in range(self.order + 1):
            for m in range(-n, n + 1):
                value += self[n, m] * expansion[n, m]
        return value

    def __getitem__(self, key):
        order, mode = key
        return self._legendre[order, mode] * self._phase ** mode / (2 * np.pi)**0.5

    @property
    def order(self):
        return self._legendre.order

    @property
    def shape(self):
        return self.coordinate.shapes.angular

    def copy(self, deep=False):
        new_obj = super().copy(deep=deep)
        new_obj._legendre = self._legendre.copy(deep=deep)
        if deep:
            new_obj._phase = self._phase.copy()
        else:
            new_obj._phase = self._phase
        return new_obj
