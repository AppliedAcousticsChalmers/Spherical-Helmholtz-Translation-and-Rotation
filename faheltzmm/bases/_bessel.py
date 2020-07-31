import numpy as np
import scipy.special


class SphericalBessel:
    def __init__(self, order, x=None, shape=None):
        self._order = order
        self._shape = shape if shape is not None else np.shape(x)
        # self._data = np.zeros((self.num_unique,) + shape, dtype=float)  # Not needed since the evaluation always creates a new copy

        if x is not None:
            self.evaluate(x)

    def evaluate(self, x):
        order = np.arange(self.order + 1).reshape([-1] + [1] * len(self.shape))
        self._data = scipy.special.spherical_jn(order, x, derivative=False)

    @property
    def order(self):
        return self._order

    @property
    def num_unique(self):
        return self.order + 1

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
