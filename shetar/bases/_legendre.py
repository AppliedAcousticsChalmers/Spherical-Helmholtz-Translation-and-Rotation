import numpy as np
import scipy.special
from .. import coordinates

from ._legendre_cython import legendre_polynomials, associated_legendre_polynomials


class LegendrePolynomials(coordinates.OwnerMixin):
    def __init__(self, order, position=None, colatitude=None, x=None, normalization='orthonormal', defer_evaluation=False):
        if x is not None:
            self.coordinate = coordinates.NonspatialCoordinate(x=x)
        else:
            self.coordinate = coordinates.SpatialCoordinate.parse_args(position=position, colatitude=colatitude)
        self.normalization = normalization
        self._data = np.zeros((self.num_unique(order),) + self.shape, dtype=float)

        if not defer_evaluation:
            self.evaluate(self.coordinate)

    @property
    def order(self):
        return self._data.shape[0] - 1

    @classmethod
    def num_unique_to_order(cls, num_unique):
        return num_unique - 1

    @classmethod
    def num_unique(cls, order):
        return order + 1

    @property
    def shape(self):
        if isinstance(self.coordinate, coordinates.NonspatialCoordinate):
            return self.coordinate.shape
        return self.coordinate.shapes.colatitude

    def copy(self, deep=False):
        new_obj = super().copy(deep=deep)
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

    @property
    def _x(self):
        if isinstance(self.coordinate, coordinates.NonspatialCoordinate):
            return self.coordinate.x
        if isinstance(self.coordinate, coordinates.SpatialCoordinate):
            return np.cos(self.coordinate.colatitude)

    def evaluate(self, position=None, colatitude=None, x=None):
        if x is not None:
            self.coordinate = coordinates.NonspatialCoordinate(x=x)
        else:
            self.coordinate = coordinates.SpatialCoordinate.parse_args(position=position, colatitude=colatitude)
        

        legendre_polynomials(self._x, out=self._data)
        return self

    def _normalization_factor(self, order, mode=0):
        if 'complement' in self.normalization.lower():
            return 1

        if mode == 0:
            ortho_norm = 1
        else:
            ortho_norm = (1 - self._x**2) ** (abs(mode) / 2)

        if 'orthonormal' in self.normalization.lower():
            return ortho_norm

        if mode == 0:
            numer = 2
            denom = (2 * order + 1)
        else:
            numer = 2 * scipy.special.factorial(order + mode)
            denom = scipy.special.factorial(order - mode) * (2 * order + 1)

        if 'scipy' in self.normalization.lower():
            return ortho_norm * (numer / denom) ** 0.5

    def __getitem__(self, key):
        return self._data[key] * self._normalization_factor(key)

    def apply(self, expansion):
        value = 0
        for n in range(self.order + 1):
            value += self[n] * expansion[n]
        return value


class AssociatedLegendrePolynomials(LegendrePolynomials):
    def __init__(self, order, position=None, colatitude=None, x=None, normalization='orthonormal', defer_evaluation=False):
        if x is not None:
            self.coordinate = coordinates.NonspatialCoordinate(x=x)
        else:
            self.coordinate = coordinates.SpatialCoordinate.parse_args(position=position, colatitude=colatitude)
        self.normalization = normalization
        self._order = order
        self._data = np.zeros((self.num_unique(self.order),) + self.shape, dtype=float)

        if not defer_evaluation:
            self.evaluate(self.coordinate)

    @property
    def order(self):
        return self._order

    @classmethod
    def num_unique_to_order(cls, num_unique):
        int((8 * num_unique + 1)**0.5 - 3) // 2

    @classmethod
    def num_unique(cls, order):
        return (order + 1) * (order + 2) // 2

    def evaluate(self, position=None, colatitude=None, x=None):
        if x is not None:
            self.coordinate = coordinates.NonspatialCoordinate(x=x)
        else:
            self.coordinate = coordinates.SpatialCoordinate.parse_args(position=position, colatitude=colatitude)

        associated_legendre_polynomials(self._x, out=self._data)
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
        return value * sign * self._normalization_factor(order, mode)

    def apply(self, expansion):
        value = 0
        for n in range(self.order + 1):
            for m in range(-n, n + 1):
                value += self[n, m] * expansion[n, m]
        return value