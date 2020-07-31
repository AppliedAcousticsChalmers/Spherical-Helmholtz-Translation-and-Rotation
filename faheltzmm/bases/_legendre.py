import numpy as np
from scipy.special import factorial


class AssociatedLegendrePolynomials:
    def __init__(self, order, x=None, shape=None, normalization='orthonormal'):
        self._order = order
        self.normalization = normalization
        self._shape = shape if shape is not None else np.shape(x)
        self._data = np.zeros((self.num_unique,) + self.shape, dtype=float)

        if x is not None:
            self.evaluate(x)

    @property
    def num_unique(self):
        return (self.order + 1) * (self.order + 2) // 2

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
            numer = 2 * factorial(order + mode)
            denom = factorial(order - mode) * (2 * order + 1)
            norm = (1 - self._x**2) ** (abs(mode) / 2) * (numer / denom) ** 0.5
            if mode < 0:
                return value * (-1)**mode * norm
            else:
                return value * norm
