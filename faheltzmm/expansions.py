import numpy as np


class Expansion:
    def __init__(self, order=None, shape=None, data=None, wavenumber=None):
        self._wavenumber = wavenumber

        if data is None:
            self._order = order
            self._shape = () if shape is None else shape
            num_unique = (self.order + 1) ** 2
            self._data = np.zeros((num_unique,) + np.shape(self.wavenumber) + self.shape, dtype=complex)
        else:
            self._order = (len(data) ** 0.5 - 1) // 1
            self._shape = np.shape(data)[1 + np.ndim(self.wavenumber):]
            self._data = data

    @property
    def order(self):
        return self._order

    @property
    def shape(self):
        return self._shape

    @property
    def wavenumber(self):
        return self._wavenumber

    def _idx(self, order=None, mode=None, index=None):
        if index is None:
            # The default mode, getting the linear index from the order and mode.
            if order > self.order:
                raise IndexError(f'Order {order} out of bounds for {self.__class__.__name__} with max order {self.order}')
            if abs(mode) > order:
                raise IndexError(f'Mode {mode} out of bounds for order {order}')
            return order ** 2 + order + mode
        else:
            # The inverse mode, getting the order and mode from the linear index.
            if index >= (self.order + 1)**2 or index < 0:
                raise IndexError(f'Index {index} out of bounds for {self.__class__.__name__} with max order {self.order}')
            order = int(index ** 0.5)
            mode = index - order * (order + 1)
            return (order, mode)

    @property
    def _coefficient_indices(self):
        out = []
        for n in range(self.order + 1):
            for m in range(-n, n + 1):
                out.append((n, m))
        return out

    def __getitem__(self, key):
        n, m, = key
        return self._data[self._idx(n, m)]

    def __setitem__(self, key, value):
        n, m, = key
        self._data[self._idx(n, m)] = value

    def _compatible_with(self, other):
        # TODO: Raise meaningful exceptions when the objects are not compatible.
        return (type(self) == type(other)
                and self.order == other.order
                and self.shape == other.shape
                and (self.wavenumber is other.wavenumber or np.allclose(self.wavenumber, other.wavenumber))
                )

    def __add__(self, other):
        if self._compatible_with(other):
            return type(self)(data=self._data + other._data, wavenumber=self.wavenumber)
        return NotImplemented

    def __iadd__(self, other):
        if self._compatible_with(other):
            self._data += other._data
            return self
        return NotImplemented

    def __sub__(self, other):
        if self._compatible_with(other):
            return type(self)(data=self._data - other._data, wavenumber=self.wavenumber)
        return NotImplemented

    def __isub__(self, other):
        if self._compatible_with(other):
            self._data -= other._data
            return self
        return NotImplemented

    def __mul__(self, other):
        return type(self)(data=self._data * other, wavenumber=self.wavenumber)

    def __imul__(self, other):
        self._data *= other
        return self

    def __div__(self, other):
        return type(self)(data=self._data / other, wavenumber=self.wavenumber)

    def __idiv__(self, other):
        self._data /= other
        return self

    def __neg__(self):
        return type(self)(data=-self._data, wavenumber=self.wavenumber)
