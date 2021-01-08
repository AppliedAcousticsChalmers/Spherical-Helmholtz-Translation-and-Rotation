import numpy as np
from . import coordinates, _is_value


class Expansion:
    def __init__(self, order=None, data=None, wavenumber=None):
        self._wavenumber = wavenumber
        if _is_value(data):
            self._data = data
            if order is not None and self.order != order:
                raise ValueError(f'Received data of order {self.order} in conflict with specified order {order}')
            if np.shape(wavenumber) != np.shape(data)[1:(np.ndim(wavenumber) + 1)]:
                raise ValueError(f'Received wavenumber of shape {np.shape(wavenumber)} in conflict with data of shape {np.shape(data)}')
        elif type(data) == np.broadcast and order is None:
            self._data = np.zeros(np.shape(data), dtype=complex)
        elif order is not None:
            num_unique = (order + 1) ** 2
            self._data = np.zeros((num_unique,) + np.shape(wavenumber) + np.shape(data), dtype=complex)
        else:
            raise ValueError('Cannot initialize expansion without either raw data or known order')

    @property
    def order(self):
        return int(np.shape(self._data)[0] ** 0.5) - 1

    @property
    def shape(self):
        return np.shape(self._data)[(1 + np.ndim(self._wavenumber)):]

    @property
    def ndim(self):
        return len(self.shape)

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

    def apply(self, transform, *args, **kwargs):
        return transform.apply(self, *args, **kwargs)

