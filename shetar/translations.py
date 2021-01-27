import numpy as np
from . import coordinates, rotations, expansions, _shape_utilities


class CoaxialTranslation:
    _default_output_type = expansions.Expansion

    def __init__(self, input_order, output_order, distance=None, wavenumber=None):
        self._input_order = input_order
        self._output_order = output_order
        self._min_order = min(self.input_order, self.output_order)
        self._max_order = max(self.input_order, self.output_order)
        self._wavenumber = None

        num_unique = (
            (self._min_order + 1)**2 * (self._max_order + 1)
            - (self._min_order * (self._min_order + 1)) // 2 * (self._min_order + self._max_order + 2)
            + (self._min_order * (self._min_order - 1) * (self._min_order + 1)) // 6
        )
        self._data = np.zeros((num_unique,) + np.shape(wavenumber) + np.shape(distance), self._dtype)

        if _shape_utilities.is_value(distance):
            self.evaluate(distance=distance, wavenumber=wavenumber)

    @property
    def order(self):
        return (self.input_order, self.output_order)

    @property
    def input_order(self):
        return self._input_order

    @property
    def output_order(self):
        return self._output_order

    @property
    def shape(self):
        return np.shape(self._data)[(1 + np.ndim(self._wavenumber)):]

    @property
    def ndim(self):
        return len(self.shape)

    def copy(self, deep=False):
        new_obj = type(self).__new__(type(self))
        new_obj._input_order = self._input_order
        new_obj._output_order = self._output_order
        new_obj._max_order = self._max_order
        new_obj._min_order = self._min_order
        new_obj._wavenumber = self._wavenumber
        if deep:
            new_obj._data = self._data.copy()
        else:
            new_obj._data = self._data
        return new_obj

    def reshape(self, newshape, *args, **kwargs):
        new_obj = self.copy()
        non_shape_dims = new_obj._data.ndim - new_obj.ndim
        new_obj._data = new_obj._data.reshape(new_obj._data.shape[:non_shape_dims] + tuple(newshape))
        return new_obj

    @property
    def wavenumber(self):
        return self._wavenumber

    @property
    def _component_indices(self):
        out = []
        for m in range(self._min_order + 1):
            for n in range(m, self._min_order + 1):
                for p in range(n, self._max_order + 1):
                    out.append((n, p, m))
        return out

    def _idx(self, input_order=None, output_order=None, mode=None, index=None):
        def mode_offset(m):
            return (m * (self._min_order + 1) * (self._max_order + 1)
                    - (self._max_order + 1) * (m * (m - 1)) // 2
                    - m * (self._min_order * (self._min_order + 1)) // 2
                    + (m * (m - 1) * (m - 2)) // 6
                    )

        def input_order_offset(n, m):
            return (n - m) * (self._max_order + 1) + (m * (m - 1) - n * (n - 1)) // 2

        if index is None:
            # Default mode, getting the linear index of the component indices
            if abs(mode) > input_order:
                raise IndexError(f'Mode {mode} is out of bounds for input order {input_order}')
            if abs(mode) > output_order:
                raise IndexError(f'Mode {mode} is out of bounds for output order {output_order}')
            if input_order > self._min_order:
                raise IndexError(f'Component {(input_order, output_order, mode)} not stored in {self.__class__.__name__} (n > min(N, P). Use getter or index the object directly.')
            if output_order > self._max_order:
                raise IndexError(f'Component {(input_order, output_order, mode)} not stored in {self.__class__.__name__} (p > max(N, P)). Use getter or index the object directly.')
            if input_order > output_order:
                raise IndexError(f'Component {(input_order, output_order, mode)} not stored in {self.__class__.__name__}. Use getter or index the object directly.')
            if mode < 0:
                raise IndexError(f'Component {(input_order, output_order, mode)} not stored in {self.__class__.__name__}. Use getter or index the object directly.')
            # Data is stored in [mode, input_order, output_order] (m, n, p) order
            return mode_offset(mode) + input_order_offset(input_order, mode) + output_order - input_order
        else:
            # Inverse mode, getting the component indices of a linear index.
            mode = 0
            while mode_offset(mode + 1) <= index:
                mode += 1
            index -= mode_offset(mode)
            input_order = mode
            while input_order_offset(input_order + 1, mode) <= index:
                input_order += 1
            index -= input_order_offset(input_order, mode)
            output_order = index + input_order
            return (input_order, output_order, mode)

    def __getitem__(self, key):
        n, p, m = key
        if n > p:
            return (-1)**(n + p) * self._data[self._idx(p, n, abs(m))]
        else:
            return self._data[self._idx(n, p, abs(m))]

    def evaluate(self, distance, wavenumber=None):
        if wavenumber is not None:
            self._wavenumber = wavenumber
        # The computation is split in two domains for p, the stored and the buffered.
        # The stored range is p <= P, i.e. the values we are interested in.
        # The buffered range is P < p <= N + P - m, which are values needed to
        # complete the recurrence to higher n and m.
        # The values (n, P, m) exist in both domains, due to simplifications of
        # the indexing and implementations for the buffered values.

        # We can get away with only two buffers for n since we need (n-2, p, m)
        # only when calculating (n, p, m), and never again after that point.
        # By calculating and storing in the same statement, we can reuse the same memory directly.

        # Recurrence buffers
        N, P = self._min_order, self._max_order  # Shorthand for clarity
        m_buffer = np.zeros((N + 1,) + np.shape(self.wavenumber) + self.shape, dtype=self._dtype)
        m_minus_one_buffer = np.zeros((N + 1,) + np.shape(self.wavenumber) + self.shape, dtype=self._dtype)
        n_minus_two_buffer = np.zeros((N + 1,) + np.shape(self.wavenumber) + self.shape, dtype=self._dtype)
        n_minus_one_buffer = np.zeros((N + 1,) + np.shape(self.wavenumber) + self.shape, dtype=self._dtype)

        for m in range(N + 1):  # Main loop over modes
            # Buffer swap for sectorials.
            m_buffer, m_minus_one_buffer = m_minus_one_buffer, m_buffer
            if m == 0:  # Get starting values: (0, p, 0)
                # Somewhat ugly with this if-statement inside the loop,
                # but the alternative is to duplicate everything else in the loop
                # for m=0.
                initial_values = self._recurrence_initialization(order=N + P, radius=distance, wavenumber=self.wavenumber)
                for p in range(P):
                    self._data[self._idx(0, p, 0)] = initial_values[p] * (2 * p + 1)**0.5
                self._data[self._idx(0, P, 0)] = m_buffer[0] = n_minus_one_buffer[0] = initial_values[P] * (2 * P + 1)**0.5
                for p in range(P + 1, N + P + 1):
                    m_buffer[p - P] = n_minus_one_buffer[p - P] = initial_values[p] * (2 * p + 1)**0.5
            else:
                for p in range(m, P):  # Sectorial recurrence in the stored range
                    self._data[self._idx(m, p, m)] = (
                        self._data[self._idx(m - 1, p - 1, m - 1)] * ((p + m - 1) * (p + m) * (2 * m + 1) / ((2 * p - 1) * (2 * p + 1) * (2 * m)))**0.5
                        + self._data[self._idx(m - 1, p + 1, m - 1)] * ((p - m + 1) * (p - m + 2) * (2 * m + 1) / ((2 * p + 1) * (2 * p + 3) * (2 * m)))**0.5
                    )
                self._data[self._idx(m, P, m)] = m_buffer[0] = n_minus_one_buffer[0] = (
                    self._data[self._idx(m - 1, P - 1, m - 1)] * ((P + m - 1) * (P + m) * (2 * m + 1) / ((2 * P - 1) * (2 * P + 1) * (2 * m)))**0.5
                    + m_minus_one_buffer[1] * ((P - m + 1) * (P - m + 2) * (2 * m + 1) / ((2 * P + 1) * (2 * P + 3) * (2 * m)))**0.5
                )

                for p in range(P + 1, N + P - m + 1):  # Sectorial recurrence in the buffered range
                    m_buffer[p - P] = n_minus_one_buffer[p - P] = (
                        m_minus_one_buffer[p - P - 1] * ((p + m - 1) * (p + m) * (2 * m + 1) / ((2 * p - 1) * (2 * p + 1) * (2 * m)))**0.5
                        + m_minus_one_buffer[p - P + 1] * ((p - m + 1) * (p - m + 2) * (2 * m + 1) / ((2 * p + 1) * (2 * p + 3) * (2 * m)))**0.5
                    )
            # Remaining (non-sectorial) values.
            # n = m + 1 is a special case since n-2 < m removes one component from the recurrence
            if m < N:  # Needed to prevent n = N + 1
                scale = (2 * m + 3)**0.5
                for p in range(m + 1, P):  # n = m - 1, stored range
                    self._data[self._idx(m + 1, p, m)] = scale * (
                        self._data[self._idx(m, p - 1, m)] * ((p + m) * (p - m) / ((2 * p - 1) * (2 * p + 1)))**0.5
                        - self._data[self._idx(m, p + 1, m)] * ((p + m + 1) * (p - m + 1) / ((2 * p + 1) * (2 * p + 3)))**0.5
                    )
                self._data[self._idx(m + 1, P, m)] = n_minus_two_buffer[0] = scale * (
                    self._data[self._idx(m, P - 1, m)] * ((P + m) * (P - m) / ((2 * P - 1) * (2 * P + 1)))**0.5
                    - n_minus_one_buffer[1] * ((P + m + 1) * (P - m + 1) / ((2 * P + 1) * (2 * P + 3)))**0.5
                )
                for p in range(P + 1, N + P - m):  # n = m - 1, buffered range
                    n_minus_two_buffer[p - P] = scale * (
                        n_minus_one_buffer[p - P - 1] * ((p + m) * (p - m) / ((2 * p - 1) * (2 * p + 1)))**0.5
                        - n_minus_one_buffer[p - P + 1] * ((p + m + 1) * (p - m + 1) / ((2 * p + 1) * (2 * p + 3)))**0.5
                    )

            for n in range(m + 2, N + 1):  # Main loop over n.
                # Buffer swap for n.
                n_minus_one_buffer, n_minus_two_buffer = n_minus_two_buffer, n_minus_one_buffer
                scale = ((2 * n - 1) * (2 * n + 1) / ((n + m) * (n - m)))**0.5
                for p in range(n, P):  # Stored range
                    self._data[self._idx(n, p, m)] = scale * (
                        self._data[self._idx(n - 2, p, m)] * ((n + m - 1) * (n - m - 1) / ((2 * n - 3) * (2 * n - 1)))**0.5
                        + self._data[self._idx(n - 1, p - 1, m)] * ((p + m) * (p - m) / ((2 * p - 1) * (2 * p + 1)))**0.5
                        - self._data[self._idx(n - 1, p + 1, m)] * ((p + m + 1) * (p - m + 1) / ((2 * p + 1) * (2 * p + 3)))**0.5
                    )
                self._data[self._idx(n, P, m)] = n_minus_two_buffer[0] = scale * (
                    self._data[self._idx(n - 2, P, m)] * ((n + m - 1) * (n - m - 1) / ((2 * n - 3) * (2 * n - 1)))**0.5
                    + self._data[self._idx(n - 1, P - 1, m)] * ((P + m) * (P - m) / ((2 * P - 1) * (2 * P + 1)))**0.5
                    - n_minus_one_buffer[1] * ((P + m + 1) * (P - m + 1) / ((2 * P + 1) * (2 * P + 3)))**0.5
                )
                for p in range(P + 1, N + P - n + 1):  # Buffered range
                    n_minus_two_buffer[p - P] = scale * (
                        n_minus_two_buffer[p - P] * ((n + m - 1) * (n - m - 1) / ((2 * n - 3) * (2 * n - 1)))**0.5
                        + n_minus_one_buffer[p - P - 1] * ((p + m) * (p - m) / ((2 * p - 1) * (2 * p + 1)))**0.5
                        - n_minus_one_buffer[p - P + 1] * ((p + m + 1) * (p - m + 1) / ((2 * p + 1) * (2 * p + 3)))**0.5
                    )
        return self

    def apply(self, expansion, inverse=False, out=None):
        wavenumber = getattr(expansion, 'wavenumber', None)
        if wavenumber is not None:
            if not np.allclose(wavenumber, self.wavenumber):
                raise ValueError('Cannot apply translation to expansion of different wavenuber')

        # TODO: Limit the order when the input expansion is lower order
        # TODO: Limit the order when the output expansion is lower order
        # TODO: output type for inverse translations? If the translation is exterior to interior, what should the inverse translation do?
        if out is None:
            self_shape = self.shape
            expansion_shape = np.shape(expansion)
            output_shape = np.broadcast(np.empty(self_shape, dtype=[]), np.empty(expansion_shape, dtype=[]))
            out = self._default_output_type(order=self.input_order if inverse else self.output_order, wavenumber=self.wavenumber, data=output_shape)
        elif expansion is out:
            raise NotImplementedError('Translations cannot currently be applied in place')
        if not inverse:
            for m in range(-self._min_order, self._min_order + 1):
                for p in range(abs(m), self.output_order + 1):
                    value = 0
                    for n in range(abs(m), self.input_order + 1):
                        value += self[n, p, m] * expansion[n, m]
                    out[p, m] = value
        else:
            for m in range(-self._min_order, self._min_order + 1):
                for n in range(abs(m), self.input_order + 1):
                    value = 0
                    for p in range(abs(m), self.output_order + 1):
                        value += self[n, p, m] * expansion[p, m]
                    out[n, m] = value
        return out


class InteriorCoaxialTranslation(CoaxialTranslation):
    _dtype = float
    from .bases import RegularRadialBase as _recurrence_initialization
    _default_output_type = expansions.InteriorExpansion


class ExteriorCoaxialTranslation(CoaxialTranslation):
    _dtype = float
    from .bases import RegularRadialBase as _recurrence_initialization
    _default_output_type = expansions.ExteriorExpansion


class ExteriorInteriorCoaxialTranslation(CoaxialTranslation):
    _dtype = complex
    from .bases import SingularRadialBase as _recurrence_initialization
    _default_output_type = expansions.InteriorExpansion


class Translation:
    def __init__(self, input_order, output_order, position=None, wavenumber=None,
                 radius=None, colatitude=None, azimuth=None):

        if position is not None:
            radial_shape = np.broadcast(position[0])
            colatitude_shape = np.broadcast(position[0])
            azimuth_shape = np.broadcast(position[0])
        else:
            radial_shape = np.broadcast(radius)
            colatitude_shape = np.broadcast(colatitude)
            azimuth_shape = np.broadcast(azimuth)

        self._coaxial = self._coaxial_cls(
            input_order=input_order, output_order=output_order,
            distance=radial_shape, wavenumber=wavenumber
        )
        self._rotation = rotations.Rotation(
            order=max(input_order, output_order),
            colatitude=colatitude_shape, azimuth=azimuth_shape
        )

        if _shape_utilities.is_value(position):
            self.evaluate(position=position, wavenumber=wavenumber)
        elif _shape_utilities.is_value(radius) and _shape_utilities.is_value(colatitude) and _shape_utilities.is_value(azimuth):
            self.evaluate(radius=radius, colatitude=colatitude, azimuth=azimuth, wavenumber=wavenumber)

    def evaluate(self, position=None, wavenumber=None, radius=None, colatitude=None, azimuth=None):
        if position is not None:
            radius, colatitude, azimuth = coordinates.cartesian_2_spherical(position)
        if radius is not None:
            self._coaxial.evaluate(distance=radius, wavenumber=wavenumber)
        self._rotation.evaluate(colatitude=colatitude, azimuth=azimuth)
        return self

    def apply(self, expansion, inverse=False):
        if not inverse:
            return expansion.apply(self._rotation, inverse=True).apply(self._coaxial).apply(self._rotation)
        else:
            raise NotImplementedError('Inverse translations not implemented yet.')

    @property
    def shape(self):
        return _shape_utilities.broadcast_shapes(self._coaxial.shape, self._rotation.shape, output='new')

    @property
    def ndim(self):
        return max(self._coaxial.ndim, self._rotation.ndim)

    def copy(self, deep=False):
        new_obj = type(self).__new__(type(self))
        new_obj._coaxial = self._coaxial.copy(deep=deep)
        new_obj._rotation = self._rotation.copy(deep=deep)
        return new_obj

    def reshape(self, newshape, *args, **kwargs):
        coaxial_newshape, rotation_newshape = _shape_utilities.broadcast_reshape(self._coaxial.shape, self._rotation.shape, newshape=newshape)
        new_obj = self.copy()
        new_obj._coaxial = new_obj._coaxial.reshape(coaxial_newshape)
        new_obj._rotation = new_obj._rotation.reshape(rotation_newshape)
        return new_obj


class InteriorTranslation(Translation):
    _coaxial_cls = InteriorCoaxialTranslation


class ExteriorTranslation(Translation):
    _coaxial_cls = ExteriorCoaxialTranslation


class ExteriorInteriorTranslation(Translation):
    _coaxial_cls = ExteriorInteriorCoaxialTranslation
