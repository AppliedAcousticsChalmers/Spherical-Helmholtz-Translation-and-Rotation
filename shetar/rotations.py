import numpy as np
from . import coordinates, bases, expansions


class ColatitudeRotation(coordinates.OwnerMixin):
    def __init__(self, order, position=None, colatitude=None, defer_evaluation=False):
        self.coordinate = coordinates.Rotation.parse_args(position=position, colatitude=colatitude)
        self._order = order
        num_unique = (self.order + 1) * (self.order + 2) * (2 * self.order + 3) // 6
        self._data = np.zeros((num_unique,) + self.coordinate.shapes.colatitude, dtype=float)
        if not defer_evaluation:
            self.evaluate(self.coordinate)

    @property
    def order(self):
        return self._order

    @property
    def shape(self):
        return self.coordinate.shapes.colatitude

    def copy(self, deep=False):
        new_obj = super().copy(deep=deep)
        new_obj._order = self._order
        if deep:
            new_obj._data = self._data.copy()
        else:
            new_obj._data = self._data
        return new_obj

    def _idx(self, order=None, mode_out=None, mode_in=None, index=None):
        if index is None:
            # Default mode, get the linear index of a component
            if abs(mode_out) > order:
                raise IndexError(f'Mode {mode_out} is out of bounds for order {order}')
            if abs(mode_in) > order:
                raise IndexError(f'Mode {mode_in} is out of bounds for order {order}')
            if order < 0 or order > self.order:
                raise IndexError(f'Order {order} is out of bounds for {self.__class__.__name__} with max order {self.order}')
            if mode_out < 0:
                raise IndexError(f'Component {(order, mode_out, mode_in)} not stored in {self.__class__.__name__}. Use getter or index the object directly.')
            if abs(mode_in) > mode_out:
                raise IndexError(f'Component {(order, mode_out, mode_in)} not stored in {self.__class__.__name__}. Use getter or index the object directly.')

            return order * (order + 1) * (2 * order + 1) // 6 + mode_out ** 2 + mode_out + mode_in
        else:
            # Inverse mode, get the component indices of a linear index
            order = 0
            while (order + 1) * (order + 2) * (2 * order + 3) // 6 <= index:
                order += 1
            index -= order * (order + 1) * (2 * order + 1) // 6
            mode_out = int(index ** 0.5)
            mode_in = index - mode_out * (mode_out + 1)
            return order, mode_out, mode_in

    @property
    def _component_indices(self):
        out = []
        for n in range(self.order + 1):
            for p in range(n + 1):
                for m in range(-p, p + 1):
                    out.append((n, p, m))
        return out

    def __getitem__(self, key):
        n, p, m = key

        if abs(p) < m:
            p, m = m, p
            sign = (-1)**(m + p)
        elif abs(m) <= -p > 0:
            p, m = -p, -m
            sign = (-1)**(m + p)
        elif abs(p) < -m:
            p, m, = -m, -p
            sign = 1
        elif p >= 0 and abs(m) <= p:
            # If don't end up here it means we missed a case.
            # It would be obvious since the sign variable will not exist for the return statement.
            sign = 1
        return sign * self._data[self._idx(n, p, m)]

    def evaluate(self, position=None, colatitude=None):
        self.coordinate = coordinates.Rotation.parse_args(position=position, colatitude=colatitude)
        cosine_colatitude = np.cos(self.coordinate.colatitude)
        sine_colatitude = (1 - cosine_colatitude**2)**0.5

        legendre = bases.AssociatedLegendrePolynomials(order=self.order, x=cosine_colatitude)

        # n=0 and n=1 will be special cases since the p=n-1 and p=n special cases will not behave
        self._data[self._idx(0, 0, 0)] = 1
        if self.order > 0:
            self._data[self._idx(1, 0, 0)] = cosine_colatitude
            self._data[self._idx(1, 1, -1)] = (1 - cosine_colatitude) * 0.5
            self._data[self._idx(1, 1, 0)] = sine_colatitude * 2**-0.5
            self._data[self._idx(1, 1, 1)] = (1 + cosine_colatitude) * 0.5

        for n in range(2, self.order + 1):
            for p in range(0, n - 1):
                for m in range(-p, 0):
                    # Use recurrence for decreasing m here
                    self._data[self._idx(n, p, m)] = (
                        self._data[self._idx(n - 1, p - 1, m + 1)] * 0.5 * (1 - cosine_colatitude) * ((n + p - 1) * (n + p))**0.5
                        + self._data[self._idx(n - 1, p, m + 1)] * sine_colatitude * ((n + p) * (n - p))**0.5
                        + self._data[self._idx(n - 1, p + 1, m + 1)] * 0.5 * (1 + cosine_colatitude) * ((n - p - 1) * (n - p))**0.5
                    ) / ((n - m - 1) * (n - m))**0.5
                # Assign the m=0 value here
                self._data[self._idx(n, p, 0)] = legendre[n, p] * (-1)**p * (2 / (2 * n + 1))**0.5
                for m in range(1, p + 1):
                    # Use recurrence for increasing m here
                    self._data[self._idx(n, p, m)] = (
                        self._data[self._idx(n - 1, p - 1, m - 1)] * 0.5 * (1 + cosine_colatitude) * ((n + p - 1) * (n + p))**0.5
                        - self._data[self._idx(n - 1, p, m - 1)] * sine_colatitude * ((n + p) * (n - p))**0.5
                        + self._data[self._idx(n - 1, p + 1, m - 1)] * 0.5 * (1 - cosine_colatitude) * ((n - p - 1) * (n - p))**0.5
                    ) / ((n + m - 1) * (n + m))**0.5

            # p=n-1 and p=n are special cases since one or two values are missing in the recurrences
            # p = n-1
            for m in range(-(n - 1), 0):
                self._data[self._idx(n, n - 1, m)] = (
                    self._data[self._idx(n - 1, n - 2, m + 1)] * 0.5 * (1 - cosine_colatitude) * (2 * (n - 1) * (2 * n - 1))**0.5
                    + self._data[self._idx(n - 1, n - 1, m + 1)] * sine_colatitude * (2 * n - 1)**0.5
                ) / ((n - m - 1) * (n - m))**0.5
            self._data[self._idx(n, n - 1, 0)] = legendre[n, n - 1] * (-1)**(n - 1) * (2 / (2 * n + 1))**0.5
            for m in range(1, n):
                self._data[self._idx(n, n - 1, m)] = (
                    self._data[self._idx(n - 1, n - 1 - 1, m - 1)] * 0.5 * (1 + cosine_colatitude) * (2 * (n - 1) * (2 * n - 1))**0.5
                    - self._data[self._idx(n - 1, n - 1, m - 1)] * sine_colatitude * (2 * n - 1)**0.5
                ) / ((n + m - 1) * (n + m))**0.5
            # p = n
            for m in range(-n, 0):
                self._data[self._idx(n, n, m)] = (
                    self._data[self._idx(n - 1, n - 1, m + 1)] * 0.5 * (1 - cosine_colatitude) * ((2 * n - 1) * 2 * n)**0.5
                ) / ((n - m - 1) * (n - m))**0.5
            self._data[self._idx(n, n, 0)] = legendre[n, n] * (-1)**n * (2 / (2 * n + 1))**0.5
            for m in range(1, n + 1):
                # Use recurrence for increasing m here
                self._data[self._idx(n, n, m)] = (
                    self._data[self._idx(n - 1, n - 1, m - 1)] * 0.5 * (1 + cosine_colatitude) * ((2 * n - 1) * 2 * n)**0.5
                ) / ((n + m - 1) * (n + m))**0.5
        return self

    def apply(self, expansion, inverse=False, out=None):
        N = min(self.order, expansion.order)  # Allows the rotation coefficients to be used for lower order expansions
        if out is None:
            wavenumber = getattr(expansion, 'wavenumber', None)
            self_shape = self.shape
            expansion_shape = np.shape(expansion)
            output_shape = np.broadcast(np.empty(self_shape, dtype=[]), np.empty(expansion_shape, dtype=[])).shape
            if isinstance(expansion, expansions.Expansion):
                out = type(expansion)(order=N, shape=output_shape, wavenumber=wavenumber)
            else:
                out = expansions.Expansion(order=N, shape=output_shape, wavenumber=wavenumber)
        elif expansion is out:
            raise NotImplementedError('Rotations cannot currently be applied in place')
        if inverse:
            for n in range(N + 1):
                for p in range(-n, n + 1):
                    # Use a local temporary variable to accumulate the summed value.
                    # We might not always be able to add in place to the components of the output.
                    value = 0
                    for m in range(-n, n + 1):
                        value += expansion[n, m] * self[n, p, m]
                    out[n, p] = value
        else:
            for n in range(N + 1):
                for m in range(-n, n + 1):
                    value = 0
                    for p in range(-n, n + 1):
                        # Conjugate will not do anything for colatitude rotations,
                        # but full rotations use this method as well.
                        # The method version of conjugate (as opposed to np.conjugate)
                        # will not create copies of real arrays.
                        # value += expansion[n, p] * self[n, p, m].conjugate()

                        # Alternate version doing the same thing but exploiting
                        # symmetries instead of actuallly conjugating.
                        # This seems to be a little bit faster.
                        value += expansion[n, p] * (-1)**(m + p) * self[n, -p, -m]
                    out[n, m] = value
        return out


class Rotation(ColatitudeRotation):
    # Subclass of ColatitudeRotation to get access to the `apply` method, which work the same for both types of rotation.
    def __init__(self, order, position=None, colatitude=None, azimuth=None, secondary_azimuth=None, new_z_axis=None, old_z_axis=None, defer_evaluation=False):
        coordinate = coordinates.Rotation.parse_args(
            position=position, new_z_axis=new_z_axis, old_z_axis=old_z_axis,
            colatitude=colatitude, azimuth=azimuth, secondary_azimuth=secondary_azimuth)
        super().__init__(order=order, position=coordinate, defer_evaluation=defer_evaluation)

    def evaluate(self, position=None, colatitude=None, azimuth=None, secondary_azimuth=None, new_z_axis=None, old_z_axis=None):
        if (position is None) and (new_z_axis is None) and (old_z_axis is None) and (colatitude is None):
            # Allows chaing the azimuth angles without reevaluating the colatitude rotation.
            # This might be useful sometimes since changing the azimuth angles is much cheaper than
            # changing the colatitude angle. If we anyhow change the colatitude roration, reevaluating the
            # azimuth rotations will be very cheap so we won't bother with checking if we could skip it.
            if azimuth is not None:
                self._primary_phase = np.asarray(np.exp(1j * azimuth))
            if secondary_azimuth is not None:
                self._secondary_phase = np.asarray(np.exp(1j * secondary_azimuth))
        else:
            self.coordinate = coordinates.Rotation.parse_args(
                position=position, new_z_axis=new_z_axis, old_z_axis=old_z_axis,
                colatitude=colatitude, azimuth=azimuth, secondary_azimuth=secondary_azimuth)
            super().evaluate(position=self.coordinate)
            self._primary_phase = np.asarray(np.exp(1j * self.coordinate.azimuth))
            self._secondary_phase = np.asarray(np.exp(1j * self.coordinate.secondary_azimuth))
        return self

    @property
    def shape(self):
        return self.coordinate.shape

    def copy(self, deep=False):
        new_obj = super().copy(deep=deep)
        if deep:
            new_obj._primary_phase = self._primary_phase.copy()
            new_obj._secondary_phase = self._secondary_phase.copy()
        else:
            new_obj._primary_phase = self._primary_phase
            new_obj._secondary_phase = self._secondary_phase
        return new_obj

    def __getitem__(self, key):
        n, p, m = key
        phase = self._primary_phase ** m * self._secondary_phase ** p
        return super().__getitem__(key) * phase
