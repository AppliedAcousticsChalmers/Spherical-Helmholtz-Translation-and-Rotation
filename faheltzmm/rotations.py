import numpy as np
from . import generate, indexing, coordinates, bases, expansions


class ColatitudeRotation:
    def __init__(self, order, colatitude=None, shape=None, **kwargs):
        self._order = order
        self._shape = shape if shape is not None else np.shape(colatitude)
        num_unique = (self.order + 1) * (self.order + 2) * (2 * self.order + 3) // 6
        self._data = np.zeros((num_unique,) + self._shape, dtype=float)
        if colatitude is not None:
            # kwargs used to pass azimuth angles from `Rotation._init__` to `Rotation.evaluate`
            self.evaluate(colatitude=colatitude, **kwargs)

    @property
    def order(self):
        return self._order

    @property
    def shape(self):
        return self._shape

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
    def _coefficient_indices(self):
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

    def evaluate(self, colatitude=None):
        cosine_colatitude = np.cos(colatitude)
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
        if out is None:
            shape = np.broadcast(self[0, 0, 0], expansion[0, 0]).shape
            if isinstance(expansion, expansions.Expansion):
                out = type(expansion)(order=self.order, shape=shape, wavenumber=expansion.wavenumber)
            else:
                out = expansions.Expansion(order=self.order, shape=shape)
        elif expansion is out:
            raise NotImplementedError('Rotations cannot currently be applied in place')
        N = min(self.order, expansion.order)  # Allows the rotation coefficients to be used for lower order expansions
        if not inverse:
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
    def __init__(self, order, colatitude=None, primary_azimuth=None, secondary_azimuth=None, new_z_axis=None, old_z_axis=None, shape=None):
        if new_z_axis is not None or old_z_axis is not None:
            colatitude, primary_azimuth, secondary_azimuth = coordinates.z_axes_rotation_angles(new_axis=new_z_axis, old_axis=old_z_axis)
        # CoalatitudeRotation will pass the azimuth angles through to evaluate.
        super().__init__(order=order, colatitude=colatitude, shape=shape, primary_azimuth=primary_azimuth, secondary_azimuth=secondary_azimuth)
        self._shape = np.broadcast(self._shape, primary_azimuth, secondary_azimuth)  # Override the shape set by the ColatitudeRotation

    def evaluate(self, colatitude=None, primary_azimuth=0, secondary_azimuth=0, new_z_axis=None, old_z_axis=None, **kwargs):
        if new_z_axis is not None or old_z_axis is not None:
            colatitude, primary_azimuth, secondary_azimuth = coordinates.z_axes_rotation_angles(new_axis=new_z_axis, old_axis=old_z_axis)
        if colatitude is not None:
            # Allows changing the azimuth angles without changing the colatitude.
            # Useful since changing the azimuth angles is cheap, whilg chanigng
            # the colatitude is expensive.
            super().evaluate(colatitude=colatitude, **kwargs)
        self._primary_phase = np.exp(1j * primary_azimuth)
        self._secondary_phase = np.exp(1j * secondary_azimuth)
        return self

    @property
    def shape(self):
        return np.broadcast(self._data[0], self._primary_phase, self._secondary_phase).shape

    def __getitem__(self, key):
        n, p, m = key
        phase = self._primary_phase ** m * self._secondary_phase ** p
        return super().__getitem__(key) * phase


def rotation_coefficients(max_order, colatitude=0, primary_azimuth=0, secondary_azimuth=0, max_mode=None, new_z_axis=None, old_z_axis=None):
    if new_z_axis is not None:
        beta, alpha, mu = coordinates.z_axes_rotation_angles(new_axis=new_z_axis, old_axis=old_z_axis)
        return rotation_coefficients(max_order=max_order, colatitude=beta, primary_azimuth=alpha, secondary_azimuth=mu)
    max_mode = max_order if max_mode is None else min(max_mode, max_order)
    angles = np.broadcast(colatitude, primary_azimuth, secondary_azimuth)
    n, mp = indexing.expansions(max_order, 'natural', 'natural')
    m = mp if max_mode is None else indexing.expansions(min(max_mode, max_order), 'natural', 'natural')[1]
    colatitude_rotation = colatitude_rotation_coefficients(max_order=max_order, colatitude=colatitude, max_mode=max_mode)
    primary_azimuth_rotation = np.exp(1j * primary_azimuth * m.reshape([-1, 1, 1] + [1] * angles.ndim))
    secondary_azimuth_rotation = np.exp(1j * secondary_azimuth * mp.reshape([-1] + [1] * angles.ndim))
    return primary_azimuth_rotation * colatitude_rotation * secondary_azimuth_rotation


def rotate(field_coefficients, rotation_coefficients=rotation_coefficients, inverse=False, **kwargs):
    # TODO: Rename to rotation!
    # It's not really feasible to the the active form for the translations unless we accept "translate_coaxially",
    # so it might be better to use a passive form for all operations.
    if callable(rotation_coefficients):
        orders = field_coefficients.shape[0] - 1
        modes = (field_coefficients.shape[1] - 1) // 2
        rotation_coefficients = rotation_coefficients(max_order=orders, max_mode=modes, **kwargs)
    if inverse:
        return np.einsum('pnm..., nm... -> np...', rotation_coefficients.conj(), field_coefficients)
    else:
        return np.einsum('mnp..., nm... -> np...', rotation_coefficients, field_coefficients)


def colatitude_rotation_coefficients(max_order, colatitude=None, max_mode=None, cosine_colatitude=None):
    cosine_colatitude = np.asarray(cosine_colatitude) if cosine_colatitude is not None else np.cos(colatitude)
    sine_colatitude = (1 - cosine_colatitude**2)**0.5

    max_mode = max_order if max_mode is None else min(max_mode, max_order)
    coefficients = np.zeros((2 * max_mode + 1, max_order + max_mode + 1, 2 * (max_order + max_mode) + 1) + cosine_colatitude.shape, dtype=float)
    coefficients[0] = zonal_colatitude_rotation_coefficients(max_order=max_order + max_mode, cosine_colatitude=cosine_colatitude)

    # TODO: It is certainly possible to use clever indexing to get rid of the two inner for loops.
    # If this is a significant contributor to calculation times, that might be a reasonable solution.
    # Another solution might be to use cython to make the loops run faster.

    def recurrence(m, n, mp):
        return (
            0.5 * (1 + cosine_colatitude) * np.sqrt((n + mp - 1) * (n + mp)) * coefficients[m - 1, n - 1, mp - 1]
            + 0.5 * (1 - cosine_colatitude) * np.sqrt((n - mp - 1) * (n - mp)) * coefficients[m - 1, n - 1, mp + 1]
            - sine_colatitude * np.sqrt((n + mp) * (n - mp)) * coefficients[m - 1, n - 1, mp]
        ) / np.sqrt((n + m - 1) * (n + m))

    for m in range(1, max_mode + 1):
        p = max_order + max_mode - m
        for n in range(m, p + 1):
            coefficients[m, n, 0] = coefficients[0, n, m] * (-1) ** m
            coefficients[-m, n, 0] = coefficients[0, n, m]
            coefficients[m, n, m] = recurrence(m, n, m)
            coefficients[-m, n, -m] = coefficients[m, n, m]
            coefficients[m, n, -m] = recurrence(m, n, -m)
            coefficients[-m, n, m] = coefficients[m, n, -m]
            for mp in range(m + 1, min(n, max_mode) + 1):
                coefficients[m, n, mp] = recurrence(m, n, mp)
                coefficients[-m, n, -mp] = (-1)**(m + mp) * coefficients[m, n, mp]
                coefficients[mp, n, m] = (-1)**(m + mp) * coefficients[m, n, mp]
                coefficients[-mp, n, -m] = coefficients[m, n, mp]

                coefficients[m, n, -mp] = recurrence(m, n, -mp)
                coefficients[-m, n, mp] = (-1)**(m + mp) * coefficients[m, n, -mp]
                coefficients[-mp, n, m] = (-1)**(m + mp) * coefficients[m, n, -mp]
                coefficients[mp, n, -m] = coefficients[m, n, -mp]

            for mp in range(max_mode + 1, n + 1):
                coefficients[m, n, mp] = recurrence(m, n, mp)
                coefficients[-m, n, -mp] = (-1)**(m + mp) * coefficients[m, n, mp]

                coefficients[m, n, -mp] = recurrence(m, n, -mp)
                coefficients[-m, n, mp] = (-1)**(m + mp) * coefficients[m, n, -mp]
    return np.delete(coefficients[:, :max_order + 1], np.arange(max_order + 1, max_order + 1 + 2 * max_mode), axis=2)


def zonal_colatitude_rotation_coefficients(max_order, colatitude=None, cosine_colatitude=None):
    cosine_colatitude = cosine_colatitude if cosine_colatitude is not None else np.cos(colatitude)
    coeffs = np.zeros((max_order + 1, 2 * max_order + 1) + cosine_colatitude.shape, dtype=float)
    coeffs[:, :max_order + 1] = generate.legendre_all(max_order=max_order, x=cosine_colatitude, normalization='orthonormal')

    n = np.arange(max_order + 1).reshape([-1, 1] + [1] * np.ndim(cosine_colatitude))
    m = np.arange(max_order + 1).reshape([-1] + [1] * np.ndim(cosine_colatitude))
    n_positive, m_positive = indexing.expansions(max_order, 'natural', 'positive')
    n_negative, m_negative = indexing.expansions(max_order, 'natural', 'negative')

    coeffs[:, :max_order + 1] *= np.sqrt(2 / (2 * n + 1))
    # We would need to include  (-1)^m in the (m) -> (-m) symmetry,
    # but since we also need a (-1)^m factor in the calculation of the positive
    # components they cancel and we only apply them to the positive values.
    coeffs[n_negative, m_negative] = coeffs[n_positive, m_positive]
    coeffs[:, :max_order + 1] *= (-1) ** m
    return coeffs
