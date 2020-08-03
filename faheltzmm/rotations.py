import numpy as np
from . import generate, indexing, coordinates, bases


class ColatitudeRotation:
    def __init__(self, order, colatitude=None, cosine_colatitude=None, sine_colatitude=None, shape=None):
        self._order = order
        self._shape = shape if shape is not None else np.shape(cosine_colatitude if cosine_colatitude is not None else colatitude)
        self._data = np.zeros((self.num_unique,) + self.shape, dtype=float)
        if colatitude is not None or cosine_colatitude is not None:
            self.evaluate(colatitude=colatitude, cosine_colatitude=cosine_colatitude, sine_colatitude=sine_colatitude)

        # DEBUG: Storing the coefficient indices in a flat list to make sure that the indexing is working properly.
        self._indces = []
        for n in range(self.order + 1):
            for p in range(-self.order, self.order + 1):
                for m in range(-self.order, self.order + 1):
                    if abs(p) > n or abs(m) > n:
                        continue  # Is zero by definition, don't store
                    if p < 0:
                        continue  # We're storing positive values of p
                    if abs(m) > p:
                        continue  # Far away from the starting values
                    self._indces.append((n, p, m))

    def _idx(self, order, mode_out, mode_in):
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
            # Safe guard against stupidity
            sign = 1
        return sign * self._data[self._idx(n, p, m)]

        # DEBUG: The rest of this function is debugging code.
        symm = ''
        if abs(p) < m:
            p, m = m, p
            symm += 'swap'
        if abs(m) <= -p > 0:
            p, m = -p, -m
            symm += 'negate'
        if abs(p) < -m:
            p, m, = -m, -p
            symm += 'negate & swap'

        value = self._data[self._idx(n, p, m)]
        return value, symm

    def evaluate(self, colatitude=None, cosine_colatitude=None, sine_colatitude=None):
        cosine_colatitude = np.cos(colatitude) if cosine_colatitude is None else np.asarray(cosine_colatitude)
        sine_colatitude = (1 - cosine_colatitude**2)**0.5 if sine_colatitude is None else np.asarray(sine_colatitude)

        legendre = bases.AssociatedLegendrePolynomials(order=self.order, x=cosine_colatitude)

        # n=0 and n=1 will be special cases since the p=n-1 and p=n special cases will not behave
        self._data[self._idx(0, 0, 0)] = 1
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

    @property
    def order(self):
        return self._order

    @property
    def num_unique(self):
        return (self.order + 1) * (self.order + 2) * (2 * self.order + 3) // 6

    @property
    def shape(self):
        return self._shape



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
