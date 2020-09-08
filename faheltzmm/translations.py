import numpy as np
from . import generate, coordinates, rotations, expansions


class CoaxialTranslation:
    _default_output_type = expansions.Expansion

    def __init__(self, input_order, output_order, distance=None, wavenumber=None, shape=None):
        self._input_order = input_order
        self._output_order = output_order
        self._min_order = min(self.input_order, self.output_order)
        self._max_order = max(self.input_order, self.output_order)
        self._shape = np.broadcast(distance, wavenumber).shape if shape is None else shape
        num_unique = (
            (self._min_order + 1)**2 * (self._max_order + 1)
            - (self._min_order * (self._min_order + 1)) // 2 * (self._min_order + self._max_order + 2)
            + (self._min_order * (self._min_order - 1) * (self._min_order + 1)) // 6
        )
        self._data = np.zeros((num_unique,) + self._shape, self._dtype)

        if distance is not None:
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
        return self._shape

    @property
    def _coefficient_indices(self):
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

    def evaluate(self, distance, wavenumber):
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
        m_buffer = np.zeros((N + 1,) + self.shape, dtype=self._dtype)
        m_minus_one_buffer = np.zeros((N + 1,) + self.shape, dtype=self._dtype)
        n_minus_two_buffer = np.zeros((N + 1,) + self.shape, dtype=self._dtype)
        n_minus_one_buffer = np.zeros((N + 1,) + self.shape, dtype=self._dtype)

        for m in range(N + 1):  # Main loop over modes
            # Buffer swap for sectorials.
            m_buffer, m_minus_one_buffer = m_minus_one_buffer, m_buffer
            if m == 0:  # Get starting values: (0, p, 0)
                # Somewhat ugly with this if-statement inside the loop,
                # but the alternative is to duplicate everything else in the loop
                # for m=0.
                initial_values = self._recurrence_initialization(order=N + P, x=distance * wavenumber)
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
        if out is None:
            shape = np.broadcast(self[0, 0, 0], expansion[0, 0]).shape
            out = self._default_output_type(order=self.input_order if inverse else self.output_order, wavenumber=expansion.wavenumber, shape=shape)
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
    from .bases import SphericalBessel as _recurrence_initialization


class ExteriorCoaxialTranslation(CoaxialTranslation):
    _dtype = float
    from .bases import SphericalBessel as _recurrence_initialization


class ExteriorInteriorCoaxialTranslation(CoaxialTranslation):
    _dtype = complex
    from .bases import SphericalHankel as _recurrence_initialization


class Translation:
    def __init__(self, input_order, output_order, position=None, wavenumber=None, shape=None):
        self._coaxial = self._coaxial_cls(
            input_order=input_order, output_order=output_order,
            shape=shape if shape is not None else np.broadcast(position[0], wavenumber).shape
        )
        self._rotation = rotations.Rotation(
            order=max(input_order, output_order),
            shape=shape if shape is not None else np.shape(position[0])
        )

        if position is not None:
            self.evaluate(position, wavenumber)

    def evaluate(self, position, wavenumber):
        r, colatitude, azimuth = coordinates.cartesian_2_spherical(position)
        self._coaxial.evaluate(distance=r, wavenumber=wavenumber)
        self._rotation.evaluate(colatitude=colatitude, primary_azimuth=azimuth)
        return self

    def apply(self, expansion, inverse=False):
        if not inverse:
            return expansion.apply(self._rotation).apply(self._coaxial).apply(self._rotation, inverse=True)
        else:
            raise NotImplementedError('Inverse translations not currently implemented.')


class InteriorTranslation(Translation):
    _coaxial_cls = InteriorCoaxialTranslation


class ExteriorTranslation(Translation):
    _coaxial_cls = ExteriorCoaxialTranslation


class ExteriorInteriorTranslation(Translation):
    _coaxial_cls = ExteriorInteriorCoaxialTranslation


def translate(field_coefficients, position, wavenumber, input_domain, output_domain, max_output_order=None):
    # TODO: Merge this with the translation function.
    t, beta, alpha = coordinates.cartesian_2_spherical(position)
    max_input_order = field_coefficients.shape[0] - 1
    max_output_order = max_input_order if max_output_order is None else max_output_order
    rotation_coefficients = rotations.rotation_coefficients(max_order=max(max_input_order, max_output_order), primary_azimuth=alpha, colatitude=beta)
    translation_coefficients = coaxial_translation_coefficients(
        max_input_order=max_input_order, max_output_order=max_output_order,
        distance=t, wavenumber=wavenumber,
        input_domain=input_domain, output_domain=output_domain)
    return translation(field_coefficients, translation_coefficients, rotation_coefficients)


def translation(field_coefficients, translation_coefficients, rotation_coefficients):
    max_input_order = translation_coefficients.shape[1] - 1
    max_output_order = translation_coefficients.shape[2] - 1
    max_modes = (translation_coefficients.shape[0] - 1) // 2
    if max_input_order != max_output_order:
        raise NotImplementedError('Changing the order during translation not yet implemented')
    if max_modes != max_input_order:
        raise NotImplementedError('Mode-limited translations not yet supported')

    if max_input_order != field_coefficients.shape[0] - 1:
        raise ValueError('Translation coefficients and field coefficients does not have the same maximum order')
    if 2 * max_modes + 1 != field_coefficients.shape[1]:
        raise ValueError('Translation coefficients and field coefficients does not have the same maximum modes')
    if max_input_order != rotation_coefficients.shape[1] - 1:
        raise ValueError('Rotation coefficients and field coefficients does not have the same maximum order')
    if 2 * max_modes + 1 != rotation_coefficients.shape[0] or rotation_coefficients.shape[0] != rotation_coefficients.shape[2]:
        raise ValueError('Rotation coefficients does not have the correct number of modes')

    field_coefficients = rotations.rotate(field_coefficients=field_coefficients, rotation_coefficients=rotation_coefficients, inverse=False)
    field_coefficients = np.einsum('nm..., mnp... -> pm...', field_coefficients, translation_coefficients)
    field_coefficients = rotations.rotate(field_coefficients=field_coefficients, rotation_coefficients=rotation_coefficients, inverse=True)
    return field_coefficients


def coaxial_translation(field_coefficients, translation_coefficients, inverse=False):
    # TODO: Add checks for size conformity here
    # TODO: Do a similar trick with distance/wavenumber as for the rotations to allow a single use input form.
    # TODO: Allow inverse coaxial translation. This is particularly useful for multiple scattering problems.
    if inverse:
        return np.einsum('nm..., mpn -> pm', field_coefficients, translation_coefficients)
    else:
        return np.einsum('nm..., mnp -> pm', field_coefficients, translation_coefficients)


# TODO: Add some kind of function to calculate both the roation coefficients and the translation coefficients in one go. Preferably in a form which can exploit
def coaxial_translation_coefficients(max_input_order, max_output_order, distance, wavenumber, input_domain, output_domain, max_mode=None):
    if 'singular' in input_domain.lower() or 'exterior' in input_domain.lower() or 'external' in input_domain.lower():
        input_domain = 'singular'
    elif 'regular' in input_domain.lower() or 'interior' in input_domain.lower() or 'internal' in input_domain.lower():
        input_domain = 'regular'
    else:
        raise ValueError(f'Unknown domain for input coefficients {input_domain}')

    if 'singular' in output_domain.lower() or 'exterior' in output_domain.lower() or 'external' in output_domain.lower():
        output_domain = 'singular'
    elif 'regular' in output_domain.lower() or 'interior' in output_domain.lower() or 'internal' in output_domain.lower():
        output_domain = 'regular'
    else:
        raise ValueError(f'Unknown domain for output coefficients {output_domain}')

    kt = distance * wavenumber
    max_mode = max_input_order if max_mode is None else max_mode
    coefficients_shape = (2 * max_mode + 1, max_input_order + 1, max_output_order + max_input_order + 1) + np.shape(kt)

    all_n = np.arange(max_output_order + max_input_order + 1).reshape([-1] + [1] * np.ndim(kt))
    if input_domain == 'singular' and output_domain == 'regular':
        coefficients = np.zeros(coefficients_shape, dtype=complex)
        coefficients[0, 0] = (2 * all_n + 1)**0.5 * generate.spherical_hn_all(max_input_order + max_output_order, kt)
    elif input_domain == output_domain:
        coefficients = np.zeros(coefficients_shape, dtype=float)
        coefficients[0, 0] = (2 * all_n + 1)**0.5 * generate.spherical_jn_all(max_input_order + max_output_order, kt)
    else:
        raise NotImplementedError(f'Translations from {input_domain} domain to {output_domain} domain not implemented')

    def sectorial_recurrence(m, p):
        return (
            coefficients[m - 1, m - 1, p - 1] * ((p + m - 1) * (p + m) * (2 * m + 1) / ((2 * p - 1) * (2 * p + 1) * (2 * m)))**0.5
            + coefficients[m - 1, m - 1, p + 1] * ((p - m + 1) * (p - m + 2) * (2 * m + 1) / ((2 * p + 1) * (2 * p + 3) * (2 * m)))**0.5
        )

    def recurrence(m, n, p):
        value = -coefficients[m, n - 1, p + 1] * ((p + m + 1) * (p - m + 1) / ((2 * p + 1) * (2 * p + 3)))**0.5
        if p > 0:
            value += coefficients[m, n - 1, p - 1] * ((p + m) * (p - m) / ((2 * p - 1) * (2 * p + 1)))**0.5
        if n > 1:
            value += coefficients[m, n - 2, p] * ((n + m - 1) * (n - m - 1) / ((2 * n - 3) * (2 * n - 1)))**0.5
        return value * ((2 * n - 1) * (2 * n + 1) / ((n + m) * (n - m)))**0.5

    # Using the (m, n, p) -> (m, p, n) symmetry to get (0, n, 0)
    coefficients[0, :, 0] = coefficients[0, 0, :max_input_order + 1] * (-1) ** np.arange(max_input_order + 1).reshape([-1] + [1] * np.ndim(kt))
    # Recurrence to fill m=0 layer, the same as in the loop below exept for the
    # sectorial values.
    for n in range(1, max_input_order + 1):
        coefficients[0, n, n] = recurrence(0, n, n)
        for p in range(n + 1, max_input_order + 1):
            coefficients[0, n, p] = recurrence(0, n, p)
            coefficients[0, p, n] = coefficients[0, n, p] * (-1)**(n + p)
        for p in range(max_input_order + 1, max_output_order + max_input_order - n + 1):
            coefficients[0, n, p] = recurrence(0, n, p)

    for m in range(1, max_mode + 1):
        coefficients[m, m, m] = sectorial_recurrence(m, m)
        coefficients[-m, m, m] = coefficients[m, m, m]
        for p in range(m + 1, max_input_order + 1):
            coefficients[m, m, p] = sectorial_recurrence(m, p)
            coefficients[-m, m, p] = coefficients[m, m, p]
            coefficients[m, p, m] = coefficients[m, m, p] * (-1) ** (m + p)
            coefficients[-m, p, m] = coefficients[m, p, m]
        for p in range(max_input_order + 1, max_input_order + max_output_order - m + 1):
            coefficients[m, m, p] = sectorial_recurrence(m, p)
            coefficients[-m, m, p] = coefficients[m, m, p]

        for n in range(m + 1, max_input_order + 1):
            coefficients[m, n, n] = recurrence(m, n, n)
            coefficients[-m, n, n] = coefficients[m, n, n]
            for p in range(n + 1, max_input_order + 1):
                coefficients[m, n, p] = recurrence(m, n, p)
                coefficients[-m, n, p] = coefficients[m, n, p]
                coefficients[m, p, n] = coefficients[m, n, p] * (-1) ** (n + p)
                coefficients[-m, p, n] = coefficients[m, p, n]
            for p in range(max_input_order + 1, max_output_order + max_input_order - n + 1):
                coefficients[m, n, p] = recurrence(m, n, p)
                coefficients[-m, n, p] = coefficients[m, n, p]
    return coefficients[:, :, :max_output_order + 1]
