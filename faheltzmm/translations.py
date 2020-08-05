import numpy as np
from . import generate, coordinates, rotations


class CoaxialTranslation:
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
            if input_order > self.input_order:
                raise IndexError(f'Input order {input_order} is out of bounds for {self.__class__.__name__} with max input order {self.input_order}')
            if output_order > self.output_order:
                raise IndexError(f'Input order {output_order} is out of bounds for {self.__class__.__name__} with max output order {self.output_order}')
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


class InteriorCoaxialTranslation(CoaxialTranslation):
    _dtype = float


class ExteriorCoaxialTranslation(CoaxialTranslation):
    _dtype = float


class ExteriorInteriorCoaxialTranslation(CoaxialTranslation):
    _dtype = complex


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
