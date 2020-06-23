import numpy as np
from . import generate


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
    coefficients_shape = [2 * max_mode + 1, max_input_order + 1, max_output_order + max_input_order + 1] + [1] * np.ndim(kt)

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
        coefficients[m, m, p] = (
            coefficients[m - 1, m - 1, p - 1] * ((p + m - 1) * (p + m) * (2 * m + 1) / ((2 * p - 1) * (2 * p + 1) * (2 * m)))**0.5
            + coefficients[m - 1, m - 1, p + 1] * ((p - m + 1) * (p - m + 2) * (2 * m + 1) / ((2 * p + 1) * (2 * p + 3) * (2 * m)))**0.5
        )

    def recurrence(m, n, p):
        coefficients[m, n, p] = -coefficients[m, n - 1, p + 1] * ((p + m + 1) * (p - m + 1) / ((2 * p + 1) * (2 * p + 3)))**0.5
        if p > 0:
            coefficients[m, n, p] += coefficients[m, n - 1, p - 1] * ((p + m) * (p - m) / ((2 * p - 1) * (2 * p + 1)))**0.5
        if n > 1:
            coefficients[m, n, p] += coefficients[m, n - 2, p] * ((n + m - 1) * (n - m - 1) / ((2 * n - 3) * (2 * n - 1)))**0.5
        coefficients[m, n, p] *= ((2 * n - 1) * (2 * n + 1) / ((n + m) * (n - m)))**0.5

    # Recurrence to fill m=0 layer, the same as in the loop below exept for the
    # sectorial values.
    for n in range(1, max_input_order + 1):
        for p in range(max_output_order + max_input_order - n + 1):
            recurrence(0, n, p)

    for m in range(1, max_mode + 1):
        for p in range(m, max_input_order + max_output_order - m + 1):
            sectorial_recurrence(m, p)
            coefficients[-m, m, p] = coefficients[m, m, p]

        for n in range(m + 1, max_input_order + 1):
            for p in range(m, max_output_order + max_input_order - n + 1):
                recurrence(m, n, p)
                coefficients[-m, n, p] = coefficients[m, n, p]

    return coefficients[:, :, :max_output_order + 1]
