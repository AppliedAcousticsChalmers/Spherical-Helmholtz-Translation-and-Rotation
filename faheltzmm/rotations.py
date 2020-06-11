import numpy as np
from . import generate, indexing, coordinates


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
    if callable(rotation_coefficients):
        orders = field_coefficients.shape[0] - 1
        modes = (field_coefficients.shape[1] - 1) // 2
        rotation_coefficients = rotation_coefficients(max_order=orders, max_mode=modes, **kwargs)
    if inverse:
        return np.einsum('pnm, nm... -> np', rotation_coefficients.conj(), field_coefficients)
    else:
        return np.einsum('mnp, nm... -> np', rotation_coefficients, field_coefficients)


def colatitude_rotation_coefficients(max_order, colatitude=None, max_mode=None, cosine_colatitude=None):
    cosine_colatitude = np.asarray(cosine_colatitude) if cosine_colatitude is not None else np.cos(colatitude)
    sine_colatitude = (1 - cosine_colatitude**2)**0.5

    max_mode = max_order if max_mode is None else min(max_mode, max_order)
    coefficients = np.zeros((2 * max_mode + 1, max_order + max_mode + 1, 2 * (max_order + max_mode) + 1), dtype=float)
    coefficients[0] = zonal_colatitude_rotation_coefficients(max_order=max_order + max_mode, cosine_colatitude=cosine_colatitude)

    # TODO: It is certainly possible to use clever indexing to get rid of the two inner for loops.
    # If this is a significant contributor to calculation times, that might be a reasonable solution.
    # Another solution might be to use cython to make the loops run faster.
    def recurrence(m, n, mp):
        coefficients[m, n, mp] = (
            0.5 * (1 + cosine_colatitude) * np.sqrt((n + mp - 1) * (n + mp)) * coefficients[m - 1, n - 1, mp - 1]
            + 0.5 * (1 - cosine_colatitude) * np.sqrt((n - mp - 1) * (n - mp)) * coefficients[m - 1, n - 1, mp + 1]
            - sine_colatitude * np.sqrt((n + mp) * (n - mp)) * coefficients[m - 1, n - 1, mp]
        ) / np.sqrt((n + m - 1) * (n + m))

    for m in range(1, max_mode + 1):
        p = max_order + max_mode - m
        for n in range(m, p + 1):
            coefficients[m, n, 0] = coefficients[0, n, m] * (-1) ** m
            coefficients[-m, n, 0] = coefficients[0, n, m]
            recurrence(m, n, m)
            coefficients[-m, n, -m] = coefficients[m, n, m]
            recurrence(m, n, -m)
            coefficients[-m, n, m] = coefficients[m, n, -m]
            for mp in range(m + 1, min(n, max_mode) + 1):
                recurrence(m, n, mp)
                coefficients[-m, n, -mp] = (-1)**(m + mp) * coefficients[m, n, mp]
                coefficients[mp, n, m] = (-1)**(m + mp) * coefficients[m, n, mp]
                coefficients[-mp, n, -m] = coefficients[m, n, mp]

                recurrence(m, n, -mp)
                coefficients[-m, n, mp] = (-1)**(m + mp) * coefficients[m, n, -mp]
                coefficients[-mp, n, m] = (-1)**(m + mp) * coefficients[m, n, -mp]
                coefficients[mp, n, -m] = coefficients[m, n, -mp]

            for mp in range(max_mode + 1, n + 1):
                recurrence(m, n, mp)
                coefficients[-m, n, -mp] = (-1)**(m + mp) * coefficients[m, n, mp]

                recurrence(m, n, -mp)
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
