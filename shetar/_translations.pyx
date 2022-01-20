import numpy as np
import cython
from cython.parallel import prange, parallel
from ._shapes import prepare_strides, broadcast_shapes
from ._shapes cimport broadcast_index

from scipy.special import spherical_jn, spherical_yn


@cython.cdivision(True)
cdef inline Py_ssize_t coaxial_translation_mode_index(Py_ssize_t mode, Py_ssize_t min_order, Py_ssize_t max_order) nogil:
    return (
        mode * (min_order + 1) * (max_order + 1)
        - mode * (min_order * (min_order + 1)) // 2
        - max_order * (mode * (mode + 1)) // 2
        + (mode * (mode - 1) * (mode - 2)) // 6
        - mode
    )


@cython.cdivision(True)
cdef inline Py_ssize_t coaxial_translation_order_index(Py_ssize_t input_order, Py_ssize_t max_order) nogil:
    return input_order * max_order - (input_order * (input_order - 1)) // 2


cdef inline Py_ssize_t coaxial_translation_index(Py_ssize_t input_order, Py_ssize_t output_order, Py_ssize_t mode, Py_ssize_t min_order, Py_ssize_t max_order) nogil:
    return coaxial_translation_mode_index(mode, min_order, max_order) + coaxial_translation_order_index(input_order, max_order) + output_order


def coaxial_order_to_unique(input_order, output_order):
    min_order = min(input_order, output_order)
    max_order = max(input_order, output_order)
    num_unique = (
        (min_order + 1)**2 * (max_order + 1)
        - (min_order * (min_order + 1)) // 2 * (min_order + max_order + 2)
        + (min_order * (min_order - 1) * (min_order + 1)) // 6
    )
    return num_unique



def coaxial_translation_coefficients(distance, input_order, output_order, cross_domain=False, out=None):
    output_shape = np.shape(distance)

    num_unique = coaxial_order_to_unique(input_order, output_order)
    if out is None:
        out = np.zeros(output_shape + (num_unique,), dtype=complex if cross_domain else float)
    else:
        if out.shape[:-1] != output_shape:
            raise ValueError(f'Cannot use pre-allocated output of shape {out.shape} for coaxial translation coefficients with distance shape {output_shape}')
        if out.shape[-1] != num_unique:
            raise ValueError(f'Cannot use pre-allocated output of shape {out.shape} for coaxial translation coefficients of {input_order = } and {output_order = }, requiring {num_unique} unique values')

    cdef:
        int num_elements = np.size(distance)
        int buffer_shape = min(input_order, output_order) + 1

    initialization = spherical_jn(np.arange(input_order + output_order + 1), distance.reshape((-1, 1)))
    if cross_domain:
        initialization = initialization + 1j * spherical_yn(np.arange(input_order + output_order + 1), distance.reshape((-1, 1)))

    cdef:
        double[:, :] out_cy = out.reshape((-1, num_unique))
        double[:, :] initialization_cy = initialization    
        Py_ssize_t min_order = min(input_order, output_order)
        Py_ssize_t max_order = max(input_order, output_order)
        Py_ssize_t idx_elem

    if num_elements == 1:
        coaxial_translation_coefficients_calculation(
            out_cy, initialization_cy, 0,
            min_order, max_order,
            np.zeros(buffer_shape), np.zeros(buffer_shape),
            np.zeros(buffer_shape), np.zeros(buffer_shape),
        )
        return out


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void coaxial_translation_coefficients_calculation(
    double[:, :] output,
    double[:, :] initialization,
    Py_ssize_t idx_elem,
    Py_ssize_t N,
    Py_ssize_t P,
    double[:] m_buffer,
    double[:] m_minus_one_buffer,
    double[:] n_minus_one_buffer,
    double[:] n_minus_two_buffer,
):

    cdef:
        Py_ssize_t n, p, m
        Py_ssize_t m_index = 0, nm_index, nm_minus_index, nm_minus_two_index
    # Indices in comments indicate [n, p, m] i.e. [input_order, output_order, mode]
    for m in range(N + 1):  # Main loop over modes
        # Buffer swap for sectorials.
        m_buffer, m_minus_one_buffer = m_minus_one_buffer, m_buffer
        if m == 0:  # Get starting values: [0, p, 0]
            # Somewhat ugly with this if-statement inside the loop,
            # but the alternative is to duplicate everything else in the loop for m=0.
            # We cannot do the sectorial loop first and then loop again for non-sectorials,
            # since the n_minus_one_buffer is filled in the sectorial loop and then forgotten
            for p in range(P):
                output[idx_elem, p] = initialization[idx_elem, p] * (2 * p + 1)**0.5
            output[idx_elem, P] = m_buffer[0] = n_minus_one_buffer[0] = initialization[idx_elem, P] * (2 * P + 1)**0.5
            for p in range(P + 1, N + P + 1):
                m_buffer[p - P]  = initialization[idx_elem, p] * (2 * p + 1)**0.5
                n_minus_one_buffer[p - P] = m_buffer[p - P]
        else:
            # Sectorial values [m, p, m]
            nm_minus_index = m_index + coaxial_translation_order_index(m - 1, P)  # m_index still on the value for m-1
            m_index = coaxial_translation_mode_index(m, N, P)   
            nm_index = m_index + coaxial_translation_order_index(m, P)
            
            for p in range(m, P):  # Sectorial recurrence in the stored range
                output[idx_elem, nm_index + p] = (  # Calculate [m, p, m]
                    (<double>((p + m - 1) * (p + m) * (2 * m + 1)) / <double>((2 * p - 1) * (2 * p + 1) * (2 * m)))**0.5 * output[idx_elem, nm_minus_index + p - 1]  # [m - 1, p - 1, m - 1]
                    + (<double>((p - m + 1) * (p - m + 2) * (2 * m + 1)) / <double>((2 * p + 1) * (2 * p + 3) * (2 * m)))**0.5 * output[idx_elem, nm_minus_index + p + 1]  # [m - 1, p + 1, m - 1]
                )
            output[idx_elem, nm_index + P] = m_buffer[0] = n_minus_one_buffer[0] = (  # Calculate [m, P, m]  
                (<double>((P + m - 1) * (P + m) * (2 * m + 1)) / <double>((2 * P - 1) * (2 * P + 1) * (2 * m)))**0.5 * output[idx_elem, nm_minus_index + P - 1] # [m - 1, P - 1, m - 1]
                + (<double>((P - m + 1) * (P - m + 2) * (2 * m + 1)) / <double>((2 * P + 1) * (2 * P + 3) * (2 * m)))**0.5 * m_minus_one_buffer[1] # [m - 1, P + 1, m - 1]
            )

            for p in range(P + 1, N + P - m + 1):  # Sectorial recurrence in the buffered range
                m_buffer[p - P] = n_minus_one_buffer[p - P] = (  # [m, p, m]
                    (<double>((p + m - 1) * (p + m) * (2 * m + 1)) / <double>((2 * p - 1) * (2 * p + 1) * (2 * m)))**0.5 * m_minus_one_buffer[p - P - 1]  # [m - 1, p - 1, m - 1]
                    + (<double>((p - m + 1) * (p - m + 2) * (2 * m + 1)) / <double>((2 * p + 1) * (2 * p + 3) * (2 * m)))**0.5 * m_minus_one_buffer[p - P + 1]  # [m - 1, p + 1, m - 1]
                )
        # Remaining (non-sectorial) values.
        # n = m + 1 is a special case since n-2 < m removes one component from the recurrence
        if m < N:  # Needed to prevent n = N + 1
            scale = (2 * m + 3)**0.5
            nm_index = m_index + coaxial_translation_order_index(m + 1, P)
            nm_minus_index = m_index + coaxial_translation_order_index(m, P)
            for p in range(m + 1, P):
                output[idx_elem, nm_index + p] = scale * (  # [m + 1, p, m]
                    (<double>((p + m) * (p - m)) / <double>((2 * p - 1) * (2 * p + 1)))**0.5 * output[idx_elem, nm_minus_index + p - 1] # [m, p - 1, m]
                    - (<double>((p + m + 1) * (p - m + 1)) / <double>((2 * p + 1) * (2 * p + 3)))**0.5 * output[idx_elem, nm_minus_index + p + 1]  # [m, p + 1, m]
                )
            output[idx_elem, nm_index + P] = n_minus_two_buffer[0] = scale * (  # [m + 1, P, m]
                (<double>((P + m) * (P - m)) / <double>((2 * P - 1) * (2 * P + 1)))**0.5 * output[idx_elem, nm_minus_index + P - 1]  # [m, P - 1, m]
                - (<double>((P + m + 1) * (P - m + 1)) / <double>((2 * P + 1) * (2 * P + 3)))**0.5 * n_minus_one_buffer[1] # [m, P + 1, m]
            )
            for p in range(P + 1, N + P - m):
                n_minus_two_buffer[p - P] = scale * (  # [m + 1, p, m]
                    n_minus_one_buffer[p - P - 1] * (<double>((p + m) * (p - m)) / <double>((2 * p - 1) * (2 * p + 1)))**0.5  # [m, p - 1, m]
                    - n_minus_one_buffer[p - P + 1] * (<double>((p + m + 1) * (p - m + 1)) / <double>((2 * p + 1) * (2 * p + 3)))**0.5  # [m, p + 1, m]
                )

        for n in range(m + 2, N + 1):  # Main loop over n.
            # Buffer swap for n.
            n_minus_one_buffer, n_minus_two_buffer = n_minus_two_buffer, n_minus_one_buffer
            # These index calculations could in principle be optimized to only calculate one of them and get the other two from previous values
            nm_index = m_index + coaxial_translation_order_index(n, P)
            nm_minus_index = m_index + coaxial_translation_order_index(n - 1, P)
            nm_minus_two_index = m_index + coaxial_translation_order_index(n - 2, P)
            scale = (<double>((2 * n - 1) * (2 * n + 1)) / <double>((n + m) * (n - m)))**0.5
            for p in range(n, P):  # Stored range
                output[idx_elem, nm_index + p] = scale * (  # [n, p, m]
                    (<double>((n + m - 1) * (n - m - 1)) / <double>((2 * n - 3) * (2 * n - 1)))**0.5 * output[idx_elem, nm_minus_two_index + p] # [n - 2, p, m]
                    + (<double>((p + m) * (p - m)) / <double>((2 * p - 1) * (2 * p + 1)))**0.5 * output[idx_elem, nm_minus_index + p - 1] # [n - 1, p - 1, m]
                    - (<double>((p + m + 1) * (p - m + 1)) / <double>((2 * p + 1) * (2 * p + 3)))**0.5 * output[idx_elem, nm_minus_index + p + 1]  # [n - 1, p + 1, m]
                )
            output[idx_elem, nm_index + P] = n_minus_two_buffer[0] = scale * (  # [n, P, m]
                (<double>((n + m - 1) * (n - m - 1)) / <double>((2 * n - 3) * (2 * n - 1)))**0.5 * output[idx_elem, nm_minus_two_index + P]  # [n - 2, P, m]
                + (<double>((P + m) * (P - m)) / <double>((2 * P - 1) * (2 * P + 1)))**0.5 * output[idx_elem, nm_minus_index + P - 1]  # [n - 1, P - 1, m]
                - (<double>((P + m + 1) * (P - m + 1)) / <double>((2 * P + 1) * (2 * P + 3)))**0.5 * n_minus_one_buffer[1] # [n - 1, P + 1, m]
            )
            for p in range(P + 1, N + P - n + 1):  # Buffered range
                n_minus_two_buffer[p - P] = scale * (  # [n, p, m]
                    (<double>((n + m - 1) * (n - m - 1)) / <double>((2 * n - 3) * (2 * n - 1)))**0.5 * n_minus_two_buffer[p - P]  # [n - 2, p, m]
                    + (<double>((p + m) * (p - m)) / <double>((2 * p - 1) * (2 * p + 1)))**0.5 * n_minus_one_buffer[p - P - 1]  # [n - 1, p - 1, m]
                    - (<double>((p + m + 1) * (p - m + 1)) / <double>((2 * p + 1) * (2 * p + 3)))**0.5 * n_minus_one_buffer[p - P + 1] # [n - 1, p + 1, m]
                )





def coaxial_translation_transform():
    pass


cdef void coaxial_translation_transform_calculation():
    pass
