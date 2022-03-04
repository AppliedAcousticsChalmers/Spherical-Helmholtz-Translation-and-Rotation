import numpy as np
import cython
cimport cython
from ._shapes import prepare_strides, broadcast_shapes
from ._shapes cimport broadcast_index
from ._bases cimport spherical_expansion_index

from scipy.special import spherical_jn, spherical_yn

ctypedef fused double_complex:
    double
    double complex

@cython.cdivision(True)
cdef inline Py_ssize_t coaxial_translation_mode_index(Py_ssize_t mode, Py_ssize_t low_order, Py_ssize_t high_order) nogil:
    return (
        mode * (low_order + 1) * (high_order + 1)
        - mode * (low_order * (low_order + 1)) // 2
        - high_order * (mode * (mode + 1)) // 2
        + (mode * (mode - 1) * (mode - 2)) // 6
        - mode
    )


@cython.cdivision(True)
cdef inline Py_ssize_t coaxial_translation_order_index(Py_ssize_t input_order, Py_ssize_t high_order) nogil:
    return input_order * high_order - (input_order * (input_order - 1)) // 2


cdef inline Py_ssize_t coaxial_translation_index(Py_ssize_t input_order, Py_ssize_t output_order, Py_ssize_t mode, Py_ssize_t low_order, Py_ssize_t high_order) nogil:
    return coaxial_translation_mode_index(mode, low_order, high_order) + coaxial_translation_order_index(input_order, high_order) + output_order


def coaxial_order_to_unique(low_order, high_order):
    num_unique = (
        (low_order + 1)**2 * (high_order + 1)
        - (low_order * (low_order + 1)) // 2 * (low_order + high_order + 2)
        + (low_order * (low_order - 1) * (low_order + 1)) // 6
    )
    return num_unique


def coaxial_translation_interdomain_coefficients(distance, Py_ssize_t low_order, Py_ssize_t high_order, out=None):
    output_shape = np.shape(distance)

    num_unique = coaxial_order_to_unique(low_order, high_order)
    if out is None:
        out = np.zeros(output_shape + (num_unique,), dtype=complex)
    else:
        if out.shape[:-1] != output_shape:
            raise ValueError(f'Cannot use pre-allocated output of shape {out.shape} for coaxial translation coefficients with distance shape {output_shape}')
        if out.shape[-1] != num_unique:
            raise ValueError(f'Cannot use pre-allocated output of shape {out.shape} for coaxial translation coefficients of {low_order = } and {high_order = }, requiring {num_unique} unique values')

    buffer_shape = low_order + 1
    x_init = distance.reshape((-1, 1))
    n_init = np.arange(low_order + high_order + 1)
    cdef:
        int num_elements = np.size(distance)
        double complex [:, :] initialization = spherical_jn(n_init, x_init) + 1j * spherical_yn(n_init, x_init)
        double complex [:, :] out_cy = out.reshape((-1, num_unique))
        double complex [:, :] initialization_cy = initialization
        Py_ssize_t idx_elem
        double complex [:] m_buffer = np.zeros(buffer_shape, complex)
        double complex [:] m_minus_one_buffer = np.zeros(buffer_shape, complex)
        double complex [:] n_minus_one_buffer = np.zeros(buffer_shape, complex)
        double complex [:] n_minus_two_buffer = np.zeros(buffer_shape, complex)

    with nogil:
        for idx_elem in range(num_elements):
            coaxial_translation_coefficients_calculation(
                out_cy, initialization_cy, idx_elem,
                low_order, high_order,
                m_buffer, m_minus_one_buffer,
                n_minus_one_buffer, n_minus_two_buffer,
            )

    return out


def coaxial_translation_intradomain_coefficients(distance, Py_ssize_t low_order, Py_ssize_t high_order, out=None):
    output_shape = np.shape(distance)

    num_unique = coaxial_order_to_unique(low_order, high_order)
    if out is None:
        out = np.zeros(output_shape + (num_unique,), dtype=float)
    else:
        if out.shape[:-1] != output_shape:
            raise ValueError(f'Cannot use pre-allocated output of shape {out.shape} for coaxial translation coefficients with distance shape {output_shape}')
        if out.shape[-1] != num_unique:
            raise ValueError(f'Cannot use pre-allocated output of shape {out.shape} for coaxial translation coefficients of {low_order = } and {high_order = }, requiring {num_unique} unique values')

    buffer_shape = low_order + 1
    x_init = distance.reshape((-1, 1))
    n_init = np.arange(low_order + high_order + 1)
    cdef:
        int num_elements = np.size(distance)
        double[:, :] initialization = spherical_jn(n_init, x_init)
        double[:, :] out_cy = out.reshape((-1, num_unique))
        Py_ssize_t idx_elem
        double[:] m_buffer = np.zeros(buffer_shape, float)
        double[:] m_minus_one_buffer = np.zeros(buffer_shape, float)
        double[:] n_minus_one_buffer = np.zeros(buffer_shape, float)
        double[:] n_minus_two_buffer = np.zeros(buffer_shape, float)

    with nogil:
        for idx_elem in range(num_elements):
            coaxial_translation_coefficients_calculation(
                out_cy, initialization, idx_elem,
                low_order, high_order,
                m_buffer, m_minus_one_buffer,
                n_minus_one_buffer, n_minus_two_buffer,
            )

    return out


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void coaxial_translation_coefficients_calculation(
    double_complex[:, :] output,
    double_complex[:, :] initialization,
    Py_ssize_t idx_elem,
    Py_ssize_t N,
    Py_ssize_t P,
    double_complex[:] m_buffer,
    double_complex[:] m_minus_one_buffer,
    double_complex[:] n_minus_one_buffer,
    double_complex[:] n_minus_two_buffer,
) nogil:

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




def coaxial_translation_transform(expansion_data, coaxial_translation_coefficients, Py_ssize_t low_order, Py_ssize_t high_order, inverse=False, out=None):
    output_shape, expansion_shape, transform_shape = broadcast_shapes(
        expansion_data.shape[:-1], coaxial_translation_coefficients.shape[:-1]
    )

    if coaxial_translation_coefficients.shape[-1] !=  coaxial_order_to_unique(low_order, high_order):
        raise ValueError(f'Coaxial translation coefficients with {coaxial_translation_coefficients.shape[-1]} unique values does not match specifiend tranfsform orders {low_order, high_order}')
    expansion_order = int(expansion_data.shape[-1] ** 0.5) - 1

    if out is None:
        result_order = low_order if expansion_order > low_order else high_order
        out = np.zeros(output_shape + ((result_order + 1)**2,), complex)
    else:
        result_order = int(out.shape[-1] ** 0.5) - 1
        if out.shape[:-1] != output_shape:
            raise ValueError(f'Cannot use pre-allocated output of shape {out.shape} for translation of expansion of shape {expansion_shape} and transform shape {transform_shape}')

    # Checks for order.
    # This will maintain the relation between expantion_order and result_order if they are different,
    # but truncate them to the allowable order in the transform if needed.
    # If they are equal, they will be truncated to the low order if needed.
    if expansion_order < result_order:
        expansion_order = min(expansion_order, low_order)
        result_order = min(result_order, high_order)
    elif result_order < expansion_order:
        result_order = min(result_order, low_order)
        expansion_order = min(expansion_order, high_order)
    else:
        # expansion_order == result_order
        # If the input/output both are higher than the low order, we cannot possiby guess which of the two should be keps as the high one, so we truncate both of them instead.
        expansion_order = min(expansion_order, low_order)
        result_order = min(result_order, low_order)

    if not np.iscomplexobj(coaxial_translation_coefficients):
        # We need a complex object so that the assignment wont break.
        # This will use a bit extra memory while running, but probably not extra
        # cpu since the doubles will always be promoted to complex in the calculation anyhow,
        # seing how they are always multiplied with other complex values.
        coaxial_translation_coefficients = coaxial_translation_coefficients + 0j

    cdef:
        double complex [:, :] out_cy = out.reshape((-1, out.shape[-1]))
        double complex [:, :] exp_cy = expansion_data.reshape((-1, expansion_data.shape[-1]))
        double complex[:, :] trans_cy = coaxial_translation_coefficients.reshape((-1, coaxial_translation_coefficients.shape[-1]))
        translation_implementation trans_func
        Py_ssize_t N_max, P_max

    if inverse:
        trans_func = inverse_coaxial_implementation
        N_max = result_order
        P_max = expansion_order
    else:
        trans_func = forward_coaxial_implementation
        P_max = result_order
        N_max = expansion_order

    if out.size == out.shape[-1]:
        # No loop over elements
        coaxial_translation_transform_calculation(out_cy, exp_cy, trans_cy, 0, 0, 0, N_max, P_max, low_order, high_order, trans_func)
        return out

    cdef:
        Py_ssize_t[:] out_stride = prepare_strides(output_shape, output_shape)
        Py_ssize_t[:] exp_stride = prepare_strides(expansion_shape, output_shape)
        Py_ssize_t[:] trans_stride = prepare_strides(transform_shape, output_shape)
        Py_ssize_t out_elem_idx, exp_elem_idx, trans_elem_idx
        Py_ssize_t num_elements = out_cy.shape[0], ndim = out.ndim

    with nogil:
        for out_elem_idx in range(num_elements):
            exp_elem_idx = broadcast_index(out_elem_idx, exp_stride, out_stride, ndim)
            trans_elem_idx = broadcast_index(out_elem_idx, trans_stride, out_stride, ndim)
            coaxial_translation_transform_calculation(out_cy, exp_cy, trans_cy, out_elem_idx, exp_elem_idx, trans_elem_idx, N_max, P_max, low_order, high_order, trans_func)

    return out


ctypedef void (*translation_implementation)(
    Py_ssize_t n,
    Py_ssize_t p,
    Py_ssize_t m,
    double complex [:, :] output,
    double complex [:, :] expansion,
    double complex transform,
    Py_ssize_t out_elem_idx,
    Py_ssize_t exp_elem_idx,
) nogil


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void forward_coaxial_implementation(
    Py_ssize_t n,
    Py_ssize_t p,
    Py_ssize_t m,
    double complex [:, :] output,
    double complex [:, :] expansion,
    double complex transform,
    Py_ssize_t out_elem_idx,
    Py_ssize_t exp_elem_idx
) nogil:
    cdef Py_ssize_t out_idx = spherical_expansion_index(p, m)
    cdef Py_ssize_t exp_idx = spherical_expansion_index(n, m)
    output[out_elem_idx, out_idx] = output[out_elem_idx, out_idx] + expansion[exp_elem_idx, exp_idx] * transform


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void inverse_coaxial_implementation(
    Py_ssize_t n,
    Py_ssize_t p,
    Py_ssize_t m,
    double complex [:, :] output,
    double complex [:, :] expansion,
    double complex transform,
    Py_ssize_t out_elem_idx,
    Py_ssize_t exp_elem_idx
) nogil:
    cdef Py_ssize_t out_idx = spherical_expansion_index(n, m)
    cdef Py_ssize_t exp_idx = spherical_expansion_index(p, m)
    output[out_elem_idx, out_idx] = output[out_elem_idx, out_idx] + expansion[exp_elem_idx, exp_idx] * transform



@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void coaxial_translation_transform_calculation(
    double complex [:, :] output,
    double complex [:, :] expansion,
    double complex [:, :] transform,
    Py_ssize_t out_elem_idx,
    Py_ssize_t exp_elem_idx,
    Py_ssize_t trans_elem_idx,
    Py_ssize_t N_max,
    Py_ssize_t P_max,
    Py_ssize_t low_order,
    Py_ssize_t high_order,
    translation_implementation trans_func,
) nogil:
    cdef:
        Py_ssize_t trans_idx, n, p, m
        short sign
        Py_ssize_t NP_min = min(N_max, P_max)
        Py_ssize_t NP_max = max(N_max, P_max)
        Py_ssize_t N_skip = ((low_order - NP_min) * (2 * high_order - low_order - NP_min + 1))//2
        Py_ssize_t P_skip = (high_order - NP_max)
    # comments indicate [n, p, m]
    trans_idx = -1
    # deal with m=0, since that removes the -m symmetry
    for n in range(NP_min + 1):
        trans_idx += 1
        sign = 1  # (-1)**(2n)
        # trans_idx <=> trans[n, n, 0]
        trans_func(n, n, 0, output, expansion, transform[trans_elem_idx, trans_idx], out_elem_idx, exp_elem_idx)
        for p in range(n + 1, NP_min + 1):
            trans_idx += 1
            sign = -sign
            # trans_idx <=> trans[n, p, 0]
            trans_func(n, p, 0, output, expansion, transform[trans_elem_idx, trans_idx], out_elem_idx, exp_elem_idx)
            trans_func(p, n, 0, output, expansion, sign * transform[trans_elem_idx, trans_idx], out_elem_idx, exp_elem_idx)
        # Only one of the below loops will run. The one which runs is determined by which of N and P is the largest.
        # If P is the largest, the coefficent is stored.
        # If N is the largest, we use the symmetry.
        for p in range(NP_min + 1, P_max + 1):
            trans_idx += 1
            # trans_idx <=> trans[n, p, 0]
            trans_func(n, p, 0, output, expansion, transform[trans_elem_idx, trans_idx], out_elem_idx, exp_elem_idx)
        for p in range(NP_min + 1, N_max + 1):
            trans_idx += 1
            sign = -sign
            # trans_idx <=> trans[n, p, 0]
            trans_func(p, n, 0, output, expansion, sign * transform[trans_elem_idx, trans_idx], out_elem_idx, exp_elem_idx)
        trans_idx += P_skip
    trans_idx += N_skip

    for m in range(1, NP_min + 1):
        for n in range(m, NP_min + 1):
            # trans_idx <=> trans[n, n, m]
            trans_idx += 1
            sign = 1  # (-1)**(2n)
            trans_func(n, n, m, output, expansion, transform[trans_elem_idx, trans_idx], out_elem_idx, exp_elem_idx)
            trans_func(n, n, -m, output, expansion, transform[trans_elem_idx, trans_idx], out_elem_idx, exp_elem_idx)
            for p in range(n + 1, NP_min + 1):
                trans_idx += 1
                sign = -sign
                # trans_idx <=> trans[n, p, m]
                trans_func(n, p, m, output, expansion, transform[trans_elem_idx, trans_idx], out_elem_idx, exp_elem_idx)
                trans_func(n, p, -m, output, expansion, transform[trans_elem_idx, trans_idx], out_elem_idx, exp_elem_idx)
                trans_func(p, n, m, output, expansion, sign * transform[trans_elem_idx, trans_idx], out_elem_idx, exp_elem_idx)
                trans_func(p, n, -m, output, expansion, sign * transform[trans_elem_idx, trans_idx], out_elem_idx, exp_elem_idx)
            for p in range(NP_min + 1, P_max + 1):
                trans_idx += 1
                # trans_idx <=> trans[n, p, m]
                trans_func(n, p, m, output, expansion, transform[trans_elem_idx, trans_idx], out_elem_idx, exp_elem_idx)
                trans_func(n, p, -m, output, expansion, transform[trans_elem_idx, trans_idx], out_elem_idx, exp_elem_idx)
            for p in range(NP_min + 1, N_max + 1):
                trans_idx += 1
                sign = -sign
                # trans_idx <=> trans[n, p, m]
                trans_func(p, n, m, output, expansion, sign * transform[trans_elem_idx, trans_idx], out_elem_idx, exp_elem_idx)
                trans_func(p, n, -m, output, expansion, sign * transform[trans_elem_idx, trans_idx], out_elem_idx, exp_elem_idx)
            trans_idx += P_skip
        trans_idx += N_skip
