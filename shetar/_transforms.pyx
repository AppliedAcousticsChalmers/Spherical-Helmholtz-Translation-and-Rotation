import numpy as np
import cython
from cython.parallel import prange
from ._bases import associated_legendre_polynomials
from ._bases cimport associated_legendre_index, spherical_expansion_index
from ._shapes import prepare_strides, broadcast_shapes
from ._shapes cimport broadcast_index


@cython.cdivision(True)
cdef inline Py_ssize_t colatitude_rotation_index(Py_ssize_t order, Py_ssize_t mode_out, Py_ssize_t mode_in) nogil:
    return (order * (order + 1) * (2 * order + 1)) // 6 + mode_out ** 2 + mode_out + mode_in


def colatitude_unique_to_order(num_unique):
    order = 0
    while colatitude_rotation_index(order, order, order) < num_unique:
        order += 1
    order -= 1
    return order


def colatitude_order_to_unique(order):
    return (order + 1) * (order + 2) * (2 * order + 3) // 6


def colatitude_rotation_coefficients(colatitude, order=None, out=None):
    if order is None and out is None:
        raise ValueError('Cannot calculate rotation coefficients without receiving either the output array or the order')
    output_shape = np.shape(colatitude)

    if order is None:
        order = colatitude_unique_to_order(out.shape[-1])
    num_unique = colatitude_order_to_unique(order)
    if out is None:
        out = np.zeros(output_shape + (num_unique,))
    else:
        if out.shape[:-1] != output_shape:
            raise ValueError(f'Cannot use pre-allocated output of shape {out.shape} for colatitude rotation coefficients with colatitude shape {output_shape}')
        if out.shape[-1] != num_unique:
            raise ValueError(f'Cannot use pre-allocated output of shape {out.shape} for colatitude rotation coefficients of order {order}, requiring {num_unique} unique values')

    cosine_colatitude = np.cos(colatitude)
    sine_colatitude = (1 - cosine_colatitude**2)**0.5

    # special cases and initialization
    # n=0
    out[..., 0] = 1
    if order > 0:
        # n=1, p=0, m=0 special case
        out[..., 1] = cosine_colatitude
        # n=1, p=1, m=-1 special case
        out[..., 2] = (1 - cosine_colatitude) * 0.5
        # n=1, p=1, m=0 special case
        out[..., 3] = sine_colatitude * 2**-0.5
        # n=1, p=1, m=1 
        out[..., 4] = (1 + cosine_colatitude) * 0.5

    cdef:
        double[:, :] out_cy = out.reshape((-1, num_unique))
        double[:, :] legendre = associated_legendre_polynomials(cosine_colatitude.reshape(-1), order=order)
        double[:] cos = cosine_colatitude.reshape(-1)
        double[:] sin = sine_colatitude.reshape(-1)
        Py_ssize_t n, p, m, idx_elem, N = order, num_elem = np.size(colatitude)

    if out.ndim == 1:
        colatitude_rotation_coefficients_calculation(out_cy, legendre, cos, sin, 0, N)
    else:
        for idx_elem in prange(num_elem, nogil=True):
            colatitude_rotation_coefficients_calculation(out_cy, legendre, cos, sin, idx_elem, N)

    return out


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void colatitude_rotation_coefficients_calculation(
    double[:, :] output, double[:, :] legendre, double[:] cos, double[:] sin,
    Py_ssize_t idx_elem, Py_ssize_t N
) nogil:
    cdef:
        Py_ssize_t n, p, m

    for n in range(2, N + 1):
        for p in range(0, n - 1):
            for m in range(-p, 0):
                # recurrence for decreasing m, i.e. calculate (n, p, m) from (n-1, p (±1), m+1)
                output[idx_elem, colatitude_rotation_index(n, p, m)] = (
                    output[idx_elem, colatitude_rotation_index(n - 1, p - 1, m + 1)] * 0.5 * (1 - cos[idx_elem]) * ((n + p - 1) * (n + p))**0.5
                     + output[idx_elem, colatitude_rotation_index(n - 1, p, m + 1)] * sin[idx_elem] * ((n + p) * (n - p))**0.5
                     + output[idx_elem, colatitude_rotation_index(n - 1, p + 1, m + 1)] * 0.5 * (1 + cos[idx_elem]) * ((n - p - 1) * (n - p))**0.5
                )  / ((n - m - 1) * (n - m))**0.5
            # m = 0, calculated from legendre value
            output[idx_elem, colatitude_rotation_index(n, p, 0)] = legendre[idx_elem, associated_legendre_index(n, p)] * (-1)**p * (2. / (2 * n + 1))**0.5
            for m in range(1, p + 1):
                # recurrence for increasing m, i.e. calculate (n, p, m) from (n-1, p (±1), m-1)
                output[idx_elem, colatitude_rotation_index(n, p, m)] = (
                    output[idx_elem, colatitude_rotation_index(n - 1, p - 1, m - 1)] * 0.5 * (1 + cos[idx_elem]) * ((n + p - 1) * (n + p))**0.5
                    - output[idx_elem, colatitude_rotation_index(n - 1, p, m - 1)] * sin[idx_elem] * ((n + p) * (n - p))**0.5
                    + output[idx_elem, colatitude_rotation_index(n - 1, p + 1, m - 1)] * 0.5 * (1 - cos[idx_elem]) * ((n - p - 1) * (n - p))**0.5
                ) / ((n + m - 1) * (n + m))**0.5

        # p=n-1 and p=n are special cases since one or two values are missing in the recurrences
        # p = n-1
        for m in range(-(n - 1), 0):
            output[idx_elem, colatitude_rotation_index(n, n - 1, m)] = (
                output[idx_elem, colatitude_rotation_index(n - 1, n - 2, m + 1)] * 0.5 * (1 - cos[idx_elem]) * (2 * (n - 1) * (2 * n - 1))**0.5
                + output[idx_elem, colatitude_rotation_index(n - 1, n - 1, m + 1)] * sin[idx_elem] * (2 * n - 1)**0.5
            ) / ((n - m - 1) * (n - m))**0.5
        output[idx_elem, colatitude_rotation_index(n, n - 1, 0)] = legendre[idx_elem, associated_legendre_index(n, n - 1)] * (-1)**(n - 1) * (2. / (2 * n + 1))**0.5
        for m in range(1, n):
            output[idx_elem, colatitude_rotation_index(n, n - 1, m)] = (
                output[idx_elem, colatitude_rotation_index(n - 1, n - 1 - 1, m - 1)] * 0.5 * (1 + cos[idx_elem]) * (2 * (n - 1) * (2 * n - 1))**0.5
                - output[idx_elem, colatitude_rotation_index(n - 1, n - 1, m - 1)] * sin[idx_elem] * (2 * n - 1)**0.5
            ) / ((n + m - 1) * (n + m))**0.5
        # p = n
        for m in range(-n, 0):
            output[idx_elem, colatitude_rotation_index(n, n, m)] = (
                output[idx_elem, colatitude_rotation_index(n - 1, n - 1, m + 1)] * 0.5 * (1 - cos[idx_elem]) * ((2 * n - 1) * 2 * n)**0.5
            ) / ((n - m - 1) * (n - m))**0.5
        output[idx_elem, colatitude_rotation_index(n, n, 0)] = legendre[idx_elem, associated_legendre_index(n, n)] * (-1)**n * (2. / (2 * n + 1))**0.5
        for m in range(1, n + 1):
            output[idx_elem, colatitude_rotation_index(n, n, m)] = (
                output[idx_elem, colatitude_rotation_index(n - 1, n - 1, m - 1)] * 0.5 * (1 + cos[idx_elem]) * ((2 * n - 1) * 2 * n)**0.5
            ) / ((n + m - 1) * (n + m))**0.5


def colatitude_rotation_transform(expansion_data, colatitude_rotation_coefficients, inverse=False, out=None):
    output_shape, expansion_shape, transform_shape = broadcast_shapes(
        expansion_data.shape[:-1], colatitude_rotation_coefficients.shape[:-1]
    )

    expansion_order = int(expansion_data.shape[-1] ** 0.5) - 1
    transform_order = colatitude_unique_to_order(colatitude_rotation_coefficients.shape[-1])
    output_order = min(expansion_order, transform_order)
    output_unique = (output_order + 1)**2

    if out is None:
        out = np.zeros(output_shape + (output_unique,), complex)
    else:
        if out.shape[:-1] != output_shape:
            raise ValueError(f'Cannot use pre-allocated output of shape {out.shape} for rotation of expansion of shape {expansion_shape} and rotation shape {transform_shape}')
        if out.shape[-1] != output_unique:
            raise ValueError(f'Cannot use pre-allocated output of shape {out.shape} for rotation of expansion with order {output_order}, requiring {output_unique} unique values')

    cdef:
        double complex[:, :] exp_cy = expansion_data.reshape((-1, expansion_data.shape[-1]))
        double [:, :] trans_cy = colatitude_rotation_coefficients.reshape((-1, colatitude_rotation_coefficients.shape[-1]))
        double complex[:, :] out_cy = out.reshape((-1, out.shape[-1]))
        double complex[:] primary_phase = np.ones(1, complex)
        double complex[:] secondary_phase = np.ones(1, complex)
        rotation_implementation rot_impl
        Py_ssize_t N = output_order

    if inverse:
        rot_impl = colatitude_inverse_impl
    else:
        rot_impl = colatitude_forward_impl

    if out.ndim == 1:
        rotation_core_loop(out_cy, exp_cy, trans_cy, primary_phase, secondary_phase, 0, 0, 0, 0, 0, N, rot_impl)
        return out

    cdef:
        Py_ssize_t[:] exp_stride = prepare_strides(expansion_shape, output_shape)
        Py_ssize_t[:] trans_stride = prepare_strides(transform_shape, output_shape)
        Py_ssize_t[:] out_stride = prepare_strides(output_shape, output_shape)
        Py_ssize_t exp_elem_idx, trans_elem_idx, out_elem_idx
        Py_ssize_t num_elements = out.shape[0], ndim = out.ndim

    for out_elem_idx in prange(num_elements, nogil=True):
        exp_elem_idx = broadcast_index(out_elem_idx, exp_stride, out_stride, ndim)
        trans_elem_idx = broadcast_index(out_elem_idx, trans_stride, out_stride, ndim)
        rotation_core_loop(out_cy, exp_cy, trans_cy, primary_phase, secondary_phase, out_elem_idx, exp_elem_idx, trans_elem_idx, 0, 0, N, rot_impl)

    return out


def full_rotation_transform(expansion_data, colatitude_rotation_coefficients, primary_phase, secondary_phase, inverse=False, out=None):
    output_shape, expansion_shape, colatitude_shape, primary_shape, secondary_shape = broadcast_shapes(
        expansion_data.shape[:-1], colatitude_rotation_coefficients.shape[:-1], primary_phase.shape, secondary_phase.shape
    )

    expansion_order = int(expansion_data.shape[-1] ** 0.5) - 1
    transform_order = colatitude_unique_to_order(colatitude_rotation_coefficients.shape[-1])
    output_order = min(expansion_order, transform_order)
    output_unique = (output_order + 1)**2

    if out is None:
        out = np.zeros(output_shape + (output_unique,), complex)
    else:
        if out.shape[:-1] != output_shape:
            raise ValueError(f'Cannot use pre-allocated output of shape {out.shape}, expected shape {output_shape}')
        if out.shape[-1] != output_unique:
            raise ValueError(f'Cannot use pre-allocated output of shape {out.shape} for rotation of expansion with order {output_order}, requiring {output_unique} unique values')

    cdef:
        double complex[:, :] exp_cy = expansion_data.reshape((-1, expansion_data.shape[-1]))
        double [:, :] trans_cy = colatitude_rotation_coefficients.reshape((-1, colatitude_rotation_coefficients.shape[-1]))
        double complex[:, :] out_cy = out.reshape((-1, out.shape[-1]))
        double complex[:] primary_phase_cy = primary_phase.reshape(-1)
        double complex[:] secondary_phase_cy = secondary_phase.reshape(-1)
        rotation_implementation rot_impl
        Py_ssize_t N = output_order

    if inverse:
        rot_impl = rotation_inverse_impl
    else:
        rot_impl = rotation_forward_impl

    if out.ndim == 1:
        rotation_core_loop(out_cy, exp_cy, trans_cy, primary_phase_cy, secondary_phase_cy, 0, 0, 0, 0, 0, N, rot_impl)
        return out

    cdef:
        Py_ssize_t[:] exp_stride = prepare_strides(expansion_shape, output_shape)
        Py_ssize_t[:] colatitude_stride = prepare_strides(colatitude_shape, output_shape)
        Py_ssize_t[:] primary_stride = prepare_strides(primary_shape, output_shape)
        Py_ssize_t[:] secondary_stride = prepare_strides(secondary_shape, output_shape)
        Py_ssize_t[:] out_stride = prepare_strides(output_shape, output_shape)
        Py_ssize_t exp_elem_idx, colat_elem_idx, prim_elem_idx, second_elem_idx, out_elem_idx
        Py_ssize_t num_elements = out.shape[0], ndim = out.ndim

    for out_elem_idx in prange(num_elements, nogil=True):
        exp_elem_idx = broadcast_index(out_elem_idx, exp_stride, out_stride, ndim)
        colat_elem_idx = broadcast_index(out_elem_idx, colatitude_stride, out_stride, ndim)
        prim_elem_idx = broadcast_index(out_elem_idx, primary_stride, out_stride, ndim)
        second_elem_idx = broadcast_index(out_elem_idx, secondary_stride, out_stride, ndim)
        rotation_core_loop(out_cy, exp_cy, trans_cy, primary_phase_cy, secondary_phase_cy, out_elem_idx, exp_elem_idx, colat_elem_idx, prim_elem_idx, second_elem_idx, N, rot_impl)

    return out


ctypedef void (*rotation_implementation)(
    Py_ssize_t n,
    Py_ssize_t p,
    Py_ssize_t m,
    double complex [:, :] output,
    double complex [:, :] expansion,
    double transform,
    double complex [:] primary_phase,
    double complex [:] secondary_phase,
    Py_ssize_t out_elem_idx,
    Py_ssize_t exp_elem_idx,
    Py_ssize_t primary_elem_idx,
    Py_ssize_t secondary_elem_idx
) nogil


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef inline void colatitude_forward_impl(
    Py_ssize_t n,
    Py_ssize_t p,
    Py_ssize_t m,
    double complex [:, :] output,
    double complex [:, :] expansion,
    double transform,
    double complex [:] primary_phase,
    double complex [:] secondary_phase,
    Py_ssize_t out_elem_idx,
    Py_ssize_t exp_elem_idx,
    Py_ssize_t primary_elem_idx,
    Py_ssize_t secondary_elem_idx
) nogil:
    cdef Py_ssize_t out_idx = spherical_expansion_index(n, m)
    cdef Py_ssize_t exp_idx = spherical_expansion_index(n, p)
    output[out_elem_idx, out_idx] = output[out_elem_idx, out_idx] + expansion[exp_elem_idx, exp_idx] * transform


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef inline void colatitude_inverse_impl(
    Py_ssize_t n,
    Py_ssize_t p,
    Py_ssize_t m,
    double complex [:, :] output,
    double complex [:, :] expansion,
    double transform,
    double complex [:] primary_phase,
    double complex [:] secondary_phase,
    Py_ssize_t out_elem_idx,
    Py_ssize_t exp_elem_idx,
    Py_ssize_t primary_elem_idx,
    Py_ssize_t secondary_elem_idx
) nogil:
    cdef Py_ssize_t out_idx = spherical_expansion_index(n, p)
    cdef Py_ssize_t exp_idx = spherical_expansion_index(n, m)
    output[out_elem_idx, out_idx] = output[out_elem_idx, out_idx] + expansion[exp_elem_idx, exp_idx] * transform


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef inline void rotation_forward_impl(
    Py_ssize_t n,
    Py_ssize_t p,
    Py_ssize_t m,
    double complex [:, :] output,
    double complex [:, :] expansion,
    double transform,
    double complex [:] primary_phase,
    double complex [:] secondary_phase,
    Py_ssize_t out_elem_idx,
    Py_ssize_t exp_elem_idx,
    Py_ssize_t primary_elem_idx,
    Py_ssize_t secondary_elem_idx
) nogil:
    cdef Py_ssize_t out_idx = spherical_expansion_index(n, m)
    cdef Py_ssize_t exp_idx = spherical_expansion_index(n, p)
    output[out_elem_idx, out_idx] = output[out_elem_idx, out_idx] + (
        expansion[exp_elem_idx, exp_idx] * transform 
        * primary_phase[primary_elem_idx] ** (-m) * secondary_phase[secondary_elem_idx] ** (-p)
    )


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef inline void rotation_inverse_impl(
    Py_ssize_t n,
    Py_ssize_t p,
    Py_ssize_t m,
    double complex [:, :] output,
    double complex [:, :] expansion,
    double transform,
    double complex [:] primary_phase,
    double complex [:] secondary_phase,
    Py_ssize_t out_elem_idx,
    Py_ssize_t exp_elem_idx,
    Py_ssize_t primary_elem_idx,
    Py_ssize_t secondary_elem_idx
) nogil:
    cdef Py_ssize_t out_idx = spherical_expansion_index(n, p)
    cdef Py_ssize_t exp_idx = spherical_expansion_index(n, m)
    output[out_elem_idx, out_idx] = output[out_elem_idx, out_idx] + (
        expansion[exp_elem_idx, exp_idx] * transform 
        * primary_phase[primary_elem_idx] ** m * secondary_phase[secondary_elem_idx] ** p
    )


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void rotation_core_loop(
    double complex [:, :] output,
    double complex [:, :] expansion,
    double [:, :] transform,
    double complex [:] primary_phase,
    double complex [:] secondary_phase,
    Py_ssize_t out_elem_idx,
    Py_ssize_t exp_elem_idx,
    Py_ssize_t trans_elem_idx,
    Py_ssize_t primary_elem_idx,
    Py_ssize_t secondary_elem_idx,
    Py_ssize_t N,
    rotation_implementation rot_impl
) nogil:
    cdef:
        Py_ssize_t n, m, p, trans_idx
        int sign

    trans_idx = -1
    for n in range(N + 1):
        trans_idx += 1
        # trans_idx <=> trans[n, 0, 0]
        rot_impl(n, 0, 0, output, expansion, transform[trans_elem_idx, trans_idx], primary_phase, secondary_phase, out_elem_idx, exp_elem_idx, primary_elem_idx, secondary_elem_idx)
        
        for p in range(1, n + 1):
            trans_idx += 1
            # trans_idx <=> trans[n, p, -p]
            rot_impl(n, p, -p, output, expansion, transform[trans_elem_idx, trans_idx], primary_phase, secondary_phase, out_elem_idx, exp_elem_idx, primary_elem_idx, secondary_elem_idx)
            rot_impl(n, -p, p, output, expansion, transform[trans_elem_idx, trans_idx], primary_phase, secondary_phase, out_elem_idx, exp_elem_idx, primary_elem_idx, secondary_elem_idx)

            for m in range(-p + 1, p):
                trans_idx += 1
                # trans_idx <=> trans[n, p, m]
                sign = (-1) ** (p + m)
                rot_impl(n, p, m, output, expansion, transform[trans_elem_idx, trans_idx], primary_phase, secondary_phase, out_elem_idx, exp_elem_idx, primary_elem_idx, secondary_elem_idx)
                rot_impl(n, -p, -m, output, expansion, sign * transform[trans_elem_idx, trans_idx], primary_phase, secondary_phase, out_elem_idx, exp_elem_idx, primary_elem_idx, secondary_elem_idx)
                rot_impl(n, m, p, output, expansion, sign * transform[trans_elem_idx, trans_idx], primary_phase, secondary_phase, out_elem_idx, exp_elem_idx, primary_elem_idx, secondary_elem_idx)
                rot_impl(n, -m, -p, output, expansion, transform[trans_elem_idx, trans_idx], primary_phase, secondary_phase, out_elem_idx, exp_elem_idx, primary_elem_idx, secondary_elem_idx)


            trans_idx += 1
            # trans_idx <=> trans[n, p, p]
            rot_impl(n, p, p, output, expansion, transform[trans_elem_idx, trans_idx], primary_phase, secondary_phase, out_elem_idx, exp_elem_idx, primary_elem_idx, secondary_elem_idx)
            rot_impl(n, -p, -p, output, expansion, transform[trans_elem_idx, trans_idx], primary_phase, secondary_phase, out_elem_idx, exp_elem_idx, primary_elem_idx, secondary_elem_idx)
