import numpy as np
import cython
cimport cython
from ._shapes import prepare_strides, broadcast_shapes
from ._shapes cimport broadcast_index


@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline Py_ssize_t associated_legendre_index(Py_ssize_t order, Py_ssize_t mode) nogil:
    return (order * (order + 1)) / 2 + mode


@cython.boundscheck(False)
cdef inline Py_ssize_t spherical_expansion_index(Py_ssize_t order, Py_ssize_t mode) nogil:
    return order ** 2 + order + mode


def associated_legendre_indexing(base_data, indices):
    indices = np.asarray(indices).astype(int)
    out = np.zeros(base_data.shape[:-1] + (indices.shape[0],), float)
    
    for out_idx, (n, m) in enumerate(indices):
        base_idx = associated_legendre_index(n, abs(m))
        if m < 0:
            out[..., out_idx] = base_data[..., base_idx] * (-1) ** abs(m)
        else:
            out[..., out_idx] = base_data[..., base_idx]
    return out


def spherical_harmonics_indexing(legendre_data, phase_data, indices):
    indices = np.asarray(indices).astype(int)
    legendre_indexed = associated_legendre_indexing(legendre_data, indices)

    out = legendre_indexed * phase_data[..., None] ** indices[:, 1] / (2 * np.pi)**0.5
    return out


def multipole_indexing(radial_data, legendre_data, phase_data, indices):
    indices = np.asarray(indices).astype(int)
    spherical_harmonics_indexed = spherical_harmonics_indexing(legendre_data, phase_data, indices)
    out = spherical_harmonics_indexed * radial_data[..., indices[:, 0]]
    return out


def legendre_polynomials(x, order=None, out=None):
    if order is None and out is None:
        raise ValueError('Cannot calculate Legendre polynomials without receiving either the output array or the order')
    output_shape = np.shape(x)

    if order is None:
        order = out.shape[-1] - 1
    num_unique = order + 1
    if out is None:
        out = np.zeros(output_shape + (num_unique,))
    else:
        if out.shape[:-1] != output_shape:
            raise ValueError(f'Cannot use pre-allocated output of shape {out.shape} for legendre polynomials with argument shape {output_shape}')
        if out.shape[-1] != num_unique:
            raise ValueError(f'Cannot use pre-allocated output of shape {out.shape} for legendre polynomials of order {order}, requiring {num_unique} unique values')

    out[..., 0] = 1 / 2**0.5
    if order >= 1:
        out[..., 1] = x * 1.5**0.5

    cdef:
        Py_ssize_t idx, N = order, num_elements = np.size(x)
        double[:] x_cy = x.reshape(-1)
        double[:, :] out_cy = out.reshape((np.size(x), num_unique))


    with nogil:
        for idx in range(num_elements):
            legendre_polynomials_calculation(out_cy, x_cy, idx, N)

    return out


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void legendre_polynomials_calculation(
    double[:, :] output,
    double[:] x,
    Py_ssize_t idx,
    Py_ssize_t N
) nogil:
    cdef:
        Py_ssize_t n
        double n_minus_1_factor, n_minus_2_factor

    for n in range(2, N + 1):
        n_minus_1_factor = ((2 * n + 1) * (2 * n - 1))**0.5 / n
        n_minus_2_factor = <double>(n - 1) / n * (<double>(2 * n + 1) / (2 * n - 3))**0.5
        output[idx, n] = x[idx] * output[idx, n - 1] * n_minus_1_factor - output[idx, n - 2] * n_minus_2_factor


def associated_legendre_polynomials(x, order=None, out=None):
    if order is None and out is None:
        raise ValueError('Cannot calculate associated Legendre polynomials without receiving either the output array or the order')
    output_shape = np.shape(x)

    if order is None:
        order = int((8 * out.shape[-1] + 1)**0.5 - 3) // 2
    num_unique = (order + 1) * (order + 2) // 2
    if out is None:
        out = np.zeros(output_shape + (num_unique,))
    else:
        if out.shape[:-1] != output_shape:
            raise ValueError(f'Cannot use pre-allocated output of shape {out.shape} for associated legendre polynomials with argument shape {output_shape}')
        if out.shape[-1] != num_unique:
            raise ValueError(f'Cannot use pre-allocated output of shape {out.shape} for associated legendre polynomials of order {order}, requiring {num_unique} unique values')

    out[..., 0] = 1 / 2**0.5

    cdef:
        Py_ssize_t idx_elem, N = order, num_elements = np.size(x)
        double[:] x_cy = x.reshape(-1)
        double[:] one_minus_x_square = 1 - x.reshape(-1)**2
        double[:, :] out_cy = out.reshape((-1, num_unique))

    with nogil:
        for idx_elem in range(num_elements):
            associated_legendre_polynomials_calculation(out_cy, x_cy, one_minus_x_square, idx_elem, N)
    return out


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void associated_legendre_polynomials_calculation(
        double[:, :] data,
        double[:] x,
        double[:] one_minus_x_square,
        Py_ssize_t idx_elem,
        Py_ssize_t N
) nogil:
    cdef:
        Py_ssize_t n, m, idx_assign, idx_m1, idx_m2
        double fact_m1, fact_m2

    for n in range(1, N + 1):
        # Recurrence to higher orders
        idx_assign = associated_legendre_index(n, n)
        idx_m1 = associated_legendre_index(n - 1, n - 1)
        fact_m1 = - (<double>(2 * n + 1) / (2 * n))**0.5        
        data[idx_elem, idx_assign] = fact_m1 * data[idx_elem, idx_m1]

    for n in range(1, N + 1):
        # Recurrence to lower modes for each order.
        # This is separated to a new loop since we modify the values in place as we go.

        # Same recurrence as below, but excluding the mode+2 part explicitly.
        idx_assign = associated_legendre_index(n, n - 1)
        idx_m1 = associated_legendre_index(n, n)
        fact_m1 = - (2 * n)**0.5
        data[idx_elem, idx_assign] = fact_m1 * data[idx_elem, idx_m1] * x[idx_elem]

        for m in range(n - 2, -1, -1):  # [n - 2, n - 3, ... 0]
            # Recurrece to lower modes
            idx_assign = associated_legendre_index(n, m)
            idx_m1 = associated_legendre_index(n, m + 1)
            idx_m2 = associated_legendre_index(n, m + 2)
            fact_m1 = - 2 * <double>(m + 1) / ((n + m + 1) * (n - m)) ** 0.5
            fact_m2 = -(<double>((n + m + 2) * (n - m - 1)) / ((n - m) * (n + m + 1))) ** 0.5
            data[idx_elem, idx_assign] = (
                fact_m1 * data[idx_elem, idx_m1] * x[idx_elem]
                + fact_m2 * data[idx_elem, idx_m2] * one_minus_x_square[idx_elem]
            )
            # Modify the (n, m+2) components to have the correct normalization.
            # This cannot be done earlier since the recurrence is not stable if this scale is included!
            data[idx_elem, idx_m2] *= one_minus_x_square[idx_elem] ** (0.5 * m + 1)
        # Normalization for (n, 1) components. (n, 0) needs no modification.
        data[idx_elem, idx_m1] *= one_minus_x_square[idx_elem] ** 0.5


def legendre_contraction(expansion_data, base_data, out=None):
    output_shape, expansion_shape, base_shape = broadcast_shapes(
        expansion_data.shape[:-1], base_data.shape[:-1]
    )

    expansion_order = expansion_data.shape[-1]
    base_order = base_data.shape[-1]
    output_order = min(expansion_order, base_order)

    if out is None:
        out = np.zeros(output_shape)
    elif out.shape != output_shape:
        raise ValueError(f'Cannot use array of shape {out.shape} as output array for contraction between expansion with shape {expansion_shape} and bases with shape {base_shape}')

    cdef:
        double[:, :] exp_cy = expansion_data.reshape((-1, expansion_order))
        double[:, :] base_cy = base_data.reshape((-1, base_order))
        double[:] out_cy = out.reshape(-1)
        Py_ssize_t N = output_order

    if out.size == 1:
        # No loop over elements, saves notable time due to stride calculations.
        legendre_contraction_calculation(out_cy, exp_cy, base_cy, 0, 0, 0, N)
        return out

    cdef:
        Py_ssize_t[:] exp_stride = prepare_strides(expansion_shape, output_shape)
        Py_ssize_t[:] base_stride = prepare_strides(base_shape, output_shape)
        Py_ssize_t[:] out_stride = prepare_strides(output_shape, output_shape)
        Py_ssize_t exp_idx, base_idx, out_idx
        Py_ssize_t num_elements = out.size, ndim = out.ndim

    with nogil:
        for out_idx in range(num_elements):
            exp_idx = broadcast_index(out_idx, exp_stride, out_stride, ndim)
            base_idx = broadcast_index(out_idx, base_stride, out_stride, ndim)
            legendre_contraction_calculation(out_cy, exp_cy, base_cy, out_idx, exp_idx, base_idx, N)

    return out


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void legendre_contraction_calculation(
    double[:] output,
    double[:, :] expansion_data,
    double[:, :] base_data,
    Py_ssize_t out_elem_idx,
    Py_ssize_t exp_elem_idx,
    Py_ssize_t base_elem_idx,
    Py_ssize_t N
) nogil :
    cdef Py_ssize_t n
    for n in range(N):
        output[out_elem_idx] += expansion_data[exp_elem_idx, n] * base_data[base_elem_idx, n]


def associated_legendre_contraction(expansion_data, base_data, out=None):
    output_shape, expansion_shape, base_shape = broadcast_shapes(
        expansion_data.shape[:-1], base_data.shape[:-1]
    )

    base_order = int((8 * base_data.shape[-1] + 1)**0.5 - 3) // 2
    expansion_order = int(expansion_data.shape[-1] ** 0.5) - 1
    output_order = min(base_order, expansion_order)

    if out is None:
        out = np.zeros(output_shape)
    elif out.shape != output_shape:
        raise ValueError(f'Cannot use array of shape {out.shape} as output array for contraction between expansion with shape {expansion_shape} and bases with shape {base_shape}')

    cdef:
        double[:, :] exp_cy = expansion_data.reshape((-1, expansion_data.shape[-1]))
        double[:, :] base_cy = base_data.reshape((-1, base_data.shape[-1]))
        double[:] out_cy = out.reshape(-1)
        Py_ssize_t N = output_order

    if out.size == 1:
        # No loop over elements, saves notable time due to stride calculations.
        associated_legendre_contraction_calculation(out_cy, exp_cy, base_cy, 0, 0, 0, N)
        return out

    cdef:
        Py_ssize_t[:] exp_stride = prepare_strides(expansion_shape, output_shape)
        Py_ssize_t[:] base_stride = prepare_strides(base_shape, output_shape)
        Py_ssize_t[:] out_stride = prepare_strides(output_shape, output_shape)
        Py_ssize_t exp_elem_idx, base_elem_idx, out_elem_idx
        Py_ssize_t num_elements = out.size, ndim = out.ndim

    with nogil:
        for out_elem_idx in range(num_elements):
            exp_elem_idx = broadcast_index(out_elem_idx, exp_stride, out_stride, ndim)
            base_elem_idx = broadcast_index(out_elem_idx, base_stride, out_stride, ndim)
            associated_legendre_contraction_calculation(out_cy, exp_cy, base_cy, out_elem_idx, exp_elem_idx, base_elem_idx, N)
    return out


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void associated_legendre_contraction_calculation(
    double[:] output,
    double[:, :] expansion_data,
    double[:, :] base_data,
    Py_ssize_t out_elem_idx,
    Py_ssize_t exp_elem_idx,
    Py_ssize_t base_elem_idx,
    Py_ssize_t N
) nogil:
    cdef:
        Py_ssize_t n, m, exp_idx, base_idx, exp_idx_neg
        int sign

    for n in range(N + 1):
        exp_idx = spherical_expansion_index(n, 0)
        base_idx = associated_legendre_index(n, 0)
        output[out_elem_idx] += expansion_data[exp_elem_idx, exp_idx] * base_data[base_elem_idx, base_idx]

        sign = -1
        for m in range(1, n + 1):
            exp_idx = spherical_expansion_index(n, m)
            exp_idx_neg = spherical_expansion_index(n, -m)
            base_idx = associated_legendre_index(n, m)
            output[out_elem_idx] += (expansion_data[exp_elem_idx, exp_idx] + expansion_data[exp_elem_idx, exp_idx_neg] * sign) * base_data[base_elem_idx, base_idx]
            sign = - sign


def spherical_harmonics_contraction(expansion_data, legendre_data, phase_data, out=None):
    output_shape, expansion_shape, legendre_shape, phase_shape = broadcast_shapes(
        expansion_data.shape[:-1], legendre_data.shape[:-1], phase_data.shape
    )

    base_order = int((8 * legendre_data.shape[-1] + 1)**0.5 - 3) // 2
    expansion_order = int(expansion_data.shape[-1] ** 0.5) - 1
    output_order = min(base_order, expansion_order)

    if out is None:
        out = np.zeros(output_shape, complex)
    elif out.shape != output_shape:
        raise ValueError(f'Cannot use array of shape {out.shape} as output array for contraction between expansion with shape {expansion_shape}, legendre bases with shape {legendre_shape}, and azimuth phases with shape {phase_shape}')

    if not np.iscomplexobj(expansion_data):
        expansion_data = expansion_data + 0j

    cdef:
        double complex[:, :] exp_cy = expansion_data.reshape((-1, expansion_data.shape[-1]))
        double[:, :] legendre_cy = legendre_data.reshape((-1, legendre_data.shape[-1]))
        double complex[:] out_cy = out.reshape(-1), phase_cy = phase_data.reshape(-1)
        Py_ssize_t N = output_order

    if out.size == 1:
        # No loop over elements, saves notable time due to stride calculations.
        spherical_harmonics_contraction_calculation(out_cy, exp_cy, legendre_cy, phase_cy, 0, 0, 0, 0, N)
        out /= (2 * np.pi)**0.5
        return out

    cdef:
        Py_ssize_t[:] exp_stride = prepare_strides(expansion_shape, output_shape)
        Py_ssize_t[:] legendre_stride = prepare_strides(legendre_shape, output_shape)
        Py_ssize_t[:] phase_stride = prepare_strides(phase_shape, output_shape)
        Py_ssize_t[:] out_stride = prepare_strides(output_shape, output_shape)
        Py_ssize_t exp_elem_idx, legendre_elem_idx, phase_elem_idx, out_elem_idx
        Py_ssize_t num_elements = out.size, ndim = out.ndim

    with nogil:
        for out_elem_idx in range(num_elements):
            exp_elem_idx = broadcast_index(out_elem_idx, exp_stride, out_stride, ndim)
            legendre_elem_idx = broadcast_index(out_elem_idx, legendre_stride, out_stride, ndim)
            phase_elem_idx = broadcast_index(out_elem_idx, phase_stride, out_stride, ndim)
            spherical_harmonics_contraction_calculation(out_cy, exp_cy, legendre_cy, phase_cy, out_elem_idx, exp_elem_idx, legendre_elem_idx, phase_elem_idx, N)

    out /= (2 * np.pi)**0.5
    return out


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void spherical_harmonics_contraction_calculation(
    double complex[:] output,
    double complex[:, :] expansion_data,
    double[:, :] legendre_data,
    double complex[:] phase_data,
    Py_ssize_t out_elem_idx,
    Py_ssize_t exp_elem_idx,
    Py_ssize_t legendre_elem_idx,
    Py_ssize_t phase_elem_idx,
    Py_ssize_t N
) nogil:
    cdef:
        Py_ssize_t n, m, exp_idx, exp_idx_neg, legendre_idx
        double complex phase_power, positive_partial, negative_partial
        int sign

    for n in range(N + 1):
        exp_idx = spherical_expansion_index(n, 0)
        legendre_idx = associated_legendre_index(n, 0)
        output[out_elem_idx] = output[out_elem_idx] + expansion_data[exp_elem_idx, exp_idx] * legendre_data[legendre_elem_idx, legendre_idx]

    sign = -1
    for m in range(1, N + 1):
        phase_power = phase_data[phase_elem_idx] ** m
        positive_partial = 0.
        negative_partial = 0.
        for n in range(m, N + 1):
            exp_idx = spherical_expansion_index(n, m)
            exp_idx_neg = spherical_expansion_index(n, -m)
            legendre_idx = associated_legendre_index(n, m)
            positive_partial = positive_partial + legendre_data[legendre_elem_idx, legendre_idx] * expansion_data[exp_elem_idx, exp_idx]
            negative_partial = negative_partial + legendre_data[legendre_elem_idx, legendre_idx] * expansion_data[exp_elem_idx, exp_idx_neg]
        output[out_elem_idx] = output[out_elem_idx] + positive_partial * phase_power + sign * negative_partial * phase_power.conjugate()
        sign = -sign


def multipole_contraction(expansion_data, radial_data, legendre_data, phase_data, out=None):
    output_shape, expansion_shape, radial_shape, legendre_shape, phase_shape = broadcast_shapes(
        expansion_data.shape[:-1], radial_data.shape[:-1], legendre_data.shape[:-1], phase_data.shape
    )

    base_order = int((8 * legendre_data.shape[-1] + 1)**0.5 - 3) // 2
    expansion_order = int(expansion_data.shape[-1] ** 0.5) - 1
    radial_order = int(radial_data.shape[-1])
    output_order = min(base_order, expansion_order, radial_order)

    if out is None:
        out = np.zeros(output_shape, complex)
    elif out.shape != output_shape:
        raise ValueError(f'Cannot use array of shape {out.shape} as output array for contraction between expansion with shape {expansion_shape}, radial basis with shape {radial_shape}, legendre bases with shape {legendre_shape}, and azimuth phases with shape {phase_shape}')

    if not np.iscomplexobj(expansion_data):
        expansion_data = expansion_data + 0j

    if not np.iscomplexobj(radial_data):
        radial_data = radial_data + 0j

    cdef:
        double complex[:, :] exp_cy = expansion_data.reshape((-1, expansion_data.shape[-1]))
        double[:, :] legendre_cy = legendre_data.reshape((-1, legendre_data.shape[-1]))
        double complex[:, :] radial_cy = radial_data.reshape((-1, radial_data.shape[-1]))
        double complex[:] phase_cy = phase_data.reshape(-1)
        double complex[:] out_cy = out.reshape(-1)
        Py_ssize_t N = output_order

    if out.size == 1:
        # No loop over elements, saves notable time due to stride calculations.
        multipole_contraction_calculation(out_cy, exp_cy, radial_cy, legendre_cy, phase_cy, 0, 0, 0, 0, 0, N)
        out /= (2 * np.pi)**0.5
        return out

    cdef:
        Py_ssize_t[:] exp_stride = prepare_strides(expansion_shape, output_shape)
        Py_ssize_t[:] legendre_stride = prepare_strides(legendre_shape, output_shape)
        Py_ssize_t[:] phase_stride = prepare_strides(phase_shape, output_shape)
        Py_ssize_t[:] radial_stride = prepare_strides(radial_shape, output_shape)
        Py_ssize_t[:] out_stride = prepare_strides(output_shape, output_shape)
        Py_ssize_t exp_elem_idx, legendre_elem_idx, phase_elem_idx, radial_elem_idx, out_elem_idx
        Py_ssize_t num_elements = out.size, ndim = out.ndim

    with nogil:
        for out_elem_idx in range(num_elements):
            exp_elem_idx = broadcast_index(out_elem_idx, exp_stride, out_stride, ndim)
            legendre_elem_idx = broadcast_index(out_elem_idx, legendre_stride, out_stride, ndim)
            phase_elem_idx = broadcast_index(out_elem_idx, phase_stride, out_stride, ndim)
            radial_elem_idx = broadcast_index(out_elem_idx, radial_stride, out_stride, ndim)
            multipole_contraction_calculation(
                out_cy, exp_cy, radial_cy, legendre_cy, phase_cy, 
                out_elem_idx, exp_elem_idx, radial_elem_idx, legendre_elem_idx, phase_elem_idx, N
            )

    out /= (2 * np.pi)**0.5
    return out


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void multipole_contraction_calculation(
    double complex[:] output,
    double complex[:, :] expansion_data, 
    double complex[:, :] radial_data,
    double[:, :] legendre_data,
    double complex[:] phase_data,
    Py_ssize_t out_elem_idx,
    Py_ssize_t exp_elem_idx, 
    Py_ssize_t radial_elem_idx,
    Py_ssize_t legendre_elem_idx,
    Py_ssize_t phase_elem_idx,
    Py_ssize_t N
) nogil:
    cdef:
        Py_ssize_t n, m, exp_idx, exp_idx_neg, legendre_idx
        double complex phase_power, positive_partial, negative_partial
        int sign

    for n in range(N + 1):
        exp_idx = spherical_expansion_index(n, 0)
        legendre_idx = associated_legendre_index(n, 0)
        output[out_elem_idx] = output[out_elem_idx] + expansion_data[exp_elem_idx, exp_idx] * legendre_data[legendre_elem_idx, legendre_idx] * radial_data[radial_elem_idx, n]

    sign = -1
    for m in range(1, N + 1):
        phase_power = phase_data[phase_elem_idx] ** m
        positive_partial = 0.
        negative_partial = 0.
        for n in range(m, N + 1):
            exp_idx = spherical_expansion_index(n, m)
            exp_idx_neg = spherical_expansion_index(n, -m)
            legendre_idx = associated_legendre_index(n, m)
            positive_partial = positive_partial + legendre_data[legendre_elem_idx, legendre_idx] * expansion_data[exp_elem_idx, exp_idx] * radial_data[radial_elem_idx, n]
            negative_partial = negative_partial + legendre_data[legendre_elem_idx, legendre_idx] * expansion_data[exp_elem_idx, exp_idx_neg] * radial_data[radial_elem_idx, n]
        output[out_elem_idx] = output[out_elem_idx] + positive_partial * phase_power + sign * negative_partial * phase_power.conjugate()
        sign = -sign
