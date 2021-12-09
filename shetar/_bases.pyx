import numpy as np
cimport cython
from cython.parallel import prange
from ._shapes import prepare_strides, broadcast_shapes
from ._shapes cimport broadcast_index


def legendre_polynomials(x, order=None, out=None):
    if order is None and out is None:
        raise ValueError('Cannot calculate Legendre polynomials without receiving either the output array or the order')
    output_shape = np.shape(x)

    if order is None:
        order = out.shape[-1] - 1
    if out is None:
        out = np.zeros(output_shape + (order + 1,))

    out[..., 0] = 1 / 2**0.5
    if order >= 1:
        out[..., 1] = x * 1.5**0.5

    cdef:
        int idx, n, N = order, num_elements = np.size(x)
        double n_minus_1_factor, n_minus_2_factor
        double[:] x_cy = x.reshape(-1)
        double[:, :] data = out.reshape((np.size(x), order + 1))

    with cython.boundscheck(False), cython.cdivision(True), cython.wraparound(False), nogil:
        for idx in prange(num_elements):
            for n in range(2, N + 1):
                n_minus_1_factor = ((2 * n + 1) * (2 * n - 1))**0.5 / n
                n_minus_2_factor = <double>(n - 1) / n * (<double>(2 * n + 1) / (2 * n - 3))**0.5
                data[idx, n] = x_cy[idx] * data[idx, n - 1] * n_minus_1_factor - data[idx, n - 2] * n_minus_2_factor

    return out


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
        Py_ssize_t n, N = output_order

    if out.ndim == 0:
        # No loop over elements.
        with cython.boundscheck(False), nogil:
            for n in range(N):
                out_cy[0] += exp_cy[0, n] * base_cy[0, n]
        return out

    cdef:
        Py_ssize_t[:] exp_stride = prepare_strides(expansion_shape, output_shape)
        Py_ssize_t[:] base_stride = prepare_strides(base_shape, output_shape)
        Py_ssize_t[:] out_stride = prepare_strides(output_shape, output_shape)
        Py_ssize_t exp_idx, base_idx, out_idx
        Py_ssize_t num_elements = out.size, ndim = out.ndim
    
    with cython.boundscheck(False), cython.wraparound(False), cython.cdivision(True), nogil:
        for out_idx in prange(num_elements):
            exp_idx = broadcast_index(out_idx, exp_stride, out_stride, ndim)
            base_idx = broadcast_index(out_idx, base_stride, out_stride, ndim)
            for n in range(N):
                out_cy[out_idx] += exp_cy[exp_idx, n] * base_cy[base_idx, n]

    return out


def associated_legendre_polynomials(x, order=None, out=None):
    if order is None and out is None:
        raise ValueError('Cannot calculate associated Legendre polynomials without receiving either the output array or the order')
    output_shape = np.shape(x)

    if order is None:
        num_unique = out.shape[0]
        order = int((8 * num_unique + 1)**0.5 - 3) // 2
    if out is None:
        num_unique = (order + 1) * (order + 2) // 2
        out = np.zeros(output_shape + (num_unique,))

    out[..., 0] = 1 / 2**0.5

    cdef:
        int idx_elem, n, m, idx_assign, idx_m1, idx_m2
        int N = order, num_elements = np.size(x)
        double fact_m1, fact_m2
        double[:] x_cy = x.reshape(-1)
        double[:] one_minus_x_square = 1 - x.reshape(-1)**2
        double[:, :] data = out.reshape((-1, num_unique))

    with cython.boundscheck(False), cython.cdivision(True), cython.wraparound(False), nogil:
        for idx_elem in prange(num_elements):
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
                data[idx_elem, idx_assign] = fact_m1 * data[idx_elem, idx_m1] * x_cy[idx_elem]

                for m in range(n - 2, -1, -1):  # [n - 2, n - 3, ... 0]
                    # Recurrece to lower modes
                    idx_assign = associated_legendre_index(n, m)
                    idx_m1 = associated_legendre_index(n, m + 1)
                    idx_m2 = associated_legendre_index(n, m + 2)
                    fact_m1 = - 2 * <double>(m + 1) / ((n + m + 1) * (n - m)) ** 0.5
                    fact_m2 = -(<double>((n + m + 2) * (n - m - 1)) / ((n - m) * (n + m + 1))) ** 0.5
                    data[idx_elem, idx_assign] = (
                        fact_m1 * data[idx_elem, idx_m1] * x_cy[idx_elem]
                        + fact_m2 * data[idx_elem, idx_m2] * one_minus_x_square[idx_elem]
                    )
                    # Modify the (n, m+2) components to have the correct normalization.
                    # This cannot be done earlier since the recurrence is not stable if this scale is included!
                    data[idx_elem, idx_m2] *= one_minus_x_square[idx_elem] ** (0.5 * m + 1)
                # Normalization for (n, 1) components. (n, 0) needs no modification.
                data[idx_elem, idx_m1] *= one_minus_x_square[idx_elem] ** 0.5

    return out


@cython.boundscheck(False)
@cython.cdivision(True)
cdef int associated_legendre_index(int order, int mode) nogil:
    return (order * (order + 1)) / 2 + mode
