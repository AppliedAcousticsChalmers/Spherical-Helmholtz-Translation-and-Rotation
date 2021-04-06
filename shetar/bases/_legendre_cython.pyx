import numpy as np
cimport cython


def legendre_polynomials(x, order=None, out=None):
    if order is None and out is None:
        raise ValueError('Cannot calculate Legendre polynomians without receiving either the output arrey or the order')
    output_shape = np.shape(x)
    num_elements = np.size(x)

    if order is None:
        order = out.shape[0] - 1
    if out is None:
        out = np.zeros((order + 1,) + output_shape)

    out[0] = 1 / 2**0.5
    if order > 1:
        out[1] = x * 1.5**0.5

    if order >= 2:
        legendre_recurrence(x.reshape(-1), out.reshape((order + 1, num_elements)), order, num_elements)

    return out


@cython.boundscheck(False)
@cython.cdivision(True)
cdef void legendre_recurrence(double[:] x, double[:, :] data, int order, int num_elements) nogil:
    cdef:
        int idx, n
        double n_minus_1_factor, n_minus_2_factor,

    for n in range(2, order + 1):
        n_minus_1_factor = ((2 * n + 1) * (2 * n - 1))**0.5 / n
        n_minus_2_factor = <double>(n - 1) / n * (<double>(2 * n + 1) / (2 * n - 3))**0.5
        for idx in range(num_elements):
            data[n, idx] = x[idx] * data[n - 1, idx] * n_minus_1_factor - data[n - 2, idx] * n_minus_2_factor


def associated_legendre_polynomials(x, order=None, out=None):
    if order is None and out is None:
        raise ValueError('Cannot calculate Legendre polynomians without receiving either the output arrey or the order')
    output_shape = np.shape(x)
    num_elements = np.size(x)

    if order is None:
        num_unique = out.shape[0]
        order = int((8 * num_unique + 1)**0.5 - 3) // 2
    if out is None:
        num_unique = (order + 1) * (order + 2) // 2
        out = np.zeros((num_unique,) + output_shape)

    out[0] = 1 / 2**0.5

    if order >= 1:
        one_minus_x_square = 1 - x**2
        associated_legendre_recurrence(
            x.reshape(num_elements),
            one_minus_x_square.reshape(num_elements),
            out.reshape((num_unique, num_elements)),
            order, num_elements
        )

    return out

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void associated_legendre_recurrence(double[:] x, double[:] one_minus_x_square, double[:, :] data, int order, int num_elements) nogil:
    cdef:
        int idx_elem, n, m, idx_assign, idx_m1, idx_m2
        double fact_m1, fact_m2

    for n in range(1, order + 1):
            # Recurrence to higher orders
            idx_assign = associated_legendre_index(n, n)
            idx_m1 = associated_legendre_index(n - 1, n - 1)
            fact_m1 = - (<double>(2 * n + 1) / (2 * n))**0.5
            for idx_elem in range(num_elements):
                data[idx_assign, idx_elem] = fact_m1 * data[idx_m1, idx_elem]

            # Same recurrence as below, but excluding the mode+2 part explicitly.
            idx_assign = associated_legendre_index(n, n - 1)
            idx_m1 = associated_legendre_index(n, n)
            fact_m1 = - (2 * n)**0.5
            for idx_elem in range(num_elements):
                data[idx_assign, idx_elem] = fact_m1 * data[idx_m1, idx_elem] * x[idx_elem]

            for m in range(n - 2, -1, -1):  # [n - 2, n - 3, ... 0]
                # Recurrece to lower modes
                idx_assign = associated_legendre_index(n, m)
                idx_m1 = associated_legendre_index(n, m + 1)
                idx_m2 = associated_legendre_index(n, m + 2)
                fact_m1 = - 2 * <double>(m + 1) / ((n + m + 1) * (n - m)) ** 0.5
                fact_m2 = -( <double>((n + m + 2) * (n - m - 1)) / ((n - m) * (n + m + 1))) ** 0.5
                for idx_elem in range(num_elements):
                    data[idx_assign, idx_elem] = (
                        fact_m1 * data[idx_m1, idx_elem] * x[idx_elem]
                        + fact_m2 * data[idx_m2, idx_elem] * one_minus_x_square[idx_elem]
                    )


@cython.boundscheck(False)
@cython.cdivision(True)
cdef int associated_legendre_index(int order, int mode) nogil:
    return (order * (order + 1)) / 2 + mode
