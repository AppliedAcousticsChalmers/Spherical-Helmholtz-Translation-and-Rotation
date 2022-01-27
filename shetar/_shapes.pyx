import numpy as np
import cython
cimport cython


def prepare_strides(input_shape, target_shape):
    ndim = len(target_shape)
    strides = np.zeros(ndim, int)
    step = 1
    for idx in range(ndim - 1, -1, -1):
        if input_shape[idx] == target_shape[idx]:
            strides[idx] = step
            step *= target_shape[idx]
    return strides


def broadcast_shapes(*shapes, min_dims=None):
    ndim = max(len(s) for s in shapes)
    if min_dims is not None:
        ndim = max(min_dims, ndim)
    padded_shapes = [(1,) * (ndim - len(s)) + tuple(s) for s in shapes]
    out_shape = [max(s) for s in zip(*padded_shapes)]
    if not all([dim == 1 or dim == out_dim for dims, out_dim in zip(zip(*padded_shapes), out_shape) for dim in dims]):
        raise ValueError(f"Shapes {shapes} cannot be broadcast together")
    return tuple(out_shape), *padded_shapes


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef Py_ssize_t broadcast_index(Py_ssize_t broadcasted_index, Py_ssize_t[:] local_strides, Py_ssize_t[:] broadcast_strides, Py_ssize_t ndim) nogil:
    cdef:
        Py_ssize_t index = 0, dim, multi_index
    for dim in range(ndim):
        multi_index = broadcasted_index // broadcast_strides[dim]
        broadcasted_index -= multi_index * broadcast_strides[dim]
        index += local_strides[dim] * multi_index
    return index
