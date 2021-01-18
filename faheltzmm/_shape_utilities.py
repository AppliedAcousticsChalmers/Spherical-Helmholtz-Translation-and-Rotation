import numpy as np


def is_value(x):
    return x is not None and type(x) is not np.broadcast


def broadcast_shapes(*shapes, output="new"):
    """Apply boradcasting rules to shape tuples.

    Given a set of shape tuples, checks if and how arrays of those shapes will broadcast.
    With `output="new"` the function returns the shape of the new array with results.
    With `output="reshape"` the function returns the shapes that the arrays can be broadcast to prior to the operation

    Raises ValueError if the shapes are not compatible.

    """
    ndim = max(len(s) for s in shapes)
    padded_shapes = [(1,) * (ndim - len(s)) + s for s in shapes]
    out_shape = [max(s) for s in zip(*padded_shapes)]
    if not all([dim == 1 or dim == out_dim for dims, out_dim in zip(zip(*padded_shapes), out_shape) for dim in dims]):
        raise ValueError(f"Shapes {shapes} cannot be broadcast together")
    if output == 'new':
        return tuple(out_shape)
    elif output == 'reshape':
        return padded_shapes

