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


def broadcast_reshape(*shapes, newshape):
    new_elements = np.prod(newshape)

    def test_factorization(value, factors):
        if value == 1:
            return True
        possible = [factor for factor in factors if value % factor == 0 and factor != 1]
        if len(possible) == 0:
            return False
        for idx, factor in enumerate(possible):
            if test_factorization(value // factor, possible[:idx] + possible[idx + 1:]):
                return True
        else:
            return False

    def find_reshape(shape):
        elements = np.prod(shape)
        if new_elements % elements != 0:
            raise ValueError(f'Cannot reshape size {elements} to align with {newshape}')
        # Values are only possible if theay are divisors of elements.
        possible = [s if elements % s == 0 else 1 for s in newshape]
        # We need to remove a factor of the elements from the possible values.
        downsize = np.prod(possible) // elements
        if downsize == 1:
            return tuple(possible)
        # If the downsizing factor is not divisible with a value, it has to part of the solution shape.
        solution = [s if downsize % s != 0 else 1 for s in possible]
        # After picking all of the required values, we need to add a factor of the possible values again to get back to the target number of elements.
        upsize = elements // np.prod(solution)
        if upsize == 1:
            return tuple(solution)
        # Values are not possible if they are not factors of upsize, or if they have already been used in the solution.
        possible = [p if r == 1 and upsize % p == 0 else 1 for (p, r) in zip(possible, solution)]

        # Pick values and see if we can still factorize the upsize using the remaining possible values.
        # Start with the values in the original shape, they have priority.
        for val in shape:
            if val == 1 or val not in possible:
                continue
            try:
                idx = possible.index(val)
            except ValueError:
                continue
            if test_factorization(upsize // val, possible[:idx] + possible[idx + 1:]):
                # It's possible to factor the upsize including this value.
                solution[idx] = val
                possible[idx] = 1
                upsize //= val
        # Check the remainder of the possible values.
        for idx, val in enumerate(possible):
            if val == 1:
                continue
            if test_factorization(upsize // val, possible[:idx] + possible[idx + 1:]):
                # We can find a factorization of remainig upsize using the current value
                solution[idx] = val
                possible[idx] = 1
                upsize //= val
        return tuple(solution)

    return [find_reshape(shape) for shape in shapes]
