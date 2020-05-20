"""Implementations for indexing schemes of basic expansion coefficients or spherical basis functions."""

import numpy as np
import itertools


def expansions(x, scheme, new=None):
    """Indexing and conversions of basic expansion coefficients.

    There are multiple indexing schemes available for plain spherical harmonics
    or expansion coefficients. Below is a description of how the (order, mode)
    is mapped to indices in a matrix A[].

    - Full
        This scheme is indexed with A[n,m] for all indices.
        This relies on python reverse indexing for m<0, i.e. A[n,-|m|].
        Uses a (N+1, 2N+1) matrix to store the (N+1)^2 values, i.e the sparsity goes to 50%.
    - Compact
        This scheme is indexed with A[n,m] for m>=0, and A[|m|-1, n] for m<0 (or A[-(|m|+1), n]),
        i.e. the negative coefficients for a given order are stored in a single column.
    - Linear
        This scheme stores the coefficients in a 1D array of length (N+1)^2, which is indexed
        with A[n^2 + n + m].

    This function has three distinct modes, controlled by the inputs.
    - Convert
        This mode is activated by passing an ndarray as the first argument.
        The input ndarray fill be converted from the `original` indexing scheme
        to the `new` indexing scheme.
    - Indices
        This is used when the first argument is an integer, and the other two arguments are string with schemes.
        The output will be two nested lists, (n, m) which can be sed to index an array of the `scheme` scheme
        to convert it to the `new` scheme. Not that this simple indexing convertion cannot convert to the full scheme.
    - Display
        This mode is activated by passing an order as the first argument, a scheme as the second argument,
        and no third argument. The output will be an ndarray with tuples (n,m) showing
        how the scheme is organized.

    Parameters
    ----------
    x: int or ndarray
        If an int is passed it is used as the maximum order for the conversion, and an indexing convertion scheme is returned.
        If a ndarray is passed it will be converted from the original scheme to the new scheme.
    scheme: string
        The original scheme, one of "full", "compact", "linear".
        Specifies the scheme of the original data, either the input ndarray x,
        or the ndarray which should be indexed with the conversion scheme.
    new: string
        The new scheme, one of "full", "compact", "linear".
        Selects the new scheme for the data. Leave this out to display the `scheme` up to order `x`.
    """
    if type(x) is int:
        if new is None:
            return show_scheme(x, scheme)
        if 'full' in scheme.lower():
            if 'full' in new.lower():
                return slice(None), slice(None)
            if 'compact' in new.lower():
                return compact(x)
            if 'linear' in new.lower():
                return linear(x)
            if 'sectorial' in new.lower():
                if 'positive' in new.lower():
                    return positive_sectorial(x)
                if 'negative' in new.lower():
                    return negative_sectorial(x)
                return sectorial(x)
            if 'tesseral' in new.lower():
                if 'positive' in new.lower():
                    return positive_tesseral(x)
                if 'negative' in new.lower():
                    return negative_tesseral(x)
                return tesseral(x)
            if 'zonal' in new.lower():
                return zonal(x)
            if 'nonnegative' in new.lower():
                return nonnegative(x)
            if 'nonpositive' in new.lower():
                return nonpositive(x)
            if 'positive' in new.lower():
                return positive(x)
            if 'negative' in new.lower():
                return negative(x)
        if 'compact' in scheme.lower():
            if 'full' in new.lower():
                raise ValueError('No simple indexing scheme possible to convert from compact to full')
            if 'compact' in new.lower():
                return slice(None), slice(None)
            return compact_indices(*expansions(x, 'full', new))
        if 'linear' in scheme.lower():
            if 'full' in new.lower():
                raise ValueError('No simple indexing scheme possible to convert from compact to linear')
            if 'linear' in new.lower():
                return slice(None)
            return linear_indices(*expansions(x, 'full', new))
    if 'full' in scheme.lower() and 'compact' in new.lower():
        return convert_full_2_compact(x)
    if 'full' in scheme.lower() and 'linear' in new.lower():
        return convert_full_2_linear(x)
    if 'compact' in scheme.lower() and 'full' in new.lower():
        return convert_compact_2_full(x)
    if 'compact' in scheme.lower() and 'linear' in new.lower():
        return convert_compact_2_linear(x)
    if 'linear' in scheme.lower() and 'compact' in new.lower():
        return convert_linear_2_compact(x)
    if 'linear' in scheme.lower() and 'full' in new.lower():
        return convert_linear_2_full(x)
    if scheme.lower() == new.lower():
        return x
    raise ValueError(f'Unknown indexing schemes, {scheme} and {new}')


def show_scheme(order, form='full'):
    """Show an indexing form.

    Creates a numpy array with tuples (n, m) to show how the expansion coefficients
    are organized for a specific scheme.
    """
    full = np.full((order + 1, 2 * order + 1), None, object)
    for n in range(order + 1):
        full[n, 0] = (n, 0)
        for m in range(1, n + 1):
            full[n, m] = (n, m)
            full[n, -m] = (n, -m)
    if 'full' in form.lower():
        return full
    if 'compact' in form.lower():
        return full[compact(order)]
    if 'linear' in form.lower():
        return full[linear(order)]


def full_indices(n, m):
    """Find the indices of coefficients of order n and mode m in the "full" scheme."""
    return n, m


def compact_indices(n, m):
    """Find the indices of coefficients of order n and mode m in the "compact" scheme."""
    return np.where(m >= 0, n, - m - 1), np.where(m >= 0, m, n)


def linear_indices(n, m):
    """Find the indices of coefficients of order n and mode m in the "linear" scheme."""
    return n**2 + n + m


def compact(order):
    """Create compact coefficient scheme."""
    return (
        np.array([[row] * (row + 1) + list(range(row + 1, order + 1)) for row in range(order + 1)], dtype=int),
        np.array([list(range(row + 1)) + [-(row + 1)] * (order - row) for row in range(order + 1)], dtype=int)
    )


def linear(order):
    """Create linear coefficient scheme."""
    return (
        np.array([n for n in range(order + 1) for m in range(-n, n + 1)], dtype=int),
        np.array([m for n in range(order + 1) for m in range(-n, n + 1)], dtype=int)
    )


def positive(order):
    """Create positive modes only linear coefficient scheme."""
    return (
        np.array([n for n in range(order + 1) for m in range(1, n + 1)], dtype=int),
        np.array([m for n in range(order + 1) for m in range(1, n + 1)], dtype=int),
    )


def negative(order):
    """Create negative modes only linear coefficient scheme."""
    return (
        np.array([n for n in range(order + 1) for m in range(1, n + 1)], dtype=int),
        np.array([-m for n in range(order + 1) for m in range(1, n + 1)], dtype=int),
    )


def nonnegative(order):
    """Create non-negative modes only linear coefficient scheme."""
    return (
        np.array([n for n in range(order + 1) for m in range(n + 1)], dtype=int),
        np.array([m for n in range(order + 1) for m in range(n + 1)], dtype=int),
    )


def nonpositive(order):
    """Create non-positive modes only linear coefficient scheme."""
    return (
        np.array([n for n in range(order + 1) for m in range(n + 1)], dtype=int),
        np.array([-m for n in range(order + 1) for m in range(n + 1)], dtype=int),
    )


def zonal(order):
    """Create zonal modes only linear coefficient scheme."""
    return (np.arange(order + 1), np.zeros(order + 1, int))


def sectorial(order):
    """Create sectorial modes only linear coefficient scheme."""
    return (
        np.array([n for n in range(order + 1) for m in [-n, n]], dtype=int),
        np.array([m for n in range(order + 1) for m in [-n, n]], dtype=int),
    )


def positive_sectorial(order):
    """Create positive sectorial modes only linear coefficient scheme."""
    return (np.arange(1, order + 1, dtype=int), np.arange(1, order + 1, dtype=int))


def negative_sectorial(order):
    """Create negative sectorial modes only linear coefficient scheme."""
    return (np.arange(1, order + 1, dtype=int), -np.arange(1, order + 1, dtype=int))


def tesseral(order):
    """Create tesseral modes only linear coefficient scheme."""
    return (
        np.array([n for n in range(order + 1) for m in itertools.chain(range(-n + 1, 0), range(1, n))], dtype=int),
        np.array([m for n in range(order + 1) for m in itertools.chain(range(-n + 1, 0), range(1, n))], dtype=int)
    )


def positive_tesseral(order):
    """Create positive tesseral modes only linear coefficient scheme."""
    return (
        np.array([n for n in range(order + 1) for m in range(1, n)], dtype=int),
        np.array([m for n in range(order + 1) for m in range(1, n)], dtype=int)
    )


def negative_tesseral(order):
    """Create negative tesseral modes only linear coefficient scheme."""
    return (
        np.array([n for n in range(order + 1) for m in range(1, n)], dtype=int),
        np.array([-m for n in range(order + 1) for m in range(1, n)], dtype=int)
    )


def convert_full_2_compact(A):
    """Convert array from full form to compact form."""
    if 2 * A.shape[0] - 1 == A.shape[1]:
        return A[expansions(A.shape[0] - 1, 'full', 'compact')]
    else:
        raise ValueError(f"Cannot convert full form to compact with max order {A.shape[0] - 1} and max mode {(A.shape[1] - 1) // 2}")


def convert_full_2_linear(A):
    """Convert array from full form to linear form."""
    if 2 * A.shape[0] - 1 == A.shape[1]:
        return A[expansions(A.shape[0] - 1, 'full', 'linear')]
    else:
        raise ValueError(f"Cannot convert full form to linear with max order {A.shape[0] - 1} and max mode {(A.shape[1] - 1) // 2}")


def convert_compact_2_full(A):
    """Convert array from compact form to full form."""
    orders = A.shape[0] - 1
    modes = A.shape[0] - 1
    if orders != modes:
        raise ValueError(f"Invalid compact form with max order {orders} and max mode {modes}")
    A_full = np.zeros((orders + 1, 2 * modes + 1), dtype=A.dtype)
    A_full[expansions(orders, 'full', 'compact')] = A
    return A_full


def convert_compact_2_linear(A):
    """Convert array from compact form to linear form."""
    orders = A.shape[0] - 1
    modes = A.shape[0] - 1
    if orders != modes:
        raise ValueError(f"Invalid compact form with max order {orders} and max mode {modes}")
    return A[expansions(orders, 'compact', 'linear')]


def convert_linear_2_full(A):
    """Convert array from linear form to full form."""
    orders = int(A.shape[0] ** 0.5) - 1
    modes = orders
    if (orders + 1) ** 2 != A.shape[0]:
        raise ValueError(f"Cannot convert linear form to full using {A.shape[0]} components")
    A_full = np.zeros((orders + 1, 2 * modes + 1), dtype=A.dtype)
    A_full[expansions(orders, 'full', 'linear')] = A
    return A_full


def convert_linear_2_compact(A):
    """Convert array from linear form to compact form."""
    orders = int(A.shape[0] ** 0.5) - 1
    if (orders + 1) ** 2 != A.shape[0]:
        raise ValueError(f"Cannot convert linear output to full using {A.shape[0]} components")
    return A[expansions(orders, 'linear', 'compact')]
