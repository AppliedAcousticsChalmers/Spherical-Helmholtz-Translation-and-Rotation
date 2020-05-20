"""Implementations for indexing schemes of basic expansion coefficients or spherical basis functions."""

import numpy as np
import itertools


def expansions(x, scheme, new=None):
    """Indexing and conversions of basic expansion coefficients.

    There are multiple indexing schemes available for plain spherical harmonics
    or expansion coefficients. Below is a description of how the (order, mode)
    is mapped to indices in a matrix A[].

    This function has three distinct modes, controlled by the inputs.
    - Convert
        This mode is activated by passing an ndarray as the first argument.
        The input ndarray fill be converted from the `original` indexing scheme
        to the `new` indexing scheme.
        In the "convert" mode, both `scheme` and `new` has to be given,
        and both have to be one of "natural", 'compact" or "linear".
    - Indices
        This is used when the first argument is an integer, and the other two arguments are strings with schemes.
        The output will be two nested lists, (n, m) which can be sed to index an array of the `scheme` scheme
        to convert it to the `new` scheme. Note that this simple indexing convertion cannot convert to the natural scheme.
        In this mode, `scheme` has to be one of "natural", "compact" or "linear", while "new" can be one of the subset schemes.
    - Display
        This mode is activated by passing an order as the first argument, a scheme as the second argument,
        and no third argument. The output will be an ndarray with tuples (n,m) showing
        how the scheme is organized. In this mode, "scheme" can be any of the schemes.

    Available schemes
    -----------------
    In the following description [..] is used to indicate indexing of an ndarray, and (n,m)
    denotes the order n and mode m ot a coefficient or spherical basis.
    - Natural
        This scheme is indexed with [n,m] -> (n,m) for all indices.
        This relies on python reverse indexing for m<0, i.e. [n,-|m|] -> (n,m) for m<0.
        Uses a (N+1, 2N+1) matrix to store the (N+1)^2 values, i.e the sparsity goes to 50%.
    - Compact
        This scheme stores the positive modes in the lower triangular half
        of the array, i.e. [n,m] -> (n,m) for m>=0.
        The negative coefficients for a given order are stored in a single column,
        i.e. [n,m] -> (m,-n-1) for m>n, or [-m-1, n] -> (n,m) for m<0.
    - Natural (or compact) nonnegative (or positive)
        The same as natural and compact, but only for m>= 0
    - Linear
        This scheme stores the coefficients in a 1D array of length (N+1)^2,
        which is indexed as [n^2 + n + m] -> (n,m) or [idx] -> (n=floor(sqrt(idx)), m=idx - n^2 - n).
    - Zonal
        This includes only the m=0 coefficients, stored as [n] -> (n, 0).
    - Positive
        This includes only coefficients where m>0, stored as [(1,1),(2,1),(2,2),(3,1)...
    - Negative
        This includes only coefficients where m<0, stored as [(1,-1),(2,-1),(2,-2),(3,-1)...
    - Non-negative
        This includes only coefficients where m>=0, stored as [(0,0),(1,0),(1,1),(2,0),(2,1)...
    - Non-positive
        This includes only coefficients where m<=0, stored as [(0,0),(1,0),(1,-1),(2,0),(2,-1)...
    - Sectorial
        This includes coefficients where n=|m|, stored as [(0,0),(1,-1),(1,1),(2,-2),(2,2)...
    - Positive sectorial
        This includes coefficients where n=m>0, stored as [(1,1),(2,2),(3,3)...
    - Negative sectorial
        This includes coefficients where n=-m>0, stored as [(1,-1),(2,-2),(3,-3)...
    - Tesseral
        This includes coefficients where 0<|m|<n, stored as [(2,-1),(2,1),(3,-2),(3,-1),(3,1),(3,2)...
    - Positive tesseral
        This includes coefficients where 0<m<n, stored as [(2,1),(3,1),(3,2),(4,1)...
    - Negative tesseral
        This includes coefficients where 0<-m<n, stored as [(2,-1),(3,-1),(3,-2),(4,-1)...


    Parameters
    ----------
    x: int or ndarray
        If an int is passed it is used as the maximum order for the conversion, and an indexing convertion scheme is returned.
        If a ndarray is passed it will be converted from the original scheme to the new scheme.
    scheme: string
        Specifies the scheme of the original data, either the input ndarray x,
        or the ndarray which should be indexed with the conversion scheme.
    new: string
        The new scheme. Selects the new scheme for the data.
        Leave this out to display the `scheme` up to order `x`.
    """
    if type(x) is int:
        if new is None:
            return show_scheme(x, scheme)
        if 'natural' in scheme.lower():
            if 'natural' in new.lower():
                if 'positive' in new.lower() or 'nonnegative' in new.lower():
                    return np.arange(x + 1)[:, None], np.arange(x + 1)[None, :]
                return np.arange(x + 1)[:, None], np.concatenate([np.arange(x + 1), np.arange(-x, 0)])[None, :]
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
            if 'positive' in new.lower():
                if 'non' in new.lower():
                    return nonpositive(x)
                return positive(x)
            if 'negative' in new.lower():
                if 'non' in new.lower():
                    return nonnegative(x)
                return negative(x)
        if 'compact' in scheme.lower():
            if 'natural' in new.lower():
                raise ValueError('No simple indexing scheme possible to convert from compact to natural')
            # if 'compact' in new.lower():
                # return np.arange(x + 1)[:, None], np.arange(x + 1)[None, :]
                # return slice(None), slice(None)
            return compact_indices(*expansions(x, 'natural', new))
        if 'linear' in scheme.lower():
            if 'natural' in new.lower():
                raise ValueError('No simple indexing scheme possible to convert from linear to natural')
            # if 'linear' in new.lower():
                # return slice(None)
            return linear_indices(*expansions(x, 'natural', new))
    if 'natural' in scheme.lower() and 'compact' in new.lower():
        return convert_natural_2_compact(x)
    if 'natural' in scheme.lower() and 'linear' in new.lower():
        return convert_natural_2_linear(x)
    if 'compact' in scheme.lower() and 'natural' in new.lower():
        return convert_compact_2_natural(x)
    if 'compact' in scheme.lower() and 'linear' in new.lower():
        return convert_compact_2_linear(x)
    if 'linear' in scheme.lower() and 'compact' in new.lower():
        return convert_linear_2_compact(x)
    if 'linear' in scheme.lower() and 'natural' in new.lower():
        return convert_linear_2_natural(x)
    if scheme.lower() == new.lower():
        return x
    raise ValueError(f'Unknown indexing schemes, {scheme} and {new}')


def show_scheme(order, form='natural'):
    """Show an indexing form.

    Creates a numpy array with tuples (n, m) to show how the expansion coefficients
    are organized for a specific scheme.
    """
    natural = np.full((order + 1, 2 * order + 1), None, object)
    for n in range(order + 1):
        natural[n, 0] = (n, 0)
        for m in range(1, n + 1):
            natural[n, m] = (n, m)
            natural[n, -m] = (n, -m)
    return natural[expansions(order, 'natural', form)]


def natural_indices(n, m):
    """Find the indices of coefficients of order n and mode m in the "natural" scheme.

    The natural scheme is indexed using [n,m] for all n and m.
    This relies on reverse indexing in python and assumes that the array is
    large enough so that there is no overlap between the positive and negative modes.
    """
    return n, m


def compact_indices(n, m):
    """Find the indices of coefficients of order n and mode m in the "compact" scheme.

    This scheme is indexed with [n,m] for m>=0, and for m<0 one of
    [|m|-1, n], [-(|m|+1), n], [-m-1, n], which are equivalent,
    i.e. the negative coefficients for a given order are stored in a single column.
    """
    return np.where(m >= 0, n, - m - 1), np.where(m >= 0, m, n)


def linear_indices(n, m):
    """Find the indices of coefficients of order n and mode m in the "linear" scheme.

    This scheme is indexed with [n^2 + n + m], i.e. the coefficients are all stored
    in a single dimension.
    """
    return n**2 + n + m


def compact(order):
    """Create compact coefficient scheme."""
    return (
        np.array([[row] * (row + 1) + list(range(row + 1, order + 1)) for row in range(order + 1)], dtype=int),
        np.array([list(range(row + 1)) + [-(row + 1)] * (order - row) for row in range(order + 1)], dtype=int)
    )


def linear(order):
    """Create linear coefficient scheme."""
    idx = np.arange((order + 1)**2)
    n = np.floor(idx**0.5).astype(int)
    m = idx - n**2 - n
    return n, m


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
        np.array([0] + [n for n in range(1, order + 1) for m in [-n, n]], dtype=int),
        np.array([0] + [m for n in range(1, order + 1) for m in [-n, n]], dtype=int),
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


def convert_natural_2_compact(A):
    """Convert array from natural form to compact form."""
    if 2 * A.shape[0] - 1 == A.shape[1]:
        return A[expansions(A.shape[0] - 1, 'natural', 'compact')]
    else:
        raise ValueError(f"Cannot convert natural form to compact with max order {A.shape[0] - 1} and max mode {(A.shape[1] - 1) // 2}")


def convert_natural_2_linear(A):
    """Convert array from natural form to linear form."""
    if 2 * A.shape[0] - 1 == A.shape[1]:
        return A[expansions(A.shape[0] - 1, 'natural', 'linear')]
    else:
        raise ValueError(f"Cannot convert natural form to linear with max order {A.shape[0] - 1} and max mode {(A.shape[1] - 1) // 2}")


def convert_compact_2_natural(A):
    """Convert array from compact form to natural form."""
    orders = A.shape[0] - 1
    modes = A.shape[0] - 1
    if orders != modes:
        raise ValueError(f"Invalid compact form with max order {orders} and max mode {modes}")
    A_natural = np.zeros((orders + 1, 2 * modes + 1) + A.shape[2:], dtype=A.dtype)
    A_natural[expansions(orders, 'natural', 'compact')] = A
    return A_natural


def convert_compact_2_linear(A):
    """Convert array from compact form to linear form."""
    orders = A.shape[0] - 1
    modes = A.shape[0] - 1
    if orders != modes:
        raise ValueError(f"Invalid compact form with max order {orders} and max mode {modes}")
    return A[expansions(orders, 'compact', 'linear')]


def convert_linear_2_natural(A):
    """Convert array from linear form to natural form."""
    orders = int(A.shape[0] ** 0.5) - 1
    modes = orders
    if (orders + 1) ** 2 != A.shape[0]:
        raise ValueError(f"Cannot convert linear form to natural using {A.shape[0]} components")
    A_natural = np.zeros((orders + 1, 2 * modes + 1) + A.shape[1:], dtype=A.dtype)
    A_natural[expansions(orders, 'natural', 'linear')] = A
    return A_natural


def convert_linear_2_compact(A):
    """Convert array from linear form to compact form."""
    orders = int(A.shape[0] ** 0.5) - 1
    if (orders + 1) ** 2 != A.shape[0]:
        raise ValueError(f"Cannot convert linear output to natural using {A.shape[0]} components")
    return A[expansions(orders, 'linear', 'compact')]
