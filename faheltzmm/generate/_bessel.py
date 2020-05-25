"""Implementations for the spherical bessel functions.

Currently this is based on the implementations for spherical bessel functions
in scipy. The disadvantage with the scipy implementations is that any recurrence
relations are used for each order input. Since we typically need a set of
arguments evaluted for all orders, it would be more efficient to only apply
the recurrence relations ones to get all orders.

It is preferrable to use the functions here for their intended purposes, since
they might be re-implemented to suit our specialized needs at some point.
"""

import numpy as np
from scipy.special import spherical_jn, spherical_yn


def spherical_hn(n, z, derivative=False):
    """Calculate the spherical Hankel function.

    This is a convenience function to calculate

    .. math::

        h_n(z) = j_n(z) + j y_n(z)

    using the scipy implementations of the above.
    Parameters and outputs as `scipy.special.spherical_jn`, and `scipy.special.spherical_yn`,
    which can be accessed as `spherical_jn`, and `spherical_yn` from this module.
    """
    return spherical_jn(n, z, derivative=derivative) + 1j * spherical_yn(n, z, derivative=derivative)


def spherical_jn_all(max_order, z, derivative=False):
    """Calculate all spherical Bessel functions up tp a given maximum order."""
    return spherical_jn(np.arange(max_order + 1).reshape([-1] + [1] * np.ndim(z)), z, derivative=derivative)


def spherical_yn_all(max_order, z, derivative=False):
    """Calculate all spherical Neumann functions up tp a given maximum order."""
    return spherical_yn(np.arange(max_order + 1).reshape([-1] + [1] * np.ndim(z)), z, derivative=derivative)


def spherical_hn_all(max_order, z, derivative=False):
    """Calculate all spherical Hankel functions up tp a given maximum order."""
    return spherical_hn(np.arange(max_order + 1).reshape([-1] + [1] * np.ndim(z)), z, derivative=derivative)
