"""Module to handle coordinate transforms."""

import numpy as np


def cartesian_2_spherical(cartesian_positions):
    """Convert a cartesian position to the spherical variables.

    Parameters
    ----------
    cartesian_positions: array_like
        A 3x... shape array with the cartesian positions [x, y, z] in the
        first axis.

    Returns
    -------
    spherical_coordinates : np.ndarray
        A 3x... shape array with the first axis corresponding to the radius,
        the colatitude angle, and the azimuth angle.
    """
    cartesian_positions = np.asarray(cartesian_positions)
    r = np.sum(cartesian_positions**2, axis=0)**0.5
    theta = np.where(r == 0, 0, np.arccos(np.clip(cartesian_positions[2] / r, -1., 1.)))
    phi = np.arctan2(cartesian_positions[1], cartesian_positions[0])
    return np.stack([r, theta, phi], axis=0)


def spherical_2_cartesian(radius, colatitude, azimuth):
    """Convert spherical coordinates to cartesian positions.

    Parameters
    ----------
    spherical_coordinates : array_like
        A 3x... shape array with the first axis corresponding to the radius,
        the colatitude angle, and the azimuth angle.

    Returns
    -------
    cartesian_positions: np.ndarray
        A 3x... shape array with the cartesian positions [x, y, z] in the
        first axis.
    """
    x = np.sin(colatitude) * np.cos(azimuth)
    y = np.sin(colatitude) * np.sin(azimuth)
    z = np.cos(colatitude)
    return radius * np.stack([x, y, z], axis=0)
