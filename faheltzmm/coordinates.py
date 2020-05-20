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
    with np.errstate(divide='ignore', invalid='ignore'):
        theta = np.where(r == 0, 0, np.arccos(np.clip(cartesian_positions[2] / r, -1., 1.)))
    phi = np.arctan2(cartesian_positions[1], cartesian_positions[0])
    return np.stack([r, theta, phi], axis=0)


def cartesian_2_trigonometric(cartesian_positions):
    r"""Convert a cartesian position to the spherical variables.

    This defines the spherical variable transform

    .. math::
        x = r \sin\theta\cos\phi
        y = r \sin\theta\sin\phi
        z = r \cos\theta

    but instead of calculating the angles, values of the trigonometric functions
    are returned. In many applications is is the trigonometric values that are of
    interest, and it is more efficient to compute them directly.

    Parameters
    ----------
    cartesian_positions: array_like
        A 3x... shape array with the cartesian positions [x, y, z] in the
        first axis.

    Returns
    -------
    spherical_coordinates : np.ndarray
        A 5x... shape array with the first axis corresponding to the radius,
        :math:`\cos\theta`, :math:`\sin\theta`, :math:`\cos\phi`, and :math:`\sin\phi`
    """
    cartesian_positions = np.asarray(cartesian_positions)
    r = np.sum(cartesian_positions**2, axis=0)**0.5
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = np.where(r == 0, 0, cartesian_positions / r)
    cos_theta = np.clip(normalized[2], -1, 1)
    sin_theta = (1 - cos_theta**2)**0.5
    with np.errstate(divide='ignore', invalid='ignore'):
        xy_projected = np.where(sin_theta == 0, 0, normalized[:2] / sin_theta)
    cos_phi = np.clip(xy_projected[0], -1, 1)
    sin_phi = np.clip(xy_projected[1], -1, 1)
    return np.stack([r, cos_theta, sin_theta, cos_phi, sin_phi], axis=0)


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
