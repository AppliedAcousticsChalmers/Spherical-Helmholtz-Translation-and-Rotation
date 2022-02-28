"""Organization of the various expansions coefficients.

This provides convenience classes for storing expansion coefficients with
the cirrect indexing conventions for the package. There's also a few
functions which can create some commonly used expansions, e.g. monopoles
or plane waves.
The classes with an explicit domain also has methods to conviniently evaluate
the expansion at some positions in space, or to apply certain transforms.

.. autosummary::
    :nosignatures:

    Expansion
    SphericalSurfaceExpansion
    InteriorExpansion
    ExteriorExpansion
    plane_wave
    monopole
    circular_ring

"""


import numpy as np
from . import coordinates
from . import _shapes


class Expansion:
    """Basic expansion without notion of domain.

    This can be used as a plain holder for the coefficients which has both order
    and mode, e.g. spherical harmonics expansion coefficients. Instances of this
    class has no inherent notion of whas is the correct base to use, so it
    cannot create them automatically.

    Parameters
    ----------
    order : int
        The highest order included in the expansion.
        If instantiated with raw data, no order is required for the instantiated
        since it is calculated from the shape of the data. If `data` if not given,
        `order` is required and an all-zero expansion is created.
        If both `data` and `order` are given, the have to agree on the order.
    data : array_like
        Raw data for expansion coefficients.
        Has to have the correct indexing conventions with the coefficient axis last.
    wavenumber : None, optional
        Optionial wavenumber of the expansion.
        This has to be broadcastable with the spatial shape of the data.
    shape : None, optional
        Spatial shape for all-zero expansions.
        If both `data` and `shape` are given, the have to agree on the spatial shape.
    """

    def __init__(self, order=None, data=None, wavenumber=None, shape=None):
        self._wavenumber = wavenumber
        if data is not None:
            self._data = data
            if order is not None and self.order != order:
                raise ValueError(f'Received data of order {self.order} in conflict with specified order {order}')
            try:
                _shapes.broadcast_shapes(np.shape(wavenumber), np.shape(data)[:-1])
            except ValueError:
                raise ValueError(f'Received wavenumber of shape {np.shape(wavenumber)} in conflict with data of shape {np.shape(data)}')
            if shape is not None and np.shape(data)[:-1] != shape:
                raise ValueError(f'Received explicit shape {shape} in conflict with data of shape {np.shape(data)}')
        elif order is not None:
            shape = tuple() if shape is None else (shape,) if type(shape) is int else tuple(shape)
            num_unique = (order + 1) ** 2
            self._data = np.zeros(shape + (num_unique,), dtype=complex)
        else:
            raise ValueError('Cannot initialize expansion without either raw data or known order')

    @property
    def order(self):
        return int(np.shape(self._data)[-1] ** 0.5) - 1

    @property
    def shape(self):
        return np.shape(self._data)[:-1]

    @property
    def ndim(self):
        return len(self.shape)

    def copy(self, deep=False):
        new_obj = type(self).__new__(type(self))
        new_obj._wavenumber = self._wavenumber
        if deep:
            new_obj._data = self._data.copy()
        else:
            new_obj._data = self._data
        return new_obj

    @property
    def wavenumber(self):
        return self._wavenumber

    def _idx(self, order=None, mode=None, index=None):
        if index is None:
            # The default mode, getting the linear index from the order and mode.
            if order > self.order:
                raise IndexError(f'Order {order} out of bounds for {self.__class__.__name__} with max order {self.order}')
            if abs(mode) > order:
                raise IndexError(f'Mode {mode} out of bounds for order {order}')
            return order ** 2 + order + mode
        else:
            # The inverse mode, getting the order and mode from the linear index.
            if index >= (self.order + 1)**2 or index < 0:
                raise IndexError(f'Index {index} out of bounds for {self.__class__.__name__} with max order {self.order}')
            order = int(index ** 0.5)
            mode = index - order * (order + 1)
            return (order, mode)

    @property
    def _component_indices(self):
        out = []
        for n in range(self.order + 1):
            for m in range(-n, n + 1):
                out.append((n, m))
        return out

    def __getitem__(self, key):
        n, m, = key
        return self._data[..., self._idx(n, m)]

    def __setitem__(self, key, value):
        n, m, = key
        self._data[..., self._idx(n, m)] = value

    def _compatible_with(self, other):
        # TODO: Raise meaningful exceptions when the objects are not compatible.
        return (type(self) == type(other)
                and self.order == other.order
                and self.shape == other.shape
                and (self.wavenumber is other.wavenumber or np.allclose(self.wavenumber, other.wavenumber))
                )

    def __eq__(self, other):
        return self._compatible_with(other) and np.allclose(self._data, other._data)

    def __add__(self, other):
        if self._compatible_with(other):
            return type(self)(data=self._data + other._data, wavenumber=self.wavenumber)
        return NotImplemented

    def __iadd__(self, other):
        if self._compatible_with(other):
            self._data += other._data
            return self
        return NotImplemented

    def __sub__(self, other):
        if self._compatible_with(other):
            return type(self)(data=self._data - other._data, wavenumber=self.wavenumber)
        return NotImplemented

    def __isub__(self, other):
        if self._compatible_with(other):
            self._data -= other._data
            return self
        return NotImplemented

    def __mul__(self, other):
        return type(self)(data=self._data * other, wavenumber=self.wavenumber)

    def __imul__(self, other):
        self._data *= other
        return self

    def __div__(self, other):
        return type(self)(data=self._data / other, wavenumber=self.wavenumber)

    def __idiv__(self, other):
        self._data /= other
        return self

    def __neg__(self):
        return type(self)(data=-self._data, wavenumber=self.wavenumber)

    def apply(self, transform, *args, **kwargs):
        return transform.apply(self, *args, **kwargs)

    def rotate(self, colatitude=0, azimuth=0, secondary_azimuth=0, new_z_axis=None, old_z_axis=None):
        from .tranforms import Rotation
        return Rotation(
            order=self.order, colatitude=colatitude,
            azimuth=azimuth, secondary_azimuth=secondary_azimuth,
            new_z_axis=new_z_axis, old_z_axis=old_z_axis
        ).apply(self)

    def bases(self, position=None, radius=None, colatitude=None, azimuth=None):
        try:
            return self._base_cls(
                order=self.order, wavenumber=self.wavenumber, position=position,
                radius=radius, colatitude=colatitude, azimuth=azimuth
            )
        except AttributeError as e:
            if "no attribute '_base_cls'" in str(e):
                raise ValueError('Cannot get bases for expansion without a related base class')
            else:
                raise e

    def evaluate(self, position=None, radius=None, colatitude=None, azimuth=None):
        return self.bases(position=position, radius=radius, colatitude=colatitude, azimuth=azimuth).apply(self)


class SphericalSurfaceExpansion(Expansion):
    """Holds spherical harmonics expansions.

    Expansion corresponding to `SphericalHarmonics` basis functions.
    These are defined on the surface of a sphere, and can be used
    to expand arbitrary functions.

    See `Expansion` for parameters and more methods.
    """

    from .bases import SphericalHarmonics as _base_cls


class InteriorExpansion(Expansion):
    """Holds interior expansions.

    Expansion corresponding to `RegularBase` basis functions.
    These expansions are well defined when all sources are outside
    the region of evaluation. This means that the evaluation radius
    is smaller than the distance to the closest source, e.g. an incident
    sound field.

    See `Expansion` for parameters and more methods.
    """

    from .bases import RegularBase as _base_cls

    def translate(self, position=None, order=None, radius=None, colatitude=None, azimuth=None):
        from .tranforms import InteriorTranslation
        return InteriorTranslation(
            input_order=self.order, output_order=self.order if order is None else order,
            position=position, radius=radius, colatitude=colatitude, azimuth=azimuth,
            wavenumber=self.wavenumber
        ).apply(self)

    def reexpand(self, position, order=None):
        return self.translate(position=-position, order=order)


class ExteriorExpansion(Expansion):
    """Holds exterior expansions.

    Expansion corresponding to `SingularBase` basis functions.
    These expansions are well defined when all sources are contained
    within the region closer to the origin than the evaluation domain.
    This means that the evaluation radius has to be larger than the distance
    to the closest source, e.g. a source at the origin.

    See `Expansion` for parameters and more methods.
    """

    from .bases import SingularBase as _base_cls

    def translate(self, order=None, position=None, radius=None, colatitude=None, azimuth=None, domain='exterior'):
        if domain == 'exterior':
            from .tranforms import ExteriorTranslation as TranslationCls
        elif domain == 'interior':
            from .tranforms import ExteriorInteriorTranslation as TranslationCls
        else:
            raise ValueError(f'Unknown domain `{domain}`')
        return TranslationCls(
            input_order=self.order, output_order=self.order if order is None else order,
            position=position, radius=radius, colatitude=colatitude, azimuth=azimuth,
            wavenumber=self.wavenumber
        ).apply(self)

    def reexpand(self, position, order=None, domain='exterior'):
        return self.translate(position=-position, order=order, domain=domain)


def plane_wave(order, strength=1, colatitude=None, azimuth=None, wavenumber=None, wavevector=None):
    r"""Create an expansion of a plane wave.

    A plane wave has the field

    .. math:: q \exp{(i \vec k \cdot \vec r)}

    where :math:`\vec r` is the position in the field,
    and :math:`\vec k` is the wavevector.

    The propagation direction of the plane wave can be specified using either
    a wavevector, or by colatitude, azimuth, and wavenumber.

    Parameters
    ----------
    order : int
        The highest order included in the expansion.
    strength : numerical, default 1
        The strength of the souource, :math:`q` in the above definition.
    colatitude : None, optional
        Colatitude propagation angle of the wave.
    azimuth : None, optional
        Azimuth propagation direction of the wave.
    wavenumber : None, optional
        Wavenumber of the propagating wave, :math:`k` in the above definition.
    wavevector : None, optional
        Wavevector, i.e. both wavenuber and propagation direction for the wave.
        This will override the separate parameters if given.

    Returns
    -------
    `InteriorExpansion`
        The expansion of a plane wave is always an interior sound field.
    """
    from .bases import SphericalHarmonics
    wave_coordinate = coordinates.SpatialCoordinate.parse_args(position=wavevector, radius=wavenumber, colatitude=colatitude, azimuth=azimuth)
    expansion = InteriorExpansion(order=order, shape=wave_coordinate.shapes.angular, wavenumber=wave_coordinate.radius)
    spherical_harmonics = SphericalHarmonics(order=order, position=wave_coordinate)
    strength = strength * 4 * np.pi
    for n in range(order + 1):
        for m in range(-n, n + 1):
            expansion[n, m] = strength * spherical_harmonics[n, m].conj() * (1j) ** n
    return expansion


def monopole(strength=1, order=0, wavenumber=1, position=None, domain='exterior'):
    r"""Create a monopole source expansion.

    A monopole has the field

    .. math:: \frac{q}{4 \pi r} \exp{(ikr)}

    where :math:`r` is the distance beterrn the souorce and the evaluation position.


    Parameters
    ----------
    strength : numerical, default 1
        Strength or the souorce, :math:`q` in ithe above definition.
    order : int, default 0
        This can be used to return higher orders than needed for untranslated
        expansioins, or to controll the output order of the translation.
    wavenumber : numerical, default 1
        The wavenumber of the source, :math:`k` in the above definition.
    position : None, optional
        Position specifier for translation, see `coordinates` for more info.
        If given, the expansion will represent a monopole at this position.
    domain : str, optional
        Domain of translated expansions.
        Depending on where the resulting expansion should be evaluated, the target
        domain should be specified.
        If `domain='exterior'` (default), an exterior expansion will be returned,
        i.e. an expansion which is valid further away from the origin than the
        source location. Thus, the expansion should be evaluated using the singular
        bases, and in the exterior domain.
        If `domain='interior'`, an interior expansion will be returned,
        i.e. an expansion which is valid closer to from the origin than the
        source location. Thus, the expansion should be evaluated using the regular
        bases, and in the interior domain.
        A monopole at the origin (`position=None`) is always an exterior expansion.

    Returns
    -------
    `ExteriorExpansion` or `InteriorExpansion`
        See the `domain` documentation.
    """
    strength = strength * wavenumber * 1j / (4 * np.pi)**0.5
    expansion = ExteriorExpansion(order=order if position is None else 0, wavenumber=wavenumber)
    expansion[0, 0] = strength
    if position is not None:
        expansion = expansion.translate(order=order, position=position, domain=domain)
    return expansion


def circular_ring(
        order, radius, strength=1,
        colatitude=None, azimuth=None, wavenumber=None, wavevector=None,
        position=None, domain='exterior', source_order=None):
    r"""Create a circular ring source expansion.

    The returns the expansioni coefficients corresponding to a circular ring.
    In the far-field, this has the directivity

    .. math:: \frac{q}{4\pi} J_0(ka\sin\theta) \exp{ikr}

    where :math:`J_0` is the zeroth order Bessel function,
    :math:`r` is the distance from the source,
    and :math:`\theta` is the angle between the position vector
    and the source normal.


    Parameters
    ----------
    order : int
        The output order for the source.
    radius : float
        The effective radius of the source, :math:`a` in the above definition.
    strength : numerical, default 1
        Strength of the source, :math:`q` in the abive definition.
    colatitude : None, optional
        Colatitude angle of the normal of the source.
    azimuth : None, optional
        Azimuth angle of the normal of the source.
    wavenumber : float, optional
        Wavenumber if the sound field, :math:`k` in the above definition.
    wavevector : None, optional
        Wavevector, i.e. both wavenuber and propagation direction for the wave.
        This will override the separate parameters if given.
    position : None, optional
        Position specifier for translation, see `coordinates` for more info.
        If given, the expansion will represent a source at this position.
    domain : str, optional
        Domain of translated expansions.
        Depending on where the resulting expansion should be evaluated, the target
        domain should be specified.
        If `domain='exterior'` (default), an exterior expansion will be returned,
        i.e. an expansion which is valid further away from the origin than the
        source location. Thus, the expansion should be evaluated using the singular
        bases, and in the exterior domain.
        If `domain='interior'`, an interior expansion will be returned,
        i.e. an expansion which is valid closer to from the origin than the
        source location. Thus, the expansion should be evaluated using the regular
        bases, and in the interior domain.
        A monopole at the origin (`position=None`) is always an exterior expansion.
    source_order : int, optional
        Only used for translated expansions.
        The source order argument can be used in combination with the position argument
        to use a different expansion order for the initial source expansion before the
        translation is applied.

    Returns
    -------
    `ExteriorExpansion` or `InteriorExpansion`
        See the `domain` documentation.
    """
    if source_order is None:
        source_order = order
    elif position is None:
        raise ValueError('Source order argument is meaningless unless the position argument is also used.')

    wave_coordinate = coordinates.SpatialCoordinate.parse_args(position=wavevector, radius=wavenumber, colatitude=colatitude, azimuth=azimuth)
    wavenumber = wave_coordinate.radius
    expansion = ExteriorExpansion(order=source_order, wavenumber=wavenumber, shape=wave_coordinate.shapes.angular)
    strength = strength * wavenumber * 1j / 2
    from scipy.special import spherical_jn, gamma
    ka = wavenumber * radius
    even_n = np.arange(0, source_order + 1, 2).reshape([-1] + [1] * np.ndim(ka))
    # values = np.pi * radius * spherical_jn(even_n, wavenumber * radius) * (2 * even_n + 1)**0.5 / (gamma(even_n / 2 + 1) * gamma(0.5 - even_n / 2))
    values = strength * spherical_jn(even_n, ka) * (2 * even_n + 1)**0.5 / (gamma(even_n / 2 + 1) * gamma(0.5 - even_n / 2))
    # TODO: This seems to give about the right shape, but the overall scale is not correct.
    # We need to look at this derivation and compare ot to a derivation of the simple free-field solution in order to get
    # the "strength" of the source.
    for value, n in zip(values, even_n):
        expansion[n, 0] = value
    if colatitude is not None and np.any(colatitude != 0):
        expansion = expansion.rotate(colatitude=colatitude, azimuth=azimuth)
    if position is not None:
        expansion = expansion.translate(order=order, position=position, domain=domain)
    return expansion
