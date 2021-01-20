import numpy as np
from . import coordinates, _shape_utilities


class Expansion:
    def __init__(self, order=None, data=None, wavenumber=None):
        self._wavenumber = wavenumber
        if _shape_utilities.is_value(data):
            self._data = data
            if order is not None and self.order != order:
                raise ValueError(f'Received data of order {self.order} in conflict with specified order {order}')
            if np.shape(wavenumber) != np.shape(data)[1:(np.ndim(wavenumber) + 1)]:
                raise ValueError(f'Received wavenumber of shape {np.shape(wavenumber)} in conflict with data of shape {np.shape(data)}')
        elif type(data) == np.broadcast and order is None:
            self._data = np.zeros(np.shape(data), dtype=complex)
        elif order is not None:
            num_unique = (order + 1) ** 2
            self._data = np.zeros((num_unique,) + np.shape(wavenumber) + np.shape(data), dtype=complex)
        else:
            raise ValueError('Cannot initialize expansion without either raw data or known order')

    @property
    def order(self):
        return int(np.shape(self._data)[0] ** 0.5) - 1

    @property
    def shape(self):
        return np.shape(self._data)[(1 + np.ndim(self._wavenumber)):]

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

    def reshape(self, newshape, *args, **kwargs):
        new_obj = self.copy()
        new_obj._data = new_obj._data.reshape(new_obj._data.shape[:1 + np.ndim(new_obj.wavenumber)] + tuple(newshape))
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
        return self._data[self._idx(n, m)]

    def __setitem__(self, key, value):
        n, m, = key
        self._data[self._idx(n, m)] = value

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
        from .rotations import Rotation
        return Rotation(
            order=self.order, colatitude=colatitude,
            azimuth=azimuth, secondary_azimuth=secondary_azimuth,
            new_z_axis=new_z_axis, old_z_axis=old_z_axis
        ).apply(self)

    def evaluate(self, position=None, radius=None, colatitude=None, azimuth=None):
        try:
            return self._base_cls(
                order=self.order, wavenumber=self.wavenumber, position=position,
                radius=radius, colatitude=colatitude, azimuth=azimuth
            ).apply(self)
        except AttributeError as e:
            if "no attribute '_base_cls'" in str(e):
                raise ValueError('Cannot evaluate expansion without a related base')
            else:
                raise e


class SphericalSurfaceExpansion(Expansion):
    from .bases import SphericalHarmonics as _base_cls


class InteriorExpansion(Expansion):
    from .bases import RegularBase as _base_cls

    def translate(self, position=None, order=None, radius=None, colatitude=None, azimuth=None):
        from .translations import InteriorTranslation
        return InteriorTranslation(
            input_order=self.order, output_order=self.order if order is None else order,
            position=position, radius=radius, colatitude=colatitude, azimuth=azimuth,
            wavenumber=self.wavenumber
        ).apply(self)

    def reexpand(self, position, order=None):
        return self.translate(position=-position, order=order)


class ExteriorExpansion(Expansion):
    from .bases import SingularBase as _base_cls

    def translate(self, order=None, position=None, radius=None, colatitude=None, azimuth=None, domain='extreior'):
        if domain == 'exterior':
            from .translations import ExteriorTranslation as TranslationCls
        elif domain == 'interior':
            from .translations import ExteriorInteriorTranslation as TranslationCls
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

    The propagation direction of the plane wave can be specified using either
    a wavevector, or by colatitude, azimuth, and wavenumber.
    The strength :math:`q` referes to the scaling of a plane wave
    :math:`q \exp{i \vec k \cdot \vec r}`, where :math:`\vec r` is the position
    in the field, and :math:`\vec k` is the wavevector.
    """
    from .bases import SphericalHarmonics
    if wavevector is not None:
        wavenumber, colatitude, azimuth = coordinates.cartesian_2_spherical(wavevector)
    expansion = InteriorExpansion(order=order, data=np.broadcast(colatitude, azimuth), wavenumber=wavenumber)
    spherical_harmonics = SphericalHarmonics(order=order, colatitude=colatitude, azimuth=azimuth)
    strength = strength * 4 * np.pi
    for n in range(order + 1):
        for m in range(-n, n + 1):
            expansion[n, m] = strength * spherical_harmonics[n, m].conj() * (1j) ** n
    return expansion


def monopole(strength=1, order=0, wavenumber=1, position=None, domain='exterior'):
    r"""Create a monopole source expansion.

    If the monopole is not centered at the origin, it will be translated.
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

    The strength :math:`q` refers to the scaling of the monopole radiation
    :math:`{q \over 4 \pi r} \exp{ikr}` where :math:`r` is the distance to the
    source and :math:`k` is the wavenubmer.
    """
    strength = strength * wavenumber * 1j / (4 * np.pi)**0.5
    expansion = ExteriorExpansion(order=0, data=[strength], wavenumber=wavenumber)
    if position is not None:
        expansion = expansion.translate(order=order, position=position, domain=domain)
    return expansion


def circular_ring(
        order, radius, strength=1,
        colatitude=0, azimuth=None, wavenumber=None, wavevector=None,
        position=None, domain='exterior'):
    r"""Create a circular ring source expansion.

    The source strength relates to the far-field radiation characteristic of a
    circular ring, as :math:`{q \over 4\pi} J_0(ka\sin\theta) \exp{ikr}` where
    :math:`q` is the input source strength, :math:`J_0` is the zeroth order
    Bessel function, :math:`k` is the wavenuber, :math:`a` is the radius of the
    ring, :math:`r` is the distance from the source, and :math:`\theta` is the
    angle between the position vector and the source normal.
    """
    if wavevector is not None:
        wavenumber, colatitude, azimuth = coordinates.cartesian_2_spherical(wavevector)
    expansion = ExteriorExpansion(order=order, wavenumber=wavenumber, data=np.broadcast(colatitude, azimuth))
    strength = strength * wavenumber * 1j / 2
    from scipy.special import spherical_jn, gamma
    even_n = np.arange(0, order + 1, 2)
    # values = np.pi * radius * spherical_jn(even_n, wavenumber * radius) * (2 * even_n + 1)**0.5 / (gamma(even_n / 2 + 1) * gamma(0.5 - even_n / 2))
    values = strength * spherical_jn(even_n, wavenumber * radius) * (2 * even_n + 1)**0.5 / (gamma(even_n / 2 + 1) * gamma(0.5 - even_n / 2))
    # TODO: This seems to give about the right shape, but the overall scale is not correct.
    # We need to look at this derivation and compare ot to a derivation of the simple free-field solution in order to get
    # the "strength" of the source.
    for value, n in zip(values, even_n):
        expansion[n, 0] = value
    if colatitude != 0:
        expansion = expansion.rotate(colatitude=colatitude, azimuth=azimuth)
    if position is not None:
        expansion = expansion.translate(position=position, domain=domain)
    return expansion
