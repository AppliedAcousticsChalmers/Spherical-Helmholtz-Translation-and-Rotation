"""Base functions used in spherical coordinates.

This module contains classes used to calculate and organize the required base
functions for the Helmholtz equation in spherical coordinates.
Theese can be used to reuse the calculation of base functions for repeated
evaluation of some expansion coefficients at specific points in space.


.. autosummary::
    :nosignatures:

    LegendrePolynomials
    AssociatedLegendrePolynomials
    RegularRadialBase
    SingularRadialBase
    DualRadialBase
    SphericalHarmonics
    RegularBase
    SingularBase
    DualBase

"""

import numpy as np
import scipy.special
from . import coordinates
from . import _bases, _shapes


def _wrap_indexing(key, func, *args):
    try:
        n, m = key
        n, m = np.broadcast_arrays(n, m)
        indices = np.stack([n, m], axis=1)
    except np.AxisError:
        return func(*args, [[n, m]])[..., 0]
    except ValueError:
        indices = np.asarray(key)
    if indices.ndim != 2:
        raise ValueError('Cannot index with multidimentional arrays')
    return func(*args, indices)


class LegendrePolynomials(coordinates.OwnerMixin):
    """Order only Legendre polynomials.

    Parameters
    ----------
        order : int
            The highest order included for the base.
        position : None, optional
            Position specifier, see `coordinates` for more info.
        colatitude : None, optional
            Colatitude of a spatial position. Will be used as cos(colatitude)
            as the input to the base function.
        x : None, optional
            Arbitrary input directly to the base function.
        defer_evaluation : bool, optional
            Do not calculate the values upon initialization of the object.
    """

    _evaluate = staticmethod(_bases.legendre_polynomials)
    _contract = staticmethod(_bases.legendre_contraction)

    def __init__(self, order, position=None, colatitude=None, x=None, defer_evaluation=False):
        if x is not None:
            self.coordinate = coordinates.NonspatialCoordinate(x=x)
        else:
            self.coordinate = coordinates.SpatialCoordinate.parse_args(position=position, colatitude=colatitude)
        self._data = np.zeros(self.shape + (self.num_unique(order),), dtype=float)

        if not defer_evaluation:
            self.evaluate(self.coordinate)

    @property
    def order(self):
        return self.num_unique_to_order(self._data.shape[-1])

    @classmethod
    def num_unique_to_order(cls, num_unique):
        return num_unique - 1

    @classmethod
    def num_unique(cls, order):
        return order + 1

    @property
    def shape(self):
        if isinstance(self.coordinate, coordinates.NonspatialCoordinate):
            return self.coordinate.shape
        return self.coordinate.shapes.colatitude

    def copy(self, deep=False):
        new_obj = super().copy(deep=deep)
        if deep:
            new_obj._data = self._data.copy()
        else:
            new_obj._data = self._data

        return new_obj

    def evaluate(self, position=None, colatitude=None, x=None):
        if x is not None:
            self.coordinate = coordinates.NonspatialCoordinate(x=x)
        else:
            self.coordinate = coordinates.SpatialCoordinate.parse_args(position=position, colatitude=colatitude)

        if isinstance(self.coordinate, coordinates.NonspatialCoordinate):
            arg = self.coordinate.x
        elif isinstance(self.coordinate, coordinates.SpatialCoordinate):
            arg = np.cos(self.coordinate.colatitude)
        else:
            raise TypeError(f'Unknown type of coordinate {type(self.coordinate)}')

        self._evaluate(arg, order=self.order, out=self._data)
        return self

    def apply(self, expansion):
        try:
            expansion_data = expansion._data
        except AttributeError:
            expansion_data = np.asarray(expansion)
        return self._contract(expansion_data, self._data)

    def __getitem__(self, key):
        return self._data[..., key]


class AssociatedLegendrePolynomials(LegendrePolynomials):
    """Associated Legendre polynomials with both order and mode.

    Note that this is using an orthonormal definition of the basis functions,
    i.e. not the one traditionally found in textbooks.

    See `LegendrePolynomials` for details on parameters.
    """

    _evaluate = staticmethod(_bases.associated_legendre_polynomials)
    _contract = staticmethod(_bases.associated_legendre_contraction)

    @classmethod
    def num_unique_to_order(cls, num_unique):
        return int((8 * num_unique + 1)**0.5 - 3) // 2

    @classmethod
    def num_unique(cls, order):
        return (order + 1) * (order + 2) // 2

    def __getitem__(self, key):
        return _wrap_indexing(key, _bases.associated_legendre_indexing, self._data)


class RadialBaseClass(coordinates.OwnerMixin):
    """Parent class for all radial bases.

    This class should not be instantiated, only inherited from.

    Parameters
    ----------
    order : int
        The highest order included for the base.
    position : None, optional
        Position specifier, see `coordinates` for more info.
    radius : None, optional
        Radius where the base is evaluated, in meters.
    wavenumber : None, optional
        Wavenumber where the base is evaluated, in rad/m.
    defer_evaluation : bool, optional
        Do not calculate the values upon initialization of the object.
    """

    def __init__(self, order, position=None, radius=None, wavenumber=None, defer_evaluation=False):
        self.coordinate = coordinates.SpatialCoordinate.parse_args(position=position, radius=radius)
        self._order = order
        self._wavenumber = np.asarray(wavenumber)
        if not defer_evaluation:
            self.evaluate(self.coordinate)

    def evaluate(self, position=None, radius=None, wavenumber=None):
        self.coordinate = coordinates.SpatialCoordinate.parse_args(position=position, radius=radius)
        if wavenumber is not None:
            self._wavenumber = wavenumber
        if self.wavenumber is None:
            # TODO: this should raise!
            x = self.coordinate.radius
        else:
            # TODO: I don't think we should auto-broadcast the wavenumber.
            x = self.coordinate.radius * self.wavenumber
            # x = self.coordinate.radius * np.reshape(self.wavenumber, np.shape(self.wavenumber) + (1,) * self.ndim)

        order = np.arange(self.order + 1)
        self._data = self._radial_func(order, x[..., None])
        return self

    @property
    def order(self):
        return self._order

    @property
    def shape(self):
        return _shapes.broadcast_shapes(self.coordinate.shapes.radius, self.wavenumber.shape)[0]

    def copy(self, deep=False):
        new_obj = super().copy(deep=deep)
        new_obj._order = self.order
        new_obj._wavenumber = self.wavenumber
        if deep:
            new_obj._data = self._data.copy()
        else:
            new_obj._data = self._data
        return new_obj

    @property
    def wavenumber(self):
        return self._wavenumber

    def __getitem__(self, key):
        return self._data[..., key]


class RegularRadialBase(RadialBaseClass):
    """Regular radial base, for interior problems.

    This is also known as the spherical Bessel function.
    See `RadialBaseClass` for details on the parameters.
    """

    def _radial_func(self, order, x):
        return scipy.special.spherical_jn(order, x, derivative=False)


class SingularRadialBase(RadialBaseClass):
    """Singular radial base, for exterior problems.

    This is also known as the spherical Hankel function.
    See `RadialBaseClass` for details on the parameters.
    """

    def _radial_func(self, order, x):
        return scipy.special.spherical_jn(order, x, derivative=False) + 1j * scipy.special.spherical_yn(order, x, derivative=False)


class DualRadialBase(RadialBaseClass):
    """Dual radial base, for interior and problems.

    This calculates both the regular and singular radial base functions.
    See `RadialBaseClass` for details on the parameters.
    """
    def _radial_func(self, order, x):
        bessel = scipy.special.spherical_jn(order, x, derivative=False)
        neumann = scipy.special.spherical_yn(order, x, derivative=False)
        return np.stack([bessel, bessel + 1j * neumann], axis=0)


class SphericalHarmonics(coordinates.OwnerMixin):
    """Spherical harmonics.

    This evaluate the (surface) spherical harmonics. The values are with the same
    phase and scaling conventions as e.g. scipy.

    Parameters
    ----------
    order : int
        The highest order included for the base.
    position : None, optional
        Position specifier, see `coordinates` for more info.
    colatitude : None, optional
        Colatitude angle where to evaluate the spherical harmonics, in [0, π].
    azimuth : None, optional
        Azimuth angle where to evaluate the spherical harmonics, in radians.
    defer_evaluation : bool, optional
        Do not calculate the values upon initialization of the object.
    """

    _contract = staticmethod(_bases.spherical_harmonics_contraction)

    def __init__(self, order, position=None, colatitude=None, azimuth=None, defer_evaluation=False, *args, **kwargs):
        self.coordinate = coordinates.SpatialCoordinate.parse_args(position=position, colatitude=colatitude, azimuth=azimuth)
        self._legendre = AssociatedLegendrePolynomials(order, position=self.coordinate, defer_evaluation=True)
        if not defer_evaluation:
            self.evaluate(self.coordinate)
        else:
            self._phase = np.ones(self.shape, complex)

    def evaluate(self, position=None, colatitude=None, azimuth=None):
        self.coordinate = coordinates.SpatialCoordinate.parse_args(position=position, colatitude=colatitude, azimuth=azimuth)
        if (position is not None) or (colatitude is not None):
            self._legendre.evaluate(position=self.coordinate)
        if (position is not None) or (azimuth is not None):
            self._phase = np.asarray(np.exp(1j * self.coordinate.azimuth))
        return self

    def apply(self, expansion):
        try:
            expansion_data = expansion._data
        except AttributeError:
            expansion_data = np.asarray(expansion)
        legendre_data = self._legendre._data
        phase_data = self._phase
        return self._contract(expansion_data, legendre_data, phase_data)

    @property
    def order(self):
        return self._legendre.order

    @property
    def shape(self):
        return self.coordinate.shapes.angular

    def copy(self, deep=False):
        new_obj = super().copy(deep=deep)
        new_obj._legendre = self._legendre.copy(deep=deep)
        if deep:
            new_obj._phase = self._phase.copy()
        else:
            new_obj._phase = self._phase
        return new_obj

    def __getitem__(self, key):
        return _wrap_indexing(key, _bases.spherical_harmonics_indexing, self._legendre._data, self._phase)


class MultipoleBase(coordinates.OwnerMixin):
    """Parent class for all multipole bases.

    This class should not be instantiated, only inherited from.

    Parameters
    ----------
    order : int
        The highest order included for the base.
    position : None, optional
        Position specifier, see `coordinates` for more info.
    wavenumber : None, optional
        Wavenumber where the base is evaluated, in rad/m.
    radius : None, optional
        Radius where the base is evaluated, in meters.
    colatitude : None, optional
        Colatitude angle where to evaluate the spherical harmonics, in [0, π].
    azimuth : None, optional
        Azimuth angle where to evaluate the spherical harmonics, in radians.
    defer_evaluation : bool, optional
        Do not calculate the values upon initialization of the object.
    """

    _contract = staticmethod(_bases.multipole_contraction)

    def __init__(self, order, position=None, wavenumber=None,
                 radius=None, colatitude=None, azimuth=None, defer_evaluation=False):
        self.coordinate = coordinates.SpatialCoordinate.parse_args(position=position, radius=radius, colatitude=colatitude, azimuth=azimuth)
        self._angular = SphericalHarmonics(order=order, position=self.coordinate, defer_evaluation=defer_evaluation)
        self._radial = self._radial_cls(order=order, position=self.coordinate, wavenumber=wavenumber, defer_evaluation=defer_evaluation)

    def evaluate(self, position=None, wavenumber=None, radius=None, colatitude=None, azimuth=None):
        self.coordinate = coordinates.SpatialCoordinate.parse_args(position=position, radius=radius, colatitude=colatitude, azimuth=azimuth)
        if (position is not None) or (radius is not None) or (wavenumber is not None):
            self._radial.evaluate(self.coordinate, wavenumber=wavenumber)
        if (position is not None) or (colatitude is not None) or (azimuth is not None):
            # TODO: We could in principle optimize the case where only a new azimuth angle is given.
            self._angular.evaluate(self.coordinate)
        return self

    @property
    def order(self):
        return self._angular.order

    @property
    def shape(self):
        return _shapes.broadcast_shapes(self.coordinate.shape, self.wavenumber.shape)[0]

    def copy(self, deep=False):
        new_obj = super().copy(deep=deep)
        new_obj._angular = self._angular.copy(deep=deep)
        new_obj._radial = self._radial.copy(deep=deep)
        return new_obj

    @property
    def wavenumber(self):
        return self._radial.wavenumber

    def apply(self, expansion):
        try:
            expansion_data = expansion._data
        except AttributeError:
            expansion_data = np.asarray(expansion)

        wavenumber = getattr(expansion, 'wavenumber', None)
        if wavenumber is not None:
            if not np.allclose(wavenumber, self.wavenumber):
                raise ValueError('Cannot apply bases to expansion of different wavenumber')
        else:
            # An expansion can be defined without a wavenumber, which is fine
            # TODO: test if this could break things
            wavenumber = self.wavenumber

        legendre_data = self._angular._legendre._data
        phase_data = self._angular._phase
        radial_data = self._radial._data
        return self._contract(expansion_data, radial_data, legendre_data, phase_data)

    def __getitem__(self, key):
        return _wrap_indexing(key, _bases.multipole_indexing, self._radial._data, self._angular._legendre._data, self._angular._phase)


class RegularBase(MultipoleBase):
    """Regular multipole base, for interior problems.

    See `MultipoleBase` for details on the parameters.
    """

    _radial_cls = RegularRadialBase


class SingularBase(MultipoleBase):
    """Singular multipole base, for exterior problems.

    See `MultipoleBase` for details on the parameters.
    """

    _radial_cls = SingularRadialBase


class DualBase(MultipoleBase):
    """Dual multipole base, for interior and exterior problems.

    This calculates both the singular and regular multipole bases.
    See `MultipoleBase` for details on the parameters.
    """

    _radial_cls = DualRadialBase

    class _Regular:
        def __init__(self, parent):
            self.parent = parent

        def __getitem__(self, key):
            return self.parent[key][0]

    class _Singular:
        def __init__(self, parent):
            self.parent = parent

        def __getitem__(self, key):
            return self.parent[key][1]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._regular = self._Regular(self)
        self._singular = self._Singular(self)

    @property
    def regular(self):
        return self._regular

    @property
    def singular(self):
        return self._singular
