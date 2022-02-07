r"""Spatial transforms for expansions.

Spatial transforms are operations that work on expansions to express the same field
but with a different coordinate system. The can be used in two ways:
1) To find expansion coefficints of a physical field after undergoing some physical
transform, e.g. by moving a source.
2) To find the expansion coefficients of the same physical field but in a different
mathematical coordinate system, e.g. as measured at a different point in space.
These wto operations are closely linked, and are often the inverse of each other.
This means that the expansion coefficients of an "input" expansion after undegoing
a movement of the field by :math:`\vec x` are the same as the expansion coefficients
of the same "input" field expanded at a new origin :math:`-\vec x`.
The default operation here is that of translating the field, e.g. a field created by a
source at [1, 2, 3] translated by [4, 5, 6] will have the same expansion coefficients
at the field from the source placed at [5, 7, 9].

The transforms are handled with a number of classes listed below. Rotations are either
just a colatitude rotation, or a full rotation of the field. Translations are implemented
along the z-axis only, for lower computational cost. General translations are handled as
a sequence of rotation->coaxial translation->rotation. For full controll of an arbitrary
transform, use the rotation and coaxial transslation classes instead of the translation
classes.

A translation can be done either within a domain, or from the exterior domain to
the interior domain. Take care to use the correct class to preserve the validity
of the input expansion at the desired evaluation domain for the output expansion.
There is often larger errors in the translations near the boundary of the region
of validity. Take extra care if this region is of interest.


.. autosummary::
    :nosignatures:

    ColatitudeRotation
    Rotation
    CoaxialTranslation
    InteriorCoaxialTranslation
    ExteriorCoaxialTranslation
    ExteriorInteriorCoaxialTranslation
    Translation
    InteriorTranslation
    ExteriorTranslation
    ExteriorInteriorTranslation
"""

import numpy as np
from . import coordinates, expansions
from . import _rotations, _translations, _shapes


class ColatitudeRotation(coordinates.OwnerMixin):
    """Organizes rotations for colatitude directions.

    Parameters
    ----------
    order : int
        The highest order for the rotation.
        Rotations are performed at constant order, so the order of the input
        expansion and the output expansion are kept the same.
    position : optional
        Position specifier for the rotation, see `shetar.coordinates.Rotation`.
    colatitude : float, optional
        Angle in radians by which to rotate.
    defer_evaluation : bool, optional
        Do not calculate the values upon initialization of the object.
    """

    _evaluate = staticmethod(_rotations.colatitude_rotation_coefficients)

    def __init__(self, order, position=None, colatitude=None, defer_evaluation=False):
        self.coordinate = coordinates.Rotation.parse_args(position=position, colatitude=colatitude)
        num_unique = _rotations.colatitude_order_to_unique(order)
        self._data = np.zeros(self.coordinate.shapes.colatitude + (num_unique,), dtype=float)
        if not defer_evaluation:
            self.evaluate(self.coordinate)

    @property
    def order(self):
        return _rotations.colatitude_unique_to_order(self._data.shape[-1])

    @property
    def shape(self):
        return self.coordinate.shapes.colatitude

    def copy(self, deep=False):
        new_obj = super().copy(deep=deep)
        new_obj._order = self._order
        if deep:
            new_obj._data = self._data.copy()
        else:
            new_obj._data = self._data
        return new_obj

    def evaluate(self, position=None, colatitude=None):
        self.coordinate = coordinates.Rotation.parse_args(position=position, colatitude=colatitude)
        self._evaluate(self.coordinate.colatitude, order=self.order, out=self._data)
        return self

    def _transform(self, expansion_data, inverse, out_data):
        return _rotations.colatitude_rotation_transform(expansion_data, self._data, inverse, out=out_data)

    def apply(self, expansion, inverse=False, out=None):
        if isinstance(expansion, expansions.Expansion):
            expansion_data = expansion._data
        else:
            expansion_data = np.asarray(expansion)

        if out is None:
            out_data = None
        elif out is expansion:
            raise NotImplementedError('Rotations cannot currently be applied in place')
        else:
            if isinstance(out, expansions.Expansion):
                out_data = out._data
            else:
                if isinstance(out, np.ndarray):
                    out_data = out
                else:
                    raise TypeError(f'Invalid type {type(out)} for output')

        out_data = self._transform(expansion_data=expansion_data, inverse=inverse, out_data=out_data)

        if isinstance(out, expansions.Expansion):
            return out

        out_type = type(expansion) if isinstance(expansion, expansions.Expansion) else expansions.Expansion
        return out_type(data=out_data, wavenumber=getattr(expansion, 'wavenumber', None))


class Rotation(ColatitudeRotation):
    """Organizes arbitrary rotations.

    This handles rotations with both a colatitude and two azimuth parts.
    For details on how the two azimuth angles interplay with the colatitude angle,
    refer to the examples.

    See `ColatitudeRotation` and `shetar.coordinates.Rotation` for details on the parameters.
    """

    def __init__(self, order, position=None, colatitude=None, azimuth=None, secondary_azimuth=None, new_z_axis=None, old_z_axis=None, defer_evaluation=False):
        coordinate = coordinates.Rotation.parse_args(
            position=position, new_z_axis=new_z_axis, old_z_axis=old_z_axis,
            colatitude=colatitude, azimuth=azimuth, secondary_azimuth=secondary_azimuth)
        super().__init__(order=order, position=coordinate, defer_evaluation=defer_evaluation)

    def _transform(self, expansion_data, inverse, out_data):
        return _rotations.full_rotation_transform(expansion_data, self._data, self._primary_phase, self._secondary_phase, inverse, out=out_data)

    def evaluate(self, position=None, colatitude=None, azimuth=None, secondary_azimuth=None, new_z_axis=None, old_z_axis=None):
        if (position is None) and (new_z_axis is None) and (old_z_axis is None) and (colatitude is None):
            # Allows chaing the azimuth angles without reevaluating the colatitude rotation.
            # This might be useful sometimes since changing the azimuth angles is much cheaper than
            # changing the colatitude angle. If we anyhow change the colatitude roration, reevaluating the
            # azimuth rotations will be very cheap so we won't bother with checking if we could skip it.
            if azimuth is not None:
                self._primary_phase = np.asarray(np.exp(1j * azimuth))
            if secondary_azimuth is not None:
                self._secondary_phase = np.asarray(np.exp(1j * secondary_azimuth))
        else:
            self.coordinate = coordinates.Rotation.parse_args(
                position=position, new_z_axis=new_z_axis, old_z_axis=old_z_axis,
                colatitude=colatitude, azimuth=azimuth, secondary_azimuth=secondary_azimuth)
            super().evaluate(position=self.coordinate)
            self._primary_phase = np.asarray(np.exp(1j * self.coordinate.azimuth))
            self._secondary_phase = np.asarray(np.exp(1j * self.coordinate.secondary_azimuth))
        return self

    @property
    def shape(self):
        return self.coordinate.shape

    def copy(self, deep=False):
        new_obj = super().copy(deep=deep)
        if deep:
            new_obj._primary_phase = self._primary_phase.copy()
            new_obj._secondary_phase = self._secondary_phase.copy()
        else:
            new_obj._primary_phase = self._primary_phase
            new_obj._secondary_phase = self._secondary_phase
        return new_obj


class CoaxialTranslation(coordinates.OwnerMixin):
    """Patent class for coaxial translations.

    This class should not be instantiated, only inherited from.

    Parameters
    ----------
    input_order : int
        The order or the input expansions.
    output_order : int
        The order of the output expansions.
    position
        Positioin specifier, see `shetar.coordinates.Translation`.
    radius : float
        Distance to translate, see `shetar.coordinates.Translation`.
    wavenumber : float
        The wavenumber that the translation operates at.
    defer_evaluation : bool, optional
        Do not calculate the values upon initialization of the object.
    """

    _default_output_type = expansions.Expansion
    _transform = staticmethod(_translations.coaxial_translation_transform)

    def __init__(self, input_order, output_order, position=None, radius=None, wavenumber=None, defer_evaluation=False):
        self._input_order = input_order
        self._output_order = output_order
        self._wavenumber = np.asarray(wavenumber)

        self.coordinate = coordinates.Translation.parse_args(position=position, radius=radius)
        num_unique = _translations.coaxial_order_to_unique(input_order, output_order)
        self._data = np.zeros(self._coaxial_shape + (num_unique,), self._dtype)

        if not defer_evaluation:
            self.evaluate(position=self.coordinate)

    @property
    def order(self):
        return (self.input_order, self.output_order)

    @property
    def input_order(self):
        return self._input_order

    @property
    def output_order(self):
        return self._output_order

    @property
    def shape(self):
        return self._coaxial_shape

    @property
    def _coaxial_shape(self):
        # Needed since the full translation will override the .shape
        return _shapes.broadcast_shapes(self.coordinate.shapes.radius, self.wavenumber.shape)[0]

    def copy(self, deep=False):
        new_obj = super().copy(deep=deep)
        new_obj._input_order = self._input_order
        new_obj._output_order = self._output_order
        new_obj._max_order = self._max_order
        new_obj._min_order = self._min_order
        if deep:
            new_obj._wavenumber = self._wavenumber.copy()
            new_obj._data = self._data.copy()
        else:
            new_obj._wavenumber = self._wavenumber
            new_obj._data = self._data
        return new_obj

    @property
    def wavenumber(self):
        return self._wavenumber

    def evaluate(self, position=None, radius=None, wavenumber=None):
        if wavenumber is not None:
            self._wavenumber = np.asarray(wavenumber)
        self.coordinate = coordinates.Translation.parse_args(position=position, radius=radius)
        self._evaluate(self.coordinate.radius * self.wavenumber, input_order=self.input_order, output_order=self.output_order, out=self._data)
        return self

    def apply(self, expansion, inverse=False, out=None):
        wavenumber = getattr(expansion, 'wavenumber', None)
        if wavenumber is not None:
            if not np.allclose(wavenumber, self.wavenumber):
                raise ValueError('Cannot apply translation to expansion of different wavenuber')

        if isinstance(expansion, expansions.Expansion):
            expansion_data = expansion._data
        else:
            expansion_data = np.asarray(expansion)

        if out is None:
            out_data = None
        elif out is expansion:
            raise NotImplementedError('Rotations cannot currently be applied in place')
        else:
            if isinstance(out, expansions.Expansion):
                out_data = out._data
            else:
                if isinstance(out, np.ndarray):
                    out_data = out
                else:
                    raise TypeError(f'Invalid type {type(out)} for output')

        out_data = self._transform(expansion_data, self._data, self.input_order, self.output_order, inverse=inverse, out=out_data)

        if isinstance(out, expansions.Expansion):
            return out

        return self._default_output_type(data=out_data, wavenumber=self.wavenumber)


class InteriorCoaxialTranslation(CoaxialTranslation):
    """Handles translations in the interior domain.

    This applies translations from an interior translation to an interior
    translation.

    See `CoaxialTranslation` and `shetar.coordinates.Translation` for details on parameters.
    """

    _evaluate = staticmethod(_translations.coaxial_translation_intradomain_coefficients)
    _dtype = float
    from .bases import RegularRadialBase as _recurrence_initialization
    _default_output_type = expansions.InteriorExpansion


class ExteriorCoaxialTranslation(CoaxialTranslation):
    """Handles translations in the exterior domain.

    This applies translations from an exterior translation to an exterior
    translation.

    See `CoaxialTranslation` and `shetar.coordinates.Translation` for details on parameters.
    """

    _evaluate = staticmethod(_translations.coaxial_translation_intradomain_coefficients)
    _dtype = float
    from .bases import RegularRadialBase as _recurrence_initialization
    _default_output_type = expansions.ExteriorExpansion


class ExteriorInteriorCoaxialTranslation(CoaxialTranslation):
    """Handles translations in the between domains.

    This applies translations from an exterios translation to an interior
    translation.

    See `CoaxialTranslation` and `shetar.coordinates.Translation` for details on parameters.
    """

    _evaluate = staticmethod(_translations.coaxial_translation_interdomain_coefficients)
    _dtype = complex
    from .bases import SingularRadialBase as _recurrence_initialization
    _default_output_type = expansions.InteriorExpansion


class Translation(CoaxialTranslation):
    """Parent class for translations.

    This class should not be instantiated, only inherited from.

    See `shetar.coordinates.Translation` and `CoaxialTranslation` for details on parameters.
    """

    def __init__(self, input_order, output_order, position=None, wavenumber=None,
                 radius=None, colatitude=None, azimuth=None, defer_evaluation=False):
        coordinate = coordinates.Translation.parse_args(position=position, radius=radius, colatitude=colatitude, azimuth=azimuth)
        self._rotation = Rotation(
            order=max(input_order, output_order), defer_evaluation=True,
            colatitude=coordinate.colatitude, azimuth=coordinate.azimuth,
        )
        super().__init__(input_order=input_order, output_order=output_order, position=coordinate, wavenumber=wavenumber, defer_evaluation=defer_evaluation)

    def evaluate(self, position=None, wavenumber=None, radius=None, colatitude=None, azimuth=None):
        self.coordinate = coordinates.Translation.parse_args(position=position, radius=radius, colatitude=colatitude, azimuth=azimuth)
        if (position is not None) or (radius is not None) or (wavenumber is not None):
            super().evaluate(position=self.coordinate, wavenumber=wavenumber)
        if (position is not None) or (colatitude is not None) or (azimuth is not None):
            self._rotation.evaluate(colatitude=self.coordinate.colatitude, azimuth=self.coordinate.azimuth)
        return self

    def apply(self, expansion, inverse=False, _only_coaxial=False):
        if _only_coaxial:
            return super().apply(expansion, inverse=inverse)
        if not inverse:
            return expansion.apply(self._rotation, inverse=True).apply(self, _only_coaxial=True).apply(self._rotation)
        else:
            raise NotImplementedError('Inverse translations not implemented yet.')

    @property
    def shape(self):
        return _shapes.broadcast_shapes(self.coordinate.shape, self.wavenumber.shape)[0]

    def copy(self, deep=False):
        new_obj = super().copy(deep=deep)
        new_obj._rotation = self._rotation.copy(deep=deep)
        return new_obj


class InteriorTranslation(Translation, InteriorCoaxialTranslation):
    """Handles translations in the interior domain.

    This applies translations from an interior translation to an interior
    translation.

    See `CoaxialTranslation` and `shetar.coordinates.Translation` for details on parameters.
    """


class ExteriorTranslation(Translation, ExteriorCoaxialTranslation):
    """Handles translations in the exterior domain.

    This applies translations from an exterior translation to an exterior
    translation.

    See `CoaxialTranslation` and `shetar.coordinates.Translation` for details on parameters.
    """


class ExteriorInteriorTranslation(Translation, ExteriorInteriorCoaxialTranslation):
    """Handles translations in the between domains.

    This applies translations from an exterios translation to an interior
    translation.

    See `CoaxialTranslation` and `shetar.coordinates.Translation` for details on parameters.
    """
