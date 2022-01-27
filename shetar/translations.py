import numpy as np
from . import coordinates, rotations, expansions, _translations, _shapes


class CoaxialTranslation(coordinates.OwnerMixin):
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
    _evaluate = staticmethod(_translations.coaxial_translation_intradomain_coefficients)
    _dtype = float
    from .bases import RegularRadialBase as _recurrence_initialization
    _default_output_type = expansions.InteriorExpansion


class ExteriorCoaxialTranslation(CoaxialTranslation):
    _evaluate = staticmethod(_translations.coaxial_translation_intradomain_coefficients)
    _dtype = float
    from .bases import RegularRadialBase as _recurrence_initialization
    _default_output_type = expansions.ExteriorExpansion


class ExteriorInteriorCoaxialTranslation(CoaxialTranslation):
    _evaluate = staticmethod(_translations.coaxial_translation_interdomain_coefficients)
    _dtype = complex
    from .bases import SingularRadialBase as _recurrence_initialization
    _default_output_type = expansions.InteriorExpansion


class Translation(CoaxialTranslation):
    def __init__(self, input_order, output_order, position=None, wavenumber=None,
                 radius=None, colatitude=None, azimuth=None, defer_evaluation=False):
        coordinate = coordinates.Translation.parse_args(position=position, radius=radius, colatitude=colatitude, azimuth=azimuth)
        self._rotation = rotations.Rotation(
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
    pass


class ExteriorTranslation(Translation, ExteriorCoaxialTranslation):
    pass


class ExteriorInteriorTranslation(Translation, ExteriorInteriorCoaxialTranslation):
    pass
