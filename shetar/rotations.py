import numpy as np
from . import coordinates, expansions
from . import _rotations


class ColatitudeRotation(coordinates.OwnerMixin):
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
    # Subclass of ColatitudeRotation to get access to the `apply` method, which work the same for both types of rotation.
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
