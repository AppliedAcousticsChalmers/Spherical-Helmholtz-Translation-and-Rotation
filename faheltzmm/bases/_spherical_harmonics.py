import numpy as np
from ._legendre import AssociatedLegendrePolynomials


class SphericalHarmonics:
    def __init__(self, order, colatitude=None, azimuth=None, cosine_colatitude=None, shape=None):
        shape = shape if shape is not None else np.broadcast(cosine_colatitude if cosine_colatitude is not None else colatitude, azimuth).shape
        self._legendre = AssociatedLegendrePolynomials(order, shape=shape, normalization='orthonormal')
        if azimuth is not None:
            self.evaluate(colatitude=colatitude, azimuth=azimuth, cosine_colatitude=cosine_colatitude)

    def evaluate(self, colatitude=None, azimuth=None, cosine_colatitude=None):
        cosine_colatitude = np.cos(colatitude) if cosine_colatitude is None else cosine_colatitude
        self._legendre.evaluate(cosine_colatitude)
        self._azimuth = azimuth if np.iscomplexobj(azimuth) else np.exp(1j * azimuth)

    def __getitem__(self, key):
        order, mode = key
        return self._legendre[order, mode] * self._azimuth ** mode / (2 * np.pi)**0.5

    @property
    def num_unique(self):
        return self._legendre.num_unique

    @property
    def order(self):
        return self._legendre.order

    @property
    def shape(self):
        return self._legendre.shape
