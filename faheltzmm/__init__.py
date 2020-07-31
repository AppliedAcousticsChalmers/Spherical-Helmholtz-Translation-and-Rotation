"""FaHeltzMM, (Fast Hemlholtz Multipole Methods) is designed to solve the Helmholtz equation in three dimensions."""

from . import _version, generate, rotations, coordinates, translations, bases
__version__ = _version.__version__
del _version  # Keeps the namespace clean!
