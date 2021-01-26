"""FaHeltzMM, (Fast Hemlholtz Multipole Methods) is designed to solve the Helmholtz equation in three dimensions."""

from . import _version
__version__ = _version.__version__
del _version  # Keeps the namespace clean!


from . import rotations, coordinates, translations, bases  # noqa: F401, E402
