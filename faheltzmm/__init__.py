"""FaHeltzMM, (Fast Hemlholtz Multipole Methods) is designed to solve the Helmholtz equation in three dimensions."""

from . import _version
__version__ = _version.__version__
del _version  # Keeps the namespace clean!


def _is_value(x):
    from numpy import broadcast
    return x is not None and type(x) is not broadcast


from . import generate, rotations, coordinates, translations, bases  # noqa: F401, E402
