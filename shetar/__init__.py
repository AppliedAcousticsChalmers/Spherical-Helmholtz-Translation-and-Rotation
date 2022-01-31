"""SHeTaR, (Spherical Helmholtz Translation and Rotation), translations and rotations for solutions of the Helmholtz equation in spherical coordinates."""

from . import _version
__version__ = _version.__version__
del _version  # Keeps the namespace clean!


from . import coordinates, bases, expansions, transforms  # noqa: F401, E402
