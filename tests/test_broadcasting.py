import numpy as np
import pytest
import shetar


# =================== Test input paramaters ===================
@pytest.fixture(scope='module', params=[3, 5])
def input_order(request):
    return request.param


@pytest.fixture(scope='module', params=[3, 5])
def output_order(request):
    return request.param


@pytest.fixture(scope='module', params=[
    1,  # Single value
    np.linspace(1e-3, 1, 7),  # Unique size for wavenumbers
    np.linspace(1e-3, 1, 9),  # Shared size for all
    # np.linspace(1e-3, 1, 11),  # Shared with only radius
])
def wavenumber(request):
    return request.param


@pytest.fixture(scope='module', params=[
    1,  # Single value
    np.linspace(0.1, 1, 9),  # Shared size for all
    np.linspace(0.1, 1, 11),  # Unique for radius
])
def radius(request):
    return request.param


@pytest.fixture(scope='module', params=[
    np.pi / 2,  # Single value
    np.linspace(0, np.pi, 9),  # Shared size for all
    np.linspace(0, np.pi, 13),
])
def colatitude(request):
    return request.param


@pytest.fixture(scope='module', params=[
    0,  # Single value
    np.linspace(0, 2 * np.pi, 9),  # Shared size for all
    np.linspace(0, 2 * np.pi, 15),
])
def azimuth(request):
    return request.param


# ============================ Evaluated parameters ============================
@pytest.fixture(scope='module')
def colatitude_azimuth(colatitude, azimuth):
    try:
        np.broadcast(colatitude, azimuth)
    except ValueError:
        if np.size(azimuth) == np.size(radius):
            azimuth = azimuth[:, None]
        else:
            colatitude = colatitude[:, None]
    return colatitude, azimuth


@pytest.fixture(scope='module')
def radius_colatitude_azimuth(radius, colatitude_azimuth):
    colatitude, azimuth = colatitude_azimuth
    angular_shape = np.broadcast(colatitude, azimuth)

    try:
        np.broadcast(radius, angular_shape)
    except ValueError:
        if np.size(radius) == angular_shape.shape[0]:
            radius = np.reshape(radius, radius.shape + (1,) * (angular_shape.ndim - 1))
        else:
            radius = np.reshape(radius, radius.shape + (1,) * angular_shape.ndim)
    return radius, colatitude, azimuth


# ================================== Helpers ===================================
def class_name(obj):
    return '.'.join([obj.__class__.__module__, obj.__class__.__name__])


def assert_shape_match(obj, matching):
    assert obj.shape == np.shape(matching), f"Object of type {class_name(obj)} has shape {obj.shape}, expected {np.shape(matching)}"
    assert np.shape(obj) == obj.shape, f"Object of type {class_name(obj)} has `np.shape(obj)`` {np.shape(obj)} and `obj.shape` {obj.shape}"
    assert obj.ndim == np.ndim(matching), f"Object of type {class_name(obj)} has ndim {obj.ndim}, expected {np.ndim(matching)}"
    assert np.ndim(obj) == obj.ndim, f"Object of type {class_name(obj)} has `np.ndim(obj)`` {np.ndim(obj)} and `obj.ndim` {obj.ndim}"


def assert_rehsapability(obj):
    new_shape = (1, 1) + np.shape(obj) + (1, 1)
    assert hasattr(obj, 'reshape'), f"Object of type {class_name(obj)} has no implemented reshape function"
    reshaped = np.reshape(obj, new_shape)
    assert reshaped.shape == new_shape, f"Object of type {class_name(obj)} reshaped to {reshaped.shape}, expected {new_shape}"


# ================================= Test bases =================================
def test_AssociatedLegendrePolynomials(input_order, colatitude):
    obj = shetar.bases.AssociatedLegendrePolynomials(order=input_order, x=np.cos(colatitude))
    assert_shape_match(obj, colatitude)
    assert_rehsapability(obj)


def test_RegularRadialBase(input_order, radius, wavenumber):
    obj = shetar.bases.RegularRadialBase(order=input_order, radius=radius, wavenumber=wavenumber)
    assert_shape_match(obj, radius)
    assert_rehsapability(obj)


def test_SingularRadialBase(input_order, radius, wavenumber):
    obj = shetar.bases.SingularRadialBase(order=input_order, radius=radius, wavenumber=wavenumber)
    assert_shape_match(obj, radius)
    assert_rehsapability(obj)


def test_DualRadialBase(input_order, radius, wavenumber):
    obj = shetar.bases.DualRadialBase(order=input_order, radius=radius, wavenumber=wavenumber)
    assert_shape_match(obj, radius)
    assert_rehsapability(obj)


def test_SphericalHarmonics(input_order, colatitude_azimuth):
    colatitude, azimuth = colatitude_azimuth
    obj = shetar.bases.SphericalHarmonics(order=input_order, colatitude=colatitude, azimuth=azimuth)
    assert_shape_match(obj._legendre, colatitude)
    assert_shape_match(obj._azimuth, azimuth)
    assert_shape_match(obj, np.broadcast(colatitude, azimuth))
    assert_rehsapability(obj)


def test_RegularBase(input_order, wavenumber, radius_colatitude_azimuth):
    radius, colatitude, azimuth = radius_colatitude_azimuth
    obj = shetar.bases.RegularBase(order=input_order, wavenumber=wavenumber, radius=radius, colatitude=colatitude, azimuth=azimuth)
    assert_shape_match(obj, np.broadcast(radius, colatitude, azimuth))
    assert_shape_match(obj._radial, radius)
    assert_shape_match(obj._angular, np.broadcast(colatitude, azimuth))
    assert_shape_match(obj._angular._legendre, colatitude)
    assert_shape_match(obj._angular._azimuth, azimuth)
    assert_rehsapability(obj)


def test_SingularBase(input_order, wavenumber, radius_colatitude_azimuth):
    radius, colatitude, azimuth = radius_colatitude_azimuth
    obj = shetar.bases.SingularBase(order=input_order, wavenumber=wavenumber, radius=radius, colatitude=colatitude, azimuth=azimuth)
    assert_shape_match(obj, np.broadcast(radius, colatitude, azimuth))
    assert_shape_match(obj._radial, radius)
    assert_shape_match(obj._angular, np.broadcast(colatitude, azimuth))
    assert_shape_match(obj._angular._legendre, colatitude)
    assert_shape_match(obj._angular._azimuth, azimuth)
    assert_rehsapability(obj)


def test_DualBase(input_order, wavenumber, radius_colatitude_azimuth):
    radius, colatitude, azimuth = radius_colatitude_azimuth
    obj = shetar.bases.DualBase(order=input_order, wavenumber=wavenumber, radius=radius, colatitude=colatitude, azimuth=azimuth)
    assert_shape_match(obj, np.broadcast(radius, colatitude, azimuth))
    assert_shape_match(obj._radial, radius)
    assert_shape_match(obj._angular, np.broadcast(colatitude, azimuth))
    assert_shape_match(obj._angular._legendre, colatitude)
    assert_shape_match(obj._angular._azimuth, azimuth)
    assert_rehsapability(obj)


# =============================== Test rotations ===============================
def test_ColatitudeRotation(input_order, colatitude):
    obj = shetar.rotations.ColatitudeRotation(order=input_order, colatitude=colatitude)
    assert_shape_match(obj, colatitude)
    assert_rehsapability(obj)


def test_Rotation(input_order, colatitude_azimuth):
    colatitude, azimuth = colatitude_azimuth
    obj = shetar.rotations.Rotation(order=input_order, colatitude=colatitude, azimuth=azimuth)
    assert_shape_match(obj, np.broadcast(colatitude, azimuth))
    assert_shape_match(obj._primary_phase, azimuth)
    assert_rehsapability(obj)


# ============================= Test translations ==============================
def test_InteriorCoaxialTranslation(input_order, output_order, radius, wavenumber):
    obj = shetar.translations.InteriorCoaxialTranslation(input_order=input_order, output_order=output_order, distance=radius, wavenumber=wavenumber)
    assert_shape_match(obj, radius)
    assert_rehsapability(obj)


def test_ExteriorCoaxialTranslation(input_order, output_order, radius, wavenumber):
    obj = shetar.translations.ExteriorCoaxialTranslation(input_order=input_order, output_order=output_order, distance=radius, wavenumber=wavenumber)
    assert_shape_match(obj, radius)
    assert_rehsapability(obj)


def test_ExteriorInteriorCoaxialTranslation(input_order, output_order, radius, wavenumber):
    obj = shetar.translations.ExteriorInteriorCoaxialTranslation(input_order=input_order, output_order=output_order, distance=radius, wavenumber=wavenumber)
    assert_shape_match(obj, radius)
    assert_rehsapability(obj)


def test_InteriorTranslation(input_order, output_order, radius_colatitude_azimuth, wavenumber):
    radius, colatitude, azimuth = radius_colatitude_azimuth
    obj = shetar.translations.InteriorTranslation(input_order=input_order, output_order=output_order, wavenumber=wavenumber, radius=radius, colatitude=colatitude, azimuth=azimuth)
    assert_shape_match(obj, np.broadcast(radius, colatitude, azimuth))
    assert_shape_match(obj._coaxial, radius)
    assert_shape_match(obj._rotation, np.broadcast(colatitude, azimuth))
    assert_rehsapability(obj)


def test_ExteriorTranslation(input_order, output_order, radius_colatitude_azimuth, wavenumber):
    radius, colatitude, azimuth = radius_colatitude_azimuth
    obj = shetar.translations.ExteriorTranslation(input_order=input_order, output_order=output_order, wavenumber=wavenumber, radius=radius, colatitude=colatitude, azimuth=azimuth)
    assert_shape_match(obj, np.broadcast(radius, colatitude, azimuth))
    assert_shape_match(obj._coaxial, radius)
    assert_shape_match(obj._rotation, np.broadcast(colatitude, azimuth))
    assert_rehsapability(obj)


def test_ExteriorInteriorTranslation(input_order, output_order, radius_colatitude_azimuth, wavenumber):
    radius, colatitude, azimuth = radius_colatitude_azimuth
    obj = shetar.translations.ExteriorInteriorTranslation(input_order=input_order, output_order=output_order, wavenumber=wavenumber, radius=radius, colatitude=colatitude, azimuth=azimuth)
    assert_shape_match(obj, np.broadcast(radius, colatitude, azimuth))
    assert_shape_match(obj._coaxial, radius)
    assert_shape_match(obj._rotation, np.broadcast(colatitude, azimuth))
    assert_rehsapability(obj)
