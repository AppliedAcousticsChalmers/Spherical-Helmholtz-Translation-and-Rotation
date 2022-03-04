import numpy as np
import pytest
import shetar

shared_size = 9


# =================== Test input paramaters ===================
@pytest.fixture(scope='module', params=[3, 5])
def input_order(request):
    return request.param


@pytest.fixture(scope='module', params=[3, 5])
def output_order(request):
    return request.param


@pytest.fixture(scope='module', params=[
    1,  # Single value
    np.linspace(1e-3, 1, shared_size),  # Shared size for all
    np.linspace(1e-3, 1, 7),  # Unique size for wavenumbers
])
def wavenumber(request):
    return request.param


@pytest.fixture(scope='module', params=[
    1,  # Single value
    np.linspace(0.1, 1, shared_size),  # Shared size for all
    np.linspace(0.1, 1, 11),  # Unique for radius
])
def radius(request):
    return request.param


@pytest.fixture(scope='module', params=[
    np.pi / 2,  # Single value
    np.linspace(0, np.pi, shared_size),  # Shared size for all
    np.linspace(0, np.pi, 13),
])
def colatitude(request):
    return request.param


@pytest.fixture(scope='module', params=[
    0,  # Single value
    np.linspace(0, 2 * np.pi, shared_size),  # Shared size for all
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
        azimuth = azimuth[:, None]
    return colatitude, azimuth


@pytest.fixture(scope='module')
def wavenumber_radius(wavenumber, radius):
    try:
        np.broadcast(wavenumber, radius)
    except ValueError:
        radius = radius[..., None]
    return wavenumber, radius


@pytest.fixture(scope='module')
def wavenumber_radius_colatitude_azimuth(wavenumber, radius, colatitude, azimuth):
    try:
        np.broadcast(wavenumber, radius, colatitude, azimuth)
    except ValueError:
        axes = 1
        if np.size(radius) not in (1, shared_size):
            radius = radius.reshape((-1,) + (1,) * axes)
            axes += 1
        if np.size(wavenumber) not in (1, shared_size):
            wavenumber = wavenumber.reshape((-1,) + (1,) * axes)
            axes += 1
        if np.size(colatitude) not in (1, shared_size):
            colatitude = colatitude.reshape((-1,) + (1,) * axes)
            axes += 1
        if np.size(azimuth) not in (1, shared_size):
            azimuth = azimuth.reshape((-1,) + (1,) * axes)
            axes += 1
    return wavenumber, radius, colatitude, azimuth


# ================================== Helpers ===================================
def class_name(obj):
    return '.'.join([obj.__class__.__module__, obj.__class__.__name__])


def assert_shape_match(obj, matching):
    assert obj.shape == np.shape(matching), f"Object of type {class_name(obj)} has shape {obj.shape}, expected {np.shape(matching)}"
    assert np.shape(obj) == obj.shape, f"Object of type {class_name(obj)} has `np.shape(obj)`` {np.shape(obj)} and `obj.shape` {obj.shape}"
    assert obj.ndim == np.ndim(matching), f"Object of type {class_name(obj)} has ndim {obj.ndim}, expected {np.ndim(matching)}"
    assert np.ndim(obj) == obj.ndim, f"Object of type {class_name(obj)} has `np.ndim(obj)`` {np.ndim(obj)} and `obj.ndim` {obj.ndim}"


# ================================= Test bases =================================
def test_AssociatedLegendrePolynomials(input_order, colatitude):
    obj = shetar.bases.AssociatedLegendrePolynomials(order=input_order, x=np.cos(colatitude))
    assert_shape_match(obj, colatitude)


def test_RegularRadialBase(input_order, wavenumber_radius):
    wavenumber, radius = wavenumber_radius
    obj = shetar.bases.RegularRadialBase(order=input_order, radius=radius, wavenumber=wavenumber)
    assert_shape_match(obj, np.broadcast(radius, wavenumber))


def test_SingularRadialBase(input_order, wavenumber_radius):
    wavenumber, radius = wavenumber_radius
    obj = shetar.bases.SingularRadialBase(order=input_order, radius=radius, wavenumber=wavenumber)
    assert_shape_match(obj, np.broadcast(radius, wavenumber))


def test_DualRadialBase(input_order, wavenumber_radius):
    wavenumber, radius = wavenumber_radius
    obj = shetar.bases.DualRadialBase(order=input_order, radius=radius, wavenumber=wavenumber)
    assert_shape_match(obj, np.broadcast(radius, wavenumber))


def test_SphericalHarmonics(input_order, colatitude_azimuth):
    colatitude, azimuth = colatitude_azimuth
    obj = shetar.bases.SphericalHarmonics(order=input_order, colatitude=colatitude, azimuth=azimuth)
    assert_shape_match(obj._legendre, colatitude)
    assert_shape_match(obj._phase, azimuth)
    assert_shape_match(obj, np.broadcast(colatitude, azimuth))


def test_RegularBase(input_order, wavenumber_radius_colatitude_azimuth):
    wavenumber, radius, colatitude, azimuth = wavenumber_radius_colatitude_azimuth
    obj = shetar.bases.RegularBase(order=input_order, wavenumber=wavenumber, radius=radius, colatitude=colatitude, azimuth=azimuth)
    assert_shape_match(obj, np.broadcast(wavenumber, radius, colatitude, azimuth))
    assert_shape_match(obj._radial, np.broadcast(radius, wavenumber))
    assert_shape_match(obj._angular, np.broadcast(colatitude, azimuth))
    assert_shape_match(obj._angular._legendre, colatitude)
    assert_shape_match(obj._angular._phase, azimuth)


def test_SingularBase(input_order, wavenumber_radius_colatitude_azimuth):
    wavenumber, radius, colatitude, azimuth = wavenumber_radius_colatitude_azimuth
    obj = shetar.bases.SingularBase(order=input_order, wavenumber=wavenumber, radius=radius, colatitude=colatitude, azimuth=azimuth)
    assert_shape_match(obj, np.broadcast(wavenumber, radius, colatitude, azimuth))
    assert_shape_match(obj._radial, np.broadcast(radius, wavenumber))
    assert_shape_match(obj._angular, np.broadcast(colatitude, azimuth))
    assert_shape_match(obj._angular._legendre, colatitude)
    assert_shape_match(obj._angular._phase, azimuth)


def test_DualBase(input_order, wavenumber_radius_colatitude_azimuth):
    wavenumber, radius, colatitude, azimuth = wavenumber_radius_colatitude_azimuth
    obj = shetar.bases.DualBase(order=input_order, wavenumber=wavenumber, radius=radius, colatitude=colatitude, azimuth=azimuth)
    assert_shape_match(obj, np.broadcast(wavenumber, radius, colatitude, azimuth))
    assert_shape_match(obj._radial, np.broadcast(radius, wavenumber))
    assert_shape_match(obj._angular, np.broadcast(colatitude, azimuth))
    assert_shape_match(obj._angular._legendre, colatitude)
    assert_shape_match(obj._angular._phase, azimuth)


# =============================== Test rotations ===============================
def test_ColatitudeRotation(input_order, colatitude):
    obj = shetar.transforms.ColatitudeRotation(order=input_order, colatitude=colatitude)
    assert_shape_match(obj, colatitude)


def test_Rotation(input_order, colatitude_azimuth):
    colatitude, azimuth = colatitude_azimuth
    obj = shetar.transforms.Rotation(order=input_order, colatitude=colatitude, azimuth=azimuth)
    assert_shape_match(obj, np.broadcast(colatitude, azimuth))
    assert_shape_match(obj._primary_phase, azimuth)


# ============================= Test translations ==============================
def test_InteriorCoaxialTranslation(input_order, output_order, wavenumber_radius):
    wavenumber, radius = wavenumber_radius
    obj = shetar.transforms.InteriorCoaxialTranslation(orders=(input_order, output_order), radius=radius, wavenumber=wavenumber)
    assert_shape_match(obj, np.broadcast(radius, wavenumber))


def test_ExteriorCoaxialTranslation(input_order, output_order, wavenumber_radius):
    wavenumber, radius = wavenumber_radius
    obj = shetar.transforms.ExteriorCoaxialTranslation(orders=(input_order, output_order), radius=radius, wavenumber=wavenumber)
    assert_shape_match(obj, np.broadcast(radius, wavenumber))


def test_ExteriorInteriorCoaxialTranslation(input_order, output_order, wavenumber_radius):
    wavenumber, radius = wavenumber_radius
    obj = shetar.transforms.ExteriorInteriorCoaxialTranslation(orders=(input_order, output_order), radius=radius, wavenumber=wavenumber)
    assert_shape_match(obj, np.broadcast(radius, wavenumber))


def test_InteriorTranslation(input_order, output_order, wavenumber_radius_colatitude_azimuth):
    wavenumber, radius, colatitude, azimuth = wavenumber_radius_colatitude_azimuth
    obj = shetar.transforms.InteriorTranslation(orders=(input_order, output_order), wavenumber=wavenumber, radius=radius, colatitude=colatitude, azimuth=azimuth)
    assert_shape_match(obj, np.broadcast(wavenumber, radius, colatitude, azimuth))
    assert_shape_match(obj._rotation, np.broadcast(colatitude, azimuth))


def test_ExteriorTranslation(input_order, output_order, wavenumber_radius_colatitude_azimuth):
    wavenumber, radius, colatitude, azimuth = wavenumber_radius_colatitude_azimuth
    obj = shetar.transforms.ExteriorTranslation(orders=(input_order, output_order), wavenumber=wavenumber, radius=radius, colatitude=colatitude, azimuth=azimuth)
    assert_shape_match(obj, np.broadcast(wavenumber, radius, colatitude, azimuth))
    assert_shape_match(obj._rotation, np.broadcast(colatitude, azimuth))


def test_ExteriorInteriorTranslation(input_order, output_order, wavenumber_radius_colatitude_azimuth):
    wavenumber, radius, colatitude, azimuth = wavenumber_radius_colatitude_azimuth
    obj = shetar.transforms.ExteriorInteriorTranslation(orders=(input_order, output_order), wavenumber=wavenumber, radius=radius, colatitude=colatitude, azimuth=azimuth)
    assert_shape_match(obj, np.broadcast(wavenumber, radius, colatitude, azimuth))
    assert_shape_match(obj._rotation, np.broadcast(colatitude, azimuth))
