import numpy as np
import shetar.translations
import shetar.coordinates
import shetar.bases
import pytest


# =========================== Test input parameters ============================
@pytest.fixture(params=[9, 12], scope='module')
def output_order(request):
    return request.param


@pytest.fixture(params=[6, 9], scope='module')
def new_origin(request):
    r = request.param
    r = np.random.normal(loc=r, scale=r / 10)
    phi = np.random.uniform(low=0, high=2 * np.pi)
    theta = np.random.uniform(low=0, high=np.pi)
    new_origin = r * np.stack([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])
    return new_origin


@pytest.fixture(scope='module', params=[1, np.linspace(0.01, 1, 7)])
def wavenumber(request):
    return request.param


@pytest.fixture(params=[1, 8], scope='module')
def num_initial_sources(request):
    return request.param


@pytest.fixture(params=[tuple(), (5, 7)], scope='module')
def field_point_shape(request):
    return request.param


# ============================ Evaluated parameters ============================
@pytest.fixture(scope='module')
def input_order(output_order, new_origin):
    return output_order - 2


@pytest.fixture(scope='module')
def source_expansion(input_order, num_initial_sources):
    source_amplitudes = np.random.normal(size=num_initial_sources) + 1j * np.random.normal(size=num_initial_sources)
    source_positions = np.random.normal(loc=0, scale=1e-3, size=(3, num_initial_sources))
    source_expansion = shetar.expansions.Expansion(data=source_amplitudes[None, :]).apply(
        shetar.translations.InteriorTranslation(
            input_order=0, output_order=input_order,
            position=source_positions, wavenumber=1
        )
    )
    source_expansion._data = np.sum(source_expansion._data, axis=1)
    source_expansion._wavenumber = None
    return source_expansion


@pytest.fixture(scope='module')
def original_bases(input_order, field_points, wavenumber):
    original_bases = shetar.bases.SingularBase(order=input_order, position=field_points, wavenumber=wavenumber)
    return original_bases


@pytest.fixture(scope='module')
def original_values(original_bases, source_expansion):
    original_values = original_bases.apply(source_expansion)
    return original_values


@pytest.fixture(scope='module')
def field_points(translated_field_points, new_origin):
    field_points = translated_field_points + new_origin.reshape([3] + [1] * (np.ndim(translated_field_points) - 1))
    return field_points


@pytest.fixture(scope='module')
def translated_field_points(field_point_shape):
    r = np.random.uniform(low=0.1, high=1, size=field_point_shape)
    phi = np.random.uniform(low=0, high=2 * np.pi, size=field_point_shape)
    theta = np.random.uniform(low=0, high=np.pi, size=field_point_shape)
    translated_field_points = r * np.stack([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])
    return translated_field_points


@pytest.fixture(scope='module')
def translated_bases(translated_field_points, output_order, wavenumber):
    translated_bases = shetar.bases.RegularBase(order=output_order, position=translated_field_points, wavenumber=wavenumber)
    return translated_bases


@pytest.fixture(scope='module')
def translated_expansion(source_expansion, new_origin, output_order, wavenumber):
    translation = shetar.translations.ExteriorInteriorTranslation(
        input_order=source_expansion.order, output_order=output_order,
        position=-new_origin, wavenumber=wavenumber
    )
    translated_expansion = source_expansion.apply(translation)
    return translated_expansion


@pytest.fixture(scope='module')
def translated_values(translated_bases, translated_expansion):
    translated_values = translated_bases.apply(translated_expansion)
    return translated_values


# ================================ Actual tests ================================
def test_values(original_values, translated_values):
    np.testing.assert_allclose(original_values, translated_values, rtol=1e-6)
