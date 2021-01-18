import numpy as np
import faheltzmm.translations
import faheltzmm.coordinates
import faheltzmm.bases
import pytest


# =========================== Test input parameters ============================
@pytest.fixture(params=[6, 9], scope='module')
def output_order(request):
    return request.param


@pytest.fixture(params=[0.1, 0.2], scope='module')
def new_origin(request):
    scale = request.param
    r = np.random.uniform(low=1, high=2) * scale
    phi = np.random.uniform(low=0, high=2 * np.pi)
    theta = np.random.uniform(low=0, high=np.pi)
    new_origin = r * np.stack([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])
    return new_origin


@pytest.fixture(params=[tuple(), (5, 7)], scope='module')
def field_points(request):
    size = request.param
    r = np.random.uniform(low=5, high=50, size=size)
    phi = np.random.uniform(low=0, high=2 * np.pi, size=size)
    theta = np.random.uniform(low=0, high=np.pi, size=size)
    field_points = r * np.stack([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])
    return field_points


@pytest.fixture(scope='module', params=[1, np.linspace(0.01, 1, 7)])
def wavenumber(request):
    return request.param


@pytest.fixture(params=[1, 8], scope='module')
def num_initial_sources(request):
    return request.param


# ============================ Evaluated parameters ============================
@pytest.fixture(scope='module')
def input_order(output_order):
    return output_order - 4


@pytest.fixture(scope='module')
def source_expansion(input_order, num_initial_sources):
    source_amplitudes = np.random.normal(size=num_initial_sources) + 1j * np.random.normal(size=num_initial_sources)
    source_positions = np.random.normal(loc=0, scale=1e-3, size=(3, num_initial_sources))
    source_expansion = faheltzmm.expansions.Expansion(data=source_amplitudes[None, :]).apply(
        faheltzmm.translations.InteriorTranslation(
            input_order=0, output_order=input_order,
            position=source_positions, wavenumber=1
        )
    )
    source_expansion._data = np.sum(source_expansion._data, axis=1)
    source_expansion._wavenumber = None
    return source_expansion


@pytest.fixture(scope='module')
def original_bases(input_order, field_points, wavenumber):
    original_bases = faheltzmm.bases.SingularBase(order=input_order, position=field_points, wavenumber=wavenumber)
    return original_bases


@pytest.fixture(scope='module')
def original_values(original_bases, source_expansion):
    original_values = original_bases.apply(source_expansion)
    return original_values


@pytest.fixture(scope='module')
def translated_field_points(field_points, new_origin):
    translated_field_points = field_points - new_origin.reshape([3] + [1] * (np.ndim(field_points) - 1))
    return translated_field_points


@pytest.fixture(scope='module')
def translated_bases(translated_field_points, output_order, wavenumber):
    translated_bases = faheltzmm.bases.SingularBase(order=output_order, position=translated_field_points, wavenumber=wavenumber)
    return translated_bases


@pytest.fixture(scope='module')
def translated_expansion(source_expansion, new_origin, output_order, wavenumber):
    translation = faheltzmm.translations.ExteriorTranslation(
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
