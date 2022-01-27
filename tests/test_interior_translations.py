import numpy as np
import shetar
import pytest


# =========================== Test input parameters ============================
@pytest.fixture(params=[4, 7], scope='module')
def output_order(request):
    return request.param


@pytest.fixture(params=[0.1, 0.2], scope='module')
def new_origin(request):
    scale = request.param
    new_origin = np.random.uniform(low=-scale, high=scale, size=3)
    return new_origin


@pytest.fixture(params=[tuple(), (5, 7)], scope='module')
def field_points(request):
    size = request.param
    return np.random.uniform(low=-0.1, high=0.1, size=size + (3,))


@pytest.fixture(scope='module', params=[1, np.linspace(0.01, 1, 7)])
def wavenumber(request):
    return request.param


@pytest.fixture(params=[10, 30], scope='module')
def source_position(request):
    r = np.random.normal(loc=request.param, scale=0.1)
    phi = np.random.uniform(low=0, high=2 * np.pi)
    theta = np.random.uniform(low=0, high=np.pi)
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=-1)


@pytest.fixture(params=[1, 8], scope='module')
def num_initial_sources(request):
    return request.param


# ============================ Evaluated parameters ============================
@pytest.fixture(scope='module')
def input_order(output_order):
    return output_order - 2


@pytest.fixture(scope='module')
def source_expansion(input_order, source_position, num_initial_sources):
    source_amplitudes = np.random.normal(size=num_initial_sources) + 1j * np.random.normal(size=num_initial_sources)
    source_positions = source_position + np.random.normal(loc=0, scale=0.1, size=(num_initial_sources, 3))
    source_expansion = shetar.expansions.Expansion(data=source_amplitudes[None, :]).apply(
        shetar.transforms.InteriorTranslation(
            input_order=0, output_order=input_order,
            position=source_positions, wavenumber=1
        )
    )
    source_expansion._data = np.sum(source_expansion._data, axis=1)
    source_expansion._wavenumber = None
    return source_expansion


@pytest.fixture(scope='module')
def original_bases(input_order, field_points, wavenumber):
    original_bases = shetar.bases.RegularBase(order=input_order, position=field_points, wavenumber=wavenumber)
    return original_bases


@pytest.fixture(scope='module')
def original_values(original_bases, source_expansion):
    original_values = original_bases.apply(source_expansion)
    return original_values


@pytest.fixture(scope='module')
def translated_field_points(field_points, new_origin):
    translated_field_points = field_points - new_origin.reshape([1] * (np.ndim(field_points) - 1) + [3])
    return translated_field_points


@pytest.fixture(scope='module')
def translated_bases(translated_field_points, output_order, wavenumber):
    translated_bases = shetar.bases.RegularBase(order=output_order, position=translated_field_points, wavenumber=wavenumber)
    return translated_bases


@pytest.fixture(scope='module')
def translated_expansion(source_expansion, new_origin, output_order, wavenumber):
    translation = shetar.transforms.InteriorTranslation(
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
