# %%
import numpy as np
import faheltzmm
import scipy.special
import plotly.graph_objects as go

# %%
max_order = 20
wavenumber = 2 * np.pi
wavelength = 2 * np.pi / wavenumber
resolution = wavelength / 20
x_range = (-13 * wavelength, 13 * wavelength)
z_range = (-7 * wavelength, 7 * wavelength)
scatterer_a_position = np.array([-6.3, 0, 0]) * wavelength
scatterer_b_position = np.array([6.3, 0, 0]) * wavelength

fig_layout = dict(
    width=1000, height=500,
    xaxis=dict(range=x_range),
    yaxis=dict(range=z_range),
)

x = np.linspace(min(x_range), max(x_range), int((max(x_range) - min(x_range)) / resolution))
z = np.linspace(min(z_range), max(z_range), int((max(z_range) - min(z_range)) / resolution))

nx = int((max(x_range) - min(x_range)) / resolution)
nz = int((max(z_range) - min(z_range)) / resolution)
nx += (nx + 1) % 2
nz += (nz + 1) % 2
x = np.linspace(min(x_range), max(x_range), nx)
z = np.linspace(min(z_range), max(z_range), nz)
y = 0 * wavelength
x_mesh, y_mesh, z_mesh = mesh = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=0).squeeze()
exterior_bases_a = faheltzmm.generate.spherical_base_all(max_order=max_order, position=mesh - scatterer_a_position[:, None, None], wavenumber=wavenumber, domain='exterior')
exterior_bases_b = faheltzmm.generate.spherical_base_all(max_order=max_order, position=mesh - scatterer_b_position[:, None, None], wavenumber=wavenumber, domain='exterior')

# %%
source_ring_radius = 0.4 * wavelength
source_position = (np.random.uniform(-0.5, 0.5, 3) * 0 + [0, 0, -8]) * wavelength

source_coefficients = np.zeros((max_order + 1, 2 * max_order + 1), complex)
n = np.arange(max_order + 1)
source_coefficients[:, 0] = np.pi * source_ring_radius * faheltzmm.generate.spherical_jn_all(max_order, wavenumber * source_ring_radius) * (2 * n + 1)**0.5 / (scipy.special.gamma(n / 2 + 1) * scipy.special.gamma(0.5 - n / 2))
exterior_bases_source = faheltzmm.generate.spherical_base_all(max_order=max_order, position=mesh - source_position[:, None, None], wavenumber=wavenumber, domain='exterior')
input_field = np.einsum('nm, nm...', source_coefficients, exterior_bases_source)

# %%
# Define the scatterer positions, size, and scattering properties
radius_a = 5.837345 * wavelength
radius_b = 5.834934 * wavelength
relative_position = scatterer_b_position - scatterer_a_position
scatter_distance = np.sum(relative_position**2)**0.5

inside_scatterers = (((mesh - scatterer_a_position[:, None, None])**2).sum(axis=0) <= radius_a**2) | (((mesh - scatterer_b_position[:, None, None])**2).sum(axis=0) <= radius_b**2)

scattering_coefficients_a = - faheltzmm.generate.spherical_jn_all(max_order, wavenumber * radius_a, derivative=True) / faheltzmm.generate.spherical_hn_all(max_order, wavenumber * radius_a, derivative=True)
scattering_coefficients_b = - faheltzmm.generate.spherical_jn_all(max_order, wavenumber * radius_b, derivative=True) / faheltzmm.generate.spherical_hn_all(max_order, wavenumber * radius_b, derivative=True)

# Calculate the re-expansion coefficients between the two scatterers
rotation_coefficients = faheltzmm.rotations.rotation_coefficients(max_order, new_z_axis=relative_position)
translation_coefficients = faheltzmm.translations.coaxial_translation_coefficients(max_order, max_order, scatter_distance, wavenumber, 'exterior', 'interior')

# Rotate the input coefficients to point the z-axis in the desired orientation
input_coefficients_a = faheltzmm.translations.translate(source_coefficients, source_position - scatterer_a_position, wavenumber, 'exterior', 'interior')
input_coefficients_b = faheltzmm.translations.translate(source_coefficients, source_position - scatterer_b_position, wavenumber, 'exterior', 'interior')
input_coefficients_a = faheltzmm.rotations.rotate(input_coefficients_a, rotation_coefficients)
input_coefficients_b = faheltzmm.rotations.rotate(input_coefficients_b, rotation_coefficients)

# Scatter the field
scattered_a = input_coefficients_a * scattering_coefficients_a[:, None]
scattered_b = input_coefficients_b * scattering_coefficients_b[:, None]
scattered_ab = faheltzmm.translations.coaxial_translation(scattered_a, translation_coefficients, inverse=True) * scattering_coefficients_b[:, None]
scattered_ba = faheltzmm.translations.coaxial_translation(scattered_b, translation_coefficients) * scattering_coefficients_a[:, None]
scattered_aba = faheltzmm.translations.coaxial_translation(scattered_ab, translation_coefficients) * scattering_coefficients_a[:, None]
scattered_bab = faheltzmm.translations.coaxial_translation(scattered_ba, translation_coefficients, inverse=True) * scattering_coefficients_b[:, None]

# Rotate the scattered coefficlents back to the original z-axis
scattered_a = faheltzmm.rotations.rotate(scattered_a, rotation_coefficients, inverse=True)
scattered_b = faheltzmm.rotations.rotate(scattered_b, rotation_coefficients, inverse=True)
scattered_ab = faheltzmm.rotations.rotate(scattered_ab, rotation_coefficients, inverse=True)
scattered_ba = faheltzmm.rotations.rotate(scattered_ba, rotation_coefficients, inverse=True)
scattered_aba = faheltzmm.rotations.rotate(scattered_aba, rotation_coefficients, inverse=True)
scattered_bab = faheltzmm.rotations.rotate(scattered_bab, rotation_coefficients, inverse=True)

scattered_a_field = np.einsum('nm, nm...', scattered_a, exterior_bases_a)
scattered_b_field = np.einsum('nm, nm...', scattered_b, exterior_bases_b)
scattered_ab_field = np.einsum('nm, nm...', scattered_ab, exterior_bases_b)
scattered_ba_field = np.einsum('nm, nm...', scattered_ba, exterior_bases_a)
scattered_aba_field = np.einsum('nm, nm...', scattered_ab, exterior_bases_a)
scattered_bab_field = np.einsum('nm, nm...', scattered_ba, exterior_bases_b)

# %%
# plot_func = np.real
plot_func = lambda x: 20 * np.log10(np.abs(x))
plot_opts = dict(
    colorscale='viridis', showscale=False,
    zmin=np.mean(plot_func(input_field)) - 2 * np.std(plot_func(input_field)),
    zmax=np.mean(plot_func(input_field)) + 2 * np.std(plot_func(input_field)),
    zsmooth='best',
    transpose=True,
)

# %%
plot_field = input_field * 1
plot_field[inside_scatterers] = None
go.Figure([
    go.Heatmap(x=x, y=z, z=plot_func(plot_field), **plot_opts),
], fig_layout).show()

# %%
plot_field = 1 * input_field + 1 * scattered_a_field + 1 * scattered_b_field
plot_field[inside_scatterers] = None
go.Figure([
    go.Heatmap(x=x, y=z, z=plot_func(plot_field), **plot_opts),
], fig_layout).show()

# %%
plot_field = 1 * input_field + 1 * scattered_a_field + 1 * scattered_b_field + 1 * scattered_ab_field + 1 * scattered_ba_field
plot_field[inside_scatterers] = None
go.Figure([
    go.Heatmap(x=x, y=z, z=plot_func(plot_field), **plot_opts),
], fig_layout).show()

# %%
plot_field = 1 * input_field + 1 * scattered_a_field + 1 * scattered_b_field + 1 * scattered_ab_field + 1 * scattered_ba_field + 1 * scattered_aba_field + 1 * scattered_bab_field
plot_field[inside_scatterers] = None
go.Figure([
    go.Heatmap(x=x, y=z, z=plot_func(plot_field), **plot_opts),
], fig_layout).show()
