# %%
import numpy as np
import faheltzmm
import plotly.graph_objects as go

# %% Spatial points, original coefficients, etc.
max_order = 40
wavenumber = 2 * np.pi
wavelength = 2 * np.pi / wavenumber
translation_distance = 5.1237312 * wavelength

resolution = wavelength / 20
x_range = (2 * wavelength, 7 * wavelength)
z_range = (-7 * wavelength, 3 * wavelength)

fig_layout = dict(
    width=1000, height=500,
)

new_fig_layout = dict(
    xaxis=dict(range=z_range),
    yaxis=dict(range=x_range),
    **fig_layout
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
ξ, η, ζ = x, y, z + translation_distance

old_fig_layout = dict(
    fig_layout,
    xaxis=dict(range=(min(z), max(z))),
    yaxis=dict(range=(min(x), max(x))),
)
new_fig_layout = dict(
    fig_layout,
    xaxis=dict(range=(min(ζ), max(ζ))),
    yaxis=dict(range=(min(ξ), max(ξ))),
)

x_mesh, y_mesh, z_mesh = old_mesh = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=0).squeeze()
ξ_mesh, η_mesh, ζ_mesh = new_mesh = np.stack(np.meshgrid(ξ, η, ζ, indexing='ij'), axis=0).squeeze()
old_bases = faheltzmm.generate.spherical_base_all(max_order=max_order, position=old_mesh, wavenumber=wavenumber, domain='exterior')
new_bases_interior = faheltzmm.generate.spherical_base_all(max_order=max_order, position=new_mesh, wavenumber=wavenumber, domain='interior')
new_bases_exterior = faheltzmm.generate.spherical_base_all(max_order=max_order, position=new_mesh, wavenumber=wavenumber, domain='exterior')


# %%
source_positions = np.array([[-0.1, 0.1, 0.1], [0.1, -0.1, -0.1]]).T
source_amplitudes = np.array([1, np.exp(0.5j * np.pi)])
old_coefficients = faheltzmm.generate.spherical_base_all(max_order=max_order, domain='interior', wavenumber=wavenumber, position=source_positions).conj().dot(source_amplitudes)

interior_reexpansion_coefficients = faheltzmm.translations.coaxial_translation_coefficients(max_order, max_order, translation_distance, wavenumber, 'exterior', 'interior')
exterior_reexpansion_coefficients = faheltzmm.translations.coaxial_translation_coefficients(max_order, max_order, translation_distance, wavenumber, 'exterior', 'exterior')
new_coefficients_interior = np.einsum('mnp, nm -> pm', interior_reexpansion_coefficients, old_coefficients)
new_coefficients_exterior = np.einsum('mnp, nm -> pm', exterior_reexpansion_coefficients, old_coefficients)

old_coefficients_old_bases = np.einsum('nm, nm...', old_coefficients, old_bases)
new_coefficients_new_bases_interior = np.einsum('nm, nm...', new_coefficients_interior, new_bases_interior)
new_coefficients_new_bases_exterior = np.einsum('nm, nm...', new_coefficients_exterior, new_bases_exterior)
new_coefficients_new_bases = np.where(np.sum(new_mesh**2, axis=0) < translation_distance ** 2, new_coefficients_new_bases_interior, new_coefficients_new_bases_exterior)

far_field = np.where(np.sum(old_mesh**2, axis=0)**0.5 > 2 * wavelength)
color_min = np.mean(old_coefficients_old_bases[far_field].real) - 2 * np.std(old_coefficients_old_bases[far_field].real)
color_max = np.mean(old_coefficients_old_bases[far_field].real) + 2 * np.std(old_coefficients_old_bases[far_field].real)
np.allclose(old_coefficients_old_bases, new_coefficients_new_bases)

# %% Visualize the original coefficients in the original coordinates
go.Figure([
    go.Heatmap(x=z, y=x, z=np.real(old_coefficients_old_bases), showscale=False, colorscale='viridis', zmin=color_min, zmax=color_max, zsmooth='best'),
], old_fig_layout).show()

# %% Visualize the translated coefficients in the translated coordinates
go.Figure([
    go.Heatmap(x=ζ, y=ξ, z=np.real(new_coefficients_new_bases), showscale=False, colorscale='viridis', zmin=color_min, zmax=color_max, zsmooth='best'),
], new_fig_layout).show()

# %% Visualize the normalized field difference
go.Figure([
    go.Heatmap(x=ζ, y=ξ, z=np.real(new_coefficients_new_bases - old_coefficients_old_bases) / np.abs(old_coefficients_old_bases), showscale=True, colorscale='delta', zmin=-1.2, zmax=1.2, zsmooth='best'),
], new_fig_layout).show()

# %% Visualize the interior part of the field
go.Figure([
    go.Heatmap(x=ζ, y=ξ, z=np.real(new_coefficients_new_bases_interior), showscale=False, colorscale='viridis', zmin=color_min, zmax=color_max, zsmooth='best'),
    go.Scatter(x=translation_distance * np.cos(np.linspace(0, 2 * np.pi)), y=translation_distance * np.sin(np.linspace(0, 2 * np.pi))),
], new_fig_layout).show()

# %% Visualize the exterior part of the field
go.Figure([
    go.Heatmap(x=ζ, y=ξ, z=np.real(new_coefficients_new_bases_exterior), showscale=False, colorscale='viridis', zmin=color_min, zmax=color_max, zsmooth='best'),
    go.Scatter(x=translation_distance * np.cos(np.linspace(0, 2 * np.pi)), y=translation_distance * np.sin(np.linspace(0, 2 * np.pi))),
], new_fig_layout).show()
