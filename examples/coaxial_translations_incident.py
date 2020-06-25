# %%
import numpy as np
import faheltzmm.translations
import plotly.graph_objects as go

fig_layout = dict(
    width=800, height=800
)

# %% Spatial points, original coefficients, etc.
max_order = 30
wavenumber = 2 * np.pi
wavelength = 2 * np.pi / wavenumber
translation_distance = 2.2329 * wavelength

resolution = wavelength / 20
x_range = (-5 * wavelength, 5 * wavelength)
z_range = (-5 * wavelength, 5 * wavelength)

old_coefficients = np.zeros((max_order + 1, 2 * max_order + 1), dtype=complex)
old_coefficients += faheltzmm.generate.spherical_harmonics_all(max_order, colatitude=np.deg2rad(15), azimuth=np.deg2rad(0)).conj() * (1j) ** (np.arange(max_order + 1).reshape([-1, 1]))
old_coefficients -= 0.3 * faheltzmm.generate.spherical_harmonics_all(max_order, colatitude=np.deg2rad(100), azimuth=np.deg2rad(30)).conj() * (1j) ** (np.arange(max_order + 1).reshape([-1, 1]))

reexpansion_coefficients = faheltzmm.translations.coaxial_translation_coefficients(max_order, max_order, translation_distance, wavenumber, 'interior', 'interior')
new_coefficients = np.einsum('mnp, nm -> pm', reexpansion_coefficients, old_coefficients)

nx = int((max(x_range) - min(x_range)) / resolution)
nz = int((max(z_range) - min(z_range)) / resolution)
nx += (nx + 1) % 2
nz += (nz + 1) % 2
x = np.linspace(min(x_range), max(x_range), nx)
z = np.linspace(min(z_range), max(z_range), nz)
y = 0.23849234 * wavelength
ξ, η, ζ = x, y, z + translation_distance

x_mesh, y_mesh, z_mesh = old_mesh = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=0).squeeze()
ξ_mesh, η_mesh, ζ_mesh = new_mesh = np.stack(np.meshgrid(ξ, η, ζ, indexing='ij'), axis=0).squeeze()
old_bases = faheltzmm.generate.spherical_base_all(max_order=max_order, position=old_mesh, wavenumber=wavenumber, domain='interior')
new_bases = faheltzmm.generate.spherical_base_all(max_order=max_order, position=new_mesh, wavenumber=wavenumber, domain='interior')

old_coefficients_old_bases = np.einsum('nm, nm...', old_coefficients, old_bases)
new_coefficients_new_bases = np.einsum('nm, nm...', new_coefficients, new_bases)
new_coefficients_old_bases = np.einsum('nm, nm...', new_coefficients, old_bases)
old_coefficients_new_bases = np.einsum('nm, nm...', old_coefficients, new_bases)

np.allclose(old_coefficients_old_bases, new_coefficients_new_bases)

# %% Visualize the original coefficients in the original coordinates
go.Figure([
    go.Heatmap(x=x, y=z, z=np.real(old_coefficients_old_bases).T, showscale=False, colorscale='viridis', zsmooth='best'),
], fig_layout).show()

# %% Visualize the translated coefficients in the translated coordinates
go.Figure([
    go.Heatmap(x=ξ, y=ζ, z=np.real(new_coefficients_new_bases).T, showscale=False, colorscale='viridis', zsmooth='best'),
], fig_layout).show()

# %% Visualzie the difference
go.Figure([
    go.Heatmap(x=ξ, y=ζ, z=np.real(new_coefficients_new_bases - old_coefficients_old_bases).T / np.abs(old_coefficients_old_bases), showscale=False, colorscale='delta', zmin=-1.2, zmax=1.2, zsmooth='best'),
], fig_layout).show()

# %% Visualize the translated coefficients in the old coordinates
go.Figure([
    go.Heatmap(x=x, y=z, z=np.real(new_coefficients_old_bases).T, showscale=False, colorscale='viridis', zsmooth='best'),
], fig_layout).show()

# %% Visualize the original coefficients in the tralslated coordinates
go.Figure([
    go.Heatmap(x=ξ, y=ζ, z=np.real(old_coefficients_new_bases).T, showscale=False, colorscale='viridis', zsmooth='best'),
], fig_layout).show()
