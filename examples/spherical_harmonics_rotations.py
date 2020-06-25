# %%
import numpy as np
import faheltzmm.rotations
import plotly.graph_objects as go


def show_axes(x, y, z, **kwargs):
    return go.Scatter3d(
        x=[0, x[0], 0, y[0], 0, z[0]],
        y=[0, x[1], 0, y[1], 0, z[1]],
        z=[0, x[2], 0, y[2], 0, z[2]],
        hovertext=['', 'x', '', 'y', '', 'z'],
        **kwargs
    )


def show_point(x, y, z, **kwargs):
    return go.Scatter3d(x=[0, x], y=[0, y], z=[0, z], **kwargs)


fig_layout = {
    'scene': {
        'xaxis': {'range': (-1.1, 1.1)},
        'yaxis': {'range': (-1.1, 1.1)},
        'zaxis': {'range': (-1.1, 1.1)},
        'aspectratio': {'x': 1, 'y': 1, 'z': 1},
        'camera': {'eye': {'x': 1, 'y': 1, 'z': 1}},
    }
}


# %% Definitions of the two coordinate bases
beta = np.random.uniform(0, np.pi) * 0.3
alpha = np.random.uniform(-np.pi, np.pi) * 0.1
mu = np.random.uniform(-np.pi, np.pi) * 0.1
beta, alpha, mu = np.deg2rad([40, 10, -15])
max_order = 12

old_coefficients = faheltzmm.generate.spherical_harmonics_all(max_order, colatitude=np.deg2rad(30), azimuth=np.deg2rad(60)).conj()
old_coefficients -= faheltzmm.generate.spherical_harmonics_all(max_order, colatitude=np.deg2rad(100), azimuth=np.deg2rad(30)).conj()

Q = faheltzmm.coordinates.rotation_matrix(beta, alpha, mu)
T = faheltzmm.rotations.rotation_coefficients(max_order=max_order, colatitude=beta, primary_azimuth=alpha, secondary_azimuth=mu)
new_coefficients = np.einsum('mnp, nm -> np', T, old_coefficients)

# The coordinates of the old base in the old base
old_x = np.array([1, 0, 0])
old_y = np.array([0, 1, 0])
old_z = np.array([0, 0, 1])
# The coordinates of the old base in the new base
new_x = Q @ old_x
new_y = Q @ old_y
new_z = Q @ old_z
# The coordinates of the new base in the new new base
new_xi = np.array([1, 0, 0])
new_eta = np.array([0, 1, 0])
new_zeta = np.array([0, 0, 1])
# The coordinates of the new base in the old base
old_xi = Q.T @ new_xi
old_eta = Q.T @ new_eta
old_zeta = Q.T @ new_zeta

old_xyz = show_axes(old_x, old_y, old_z, name='Old axes', line=dict(color='blue'))
old_xietazeta = show_axes(old_xi, old_eta, old_zeta, name='New axes', line=dict(color='red'))
new_xyz = show_axes(new_x, new_y, new_z, name='Old axes', line=dict(color='blue'))
new_xietazeta = show_axes(new_xi, new_eta, new_zeta, name='New axes', line=dict(color='red'))

res = 5
old_theta = np.linspace(0, np.pi, 180 // res + 1)
old_phi = np.linspace(0, 2 * np.pi, 260 // res + 1)
old_theta_mesh, old_phi_mesh = np.meshgrid(old_theta, old_phi)
old_x_mesh, old_y_mesh, old_z_mesh = old_mesh = faheltzmm.coordinates.spherical_2_cartesian(0.8, old_theta_mesh, old_phi_mesh)
new_x_mesh, new_y_mesh, new_z_mesh = new_mesh = np.einsum('ij, j...->i...', Q, old_mesh)
_, new_theta_mesh, new_phi_mesh = faheltzmm.coordinates.cartesian_2_spherical(new_mesh)


old_harmonics = faheltzmm.generate.spherical_harmonics_all(max_order, colatitude=old_theta_mesh, azimuth=old_phi_mesh)
new_harmonics = faheltzmm.generate.spherical_harmonics_all(max_order=max_order, colatitude=new_theta_mesh, azimuth=new_phi_mesh)

old_coefficients_old_harmonics = np.einsum('nm, nm...', old_coefficients, old_harmonics)
new_coefficients_new_harmonics = np.einsum('nm, nm...', new_coefficients, new_harmonics)
new_coefficients_old_harmonics = np.einsum('nm, nm...', new_coefficients, old_harmonics)
old_coefficients_new_harmonics = np.einsum('nm, nm...', old_coefficients, new_harmonics)
np.allclose(old_coefficients_old_harmonics, new_coefficients_new_harmonics)
# %% Visualize the original coefficients in the original base
go.Figure([
    go.Surface(x=old_x_mesh, y=old_y_mesh, z=old_z_mesh, surfacecolor=old_coefficients_old_harmonics.real, showscale=False),
    old_xyz, old_xietazeta
], fig_layout).show()

# %% Visualize the rotated coefficients in the new base
go.Figure([
    go.Surface(x=new_x_mesh, y=new_y_mesh, z=new_z_mesh, surfacecolor=new_coefficients_new_harmonics.real, showscale=False),
    new_xyz, new_xietazeta
], fig_layout).show()

# %% Visualize the rotated coefficients in the old base
go.Figure([
    go.Surface(x=old_x_mesh, y=old_y_mesh, z=old_z_mesh, surfacecolor=new_coefficients_old_harmonics.real, showscale=False),
    old_xyz, old_xietazeta
], fig_layout).show()

# %% Visualize the original coefficients in the new base
go.Figure([
    go.Surface(x=new_x_mesh, y=new_y_mesh, z=new_z_mesh, surfacecolor=old_coefficients_new_harmonics.real, showscale=False),
    new_xyz, new_xietazeta
], fig_layout).show()
