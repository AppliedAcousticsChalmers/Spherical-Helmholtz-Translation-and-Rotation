import faheltzmm.generate._spherical_harmonics
import numpy as np
import pytest
import scipy.special
import itertools

key_angles = list(itertools.product(
    [0, np.pi / 2, np.pi],
    [-2 * np.pi, -1.5 * np.pi, -np.pi, -np.pi / 2, 0, np.pi / 2, np.pi, 1.5 * np.pi, 2 * np.pi]
))

mesh_angles = [(np.linspace(0, np.pi, 101), np.linspace(0, 2 * np.pi, 101))]
same_dimension_angles = [
    (np.random.uniform(low=0, high=np.pi, size=7), np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=7)),
    (np.random.uniform(low=0, high=np.pi, size=(5, 7)), np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=(5, 7))),
    (np.random.uniform(low=0, high=np.pi, size=(5, 7, 11)), np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=(5, 7, 11))),
]
broadcasting_angles = [
    (np.random.uniform(low=0, high=np.pi), np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=7)),
    (np.random.uniform(low=0, high=np.pi, size=7), np.random.uniform(low=-2 * np.pi, high=2 * np.pi)),
    (np.random.uniform(low=0, high=np.pi, size=(5, 1)), np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=7)),
    (np.random.uniform(low=0, high=np.pi, size=5), np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=(7, 1))),
]


@pytest.mark.parametrize('max_order', [0, 1, 20])
@pytest.mark.parametrize('colatitude, azimuth', key_angles + mesh_angles)
def test_scipy_conformity(max_order, colatitude, azimuth):
    angles = np.broadcast(colatitude, azimuth)
    all_n = np.arange(max_order + 1).reshape([-1, 1] + [1] * angles.ndim)
    pos_m = np.arange(max_order + 1).reshape([1, -1] + [1] * angles.ndim)
    all_m = np.concatenate([pos_m, -pos_m[:, :0:-1]], axis=1)
    scipy_all_m = np.nan_to_num(scipy.special.sph_harm(all_m, all_n, azimuth, colatitude))
    implemented_all_m = faheltzmm.generate._spherical_harmonics.spherical_harmonics(max_order, colatitude, azimuth, return_negative_m=True)

    np.testing.assert_allclose(implemented_all_m, scipy_all_m, atol=1e-15)


@pytest.mark.parametrize('max_order', [0, 1, 2, 3])
@pytest.mark.parametrize('colatitude, azimuth', same_dimension_angles + broadcasting_angles)
def test_broadcasting(max_order, colatitude, azimuth):
    implemented = faheltzmm.generate._spherical_harmonics.spherical_harmonics(max_order, colatitude, azimuth, return_negative_m=False).shape
    expected = (max_order + 1, max_order + 1) + np.broadcast(colatitude, azimuth).shape
    assert implemented == expected, "Broadcasted array has unexpected shape. Got {}, expected {}".format(implemented, expected)


@pytest.mark.parametrize('max_order', [0, 1, 2, 6])
def test_show_index_scheme(max_order):

    full_idx = faheltzmm.generate._spherical_harmonics.show_index_scheme(max_order, return_negative_m='full')
    compact_idx = faheltzmm.generate._spherical_harmonics.show_index_scheme(max_order, return_negative_m='compact')
    positive_idx = faheltzmm.generate._spherical_harmonics.show_index_scheme(max_order, return_negative_m='positive')

    for n in range(max_order + 1):
        assert full_idx[n, 0] == (n, 0)
        assert compact_idx[n, 0] == (n, 0)
        assert positive_idx[n, 0] == (n, 0)
        for m in range(1, n + 1):
            assert full_idx[n, m] == (n, m)
            assert full_idx[n, -m] == (n, -m)
            assert compact_idx[n, m] == (n, m)
            assert compact_idx[m - 1, n] == (n, -m)
            assert positive_idx[n, m] == (n, m)

    for n in range(max_order + 1):
        for m in range(n + 1, max_order + 1):
            assert positive_idx[n, m] is None


@pytest.mark.parametrize('max_order', [0, 1, 2, 6])
@pytest.mark.parametrize('colatitude, azimuth', [(0.5, 0.5)] + same_dimension_angles)
def test_positive_output_format(max_order, colatitude, azimuth):
    positive = faheltzmm.generate._spherical_harmonics.spherical_harmonics(max_order, colatitude, azimuth, return_negative_m=False)
    full = faheltzmm.generate._spherical_harmonics.spherical_harmonics(max_order, colatitude, azimuth, return_negative_m=True)

    positive_to_full = faheltzmm.generate._spherical_harmonics.positive_to_full(positive)
    full_to_positive = faheltzmm.generate._spherical_harmonics.full_to_positive(full)

    np.testing.assert_allclose(full, positive_to_full)
    np.testing.assert_allclose(positive, full_to_positive)

    for n in range(max_order + 1):
        np.testing.assert_allclose(positive[n, 0], full[n, 0], err_msg="Positive output does not equal full output at (n, m) = ({}, 0)".format(n))
        for m in range(1, n + 1):
            np.testing.assert_allclose(positive[n, m], full[n, m], err_msg="Positive output does not equal full output at (n,m) = ({}, {})".format(n, m))
    np.testing.assert_allclose(positive[np.triu_indices(max_order, 1)], 0, err_msg="Positive output is not zero above diagonal")


@pytest.mark.parametrize('max_order', [0, 1, 2, 6])
@pytest.mark.parametrize('colatitude, azimuth', [(0.5, 0.5)] + same_dimension_angles)
def test_compact_output_format(max_order, colatitude, azimuth):
    full = faheltzmm.generate._spherical_harmonics.spherical_harmonics(max_order, colatitude, azimuth, return_negative_m=True)
    compact = faheltzmm.generate._spherical_harmonics.spherical_harmonics(max_order, colatitude, azimuth, return_negative_m='compact')

    full_to_compact = faheltzmm.generate._spherical_harmonics.full_to_compact(full)
    compact_to_full = faheltzmm.generate._spherical_harmonics.compact_to_full(compact)

    np.testing.assert_allclose(full, compact_to_full)
    np.testing.assert_allclose(compact, full_to_compact)

    for n in range(max_order + 1):
        np.testing.assert_allclose(compact[n, 0], full[n, 0], err_msg="Compact output does not equal full output at (n,m) = ({}, 0)".format(n))
        for m in range(1, n + 1):
            np.testing.assert_allclose(compact[n, m], full[n, m], err_msg="Compact output does not equal full output at (n,m) = ({}, {})".format(n, m))
            np.testing.assert_allclose(compact[m - 1, n], full[n, -m], err_msg="Compact output does not equal full output at (n,m) = ({}, {})".format(n, -m))
