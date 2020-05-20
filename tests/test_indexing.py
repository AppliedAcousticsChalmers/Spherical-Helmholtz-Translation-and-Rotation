import faheltzmm.indexing
import pytest


@pytest.mark.parametrize('order', [0, 1, 4, 9])
def test_scheme_displays(order):
    full = faheltzmm.indexing.expansions(order, 'full')
    compact = faheltzmm.indexing.expansions(order, 'compact')
    linear = faheltzmm.indexing.expansions(order, 'linear')

    for n in range(order + 1):
        assert full[n, 0] == (n, 0), f'Full scheme display failed max order {order}. Expected {(n, 0)}, got {full[n, 0]}'
        assert compact[n, 0] == (n, 0), f'Compact scheme display failed max order {order}. Expected {(n, 0)}, got {compact[n, 0]}'
        assert linear[n**2 + n] == (n, 0), f'Linear scheme display failed max order {order}. Expected {(n, 0)}, got {linear[n**2 + n]}'
        for m in range(1, n + 1):
            assert full[n, m] == (n, m), f'Full scheme display failed max order {order}. Expected {(n, m)}, got {full[n, m]}'
            assert full[n, -m] == (n, -m), f'Full scheme display failed max order {order}. Expected {(n, -m)}, got {full[n, -m]}'
            assert compact[n, m] == (n, m), f'Compact scheme display failed max order {order}. Expected {(n, m)}, got {compact[n, m]}'
            assert compact[m - 1, n] == (n, -m), f'Compact scheme display failed max order {order}. Expected {(n, -m)}, got {compact[m - 1, n]}'
            assert linear[n**2 + n + m] == (n, m), f'Linear scheme display failed max order {order}. Expected {(n, m)}, got {linear[n**2 + n + m]}'
            assert linear[n**2 + n - m] == (n, -m), f'Linear scheme display failed max order {order}. Expected {(n, -m)}, got {linear[n**2 + n - m]}'


@pytest.mark.parametrize('new', ['full', 'compact', 'linear'])
@pytest.mark.parametrize('scheme', ['full', 'compact', 'linear'])
@pytest.mark.parametrize('order', [0, 1, 3, 8])
def test_index_conversions(order, scheme, new):
    from_scheme = faheltzmm.indexing.expansions(order, scheme)
    new_scheme = faheltzmm.indexing.expansions(order, new)
    if new == 'full' and (scheme == 'linear' or scheme == 'compact'):
        with pytest.raises(ValueError):
            conversion = faheltzmm.indexing.expansions(order, scheme, new)
        return

    conversion = faheltzmm.indexing.expansions(order, scheme, new)
    assert (from_scheme[conversion] == new_scheme).all(), f'Failed index conversion from {scheme} to {new} on max order {order}'


@pytest.mark.parametrize('new', ['full', 'compact', 'linear'])
@pytest.mark.parametrize('scheme', ['full', 'compact', 'linear'])
@pytest.mark.parametrize('order', [0, 1, 4])
def test_convetion_functions(order, scheme, new):
    from_scheme = faheltzmm.indexing.expansions(order, scheme)
    new_scheme = faheltzmm.indexing.expansions(order, new)
    from_scheme[from_scheme == None] = 0
    new_scheme[new_scheme == None] = 0

    converted = faheltzmm.indexing.expansions(from_scheme, scheme, new)
    assert (converted == new_scheme).all(), f'Failed function convertion from {scheme} to {new} at order {order}'


@pytest.mark.parametrize('scheme', ['full', 'compact', 'linear'])
@pytest.mark.parametrize('order', [0, 1, 4])
def test_zonal_indices(order, scheme):
    all_components = faheltzmm.indexing.expansions(order, scheme)
    zonal_components = all_components[faheltzmm.indexing.expansions(order, scheme, 'zonal')]
    for n in range(order + 1):
        assert zonal_components[n] == (n, 0), f'Failed zonal extraction for scheme {scheme} at order {order}, expected {(n, 0)}, got {zonal_components[n]}'


@pytest.mark.parametrize('scheme', ['full', 'compact', 'linear'])
@pytest.mark.parametrize('order', [0, 1, 4])
def test_positive_indices(order, scheme):
    all_components = faheltzmm.indexing.expansions(order, scheme)
    positive_components = all_components[faheltzmm.indexing.expansions(order, scheme, 'positive')]
    idx = 0
    for n in range(order + 1):
        for m in range(1, n + 1):
            assert positive_components[idx] == (n, m), f'Failed positive modes extraction for scheme {scheme} at order {order}, expected {(n, m)}, got {positive_components[idx]}'
            idx += 1

@pytest.mark.parametrize('scheme', ['full', 'compact', 'linear'])
@pytest.mark.parametrize('order', [0, 1, 4])
def test_negative_indices(order, scheme):
    all_components = faheltzmm.indexing.expansions(order, scheme)
    negative_components = all_components[faheltzmm.indexing.expansions(order, scheme, 'negative')]
    idx = 0
    for n in range(order + 1):
        for m in range(1, n + 1):
            assert negative_components[idx] == (n, -m), f'Failed negative modes extraction for scheme {scheme} at order {order}, expected {(n, -m)}, got {negative_components[idx]}'
            idx += 1


@pytest.mark.parametrize('scheme', ['full', 'compact', 'linear'])
@pytest.mark.parametrize('order', [0, 1, 4])
def test_nonnegative_indices(order, scheme):
    all_components = faheltzmm.indexing.expansions(order, scheme)
    nonnegative_components = all_components[faheltzmm.indexing.expansions(order, scheme, 'nonnegative')]
    idx = 0
    for n in range(order + 1):
        for m in range(n + 1):
            assert nonnegative_components[idx] == (n, m), f'Failed non-negative modes extraction for scheme {scheme} at order {order}, expected {(n, m)}, got {nonnegative_components[idx]}'
            idx += 1

@pytest.mark.parametrize('scheme', ['full', 'compact', 'linear'])
@pytest.mark.parametrize('order', [0, 1, 4])
def test_nonpositive_indices(order, scheme):
    all_components = faheltzmm.indexing.expansions(order, scheme)
    nonpositive_components = all_components[faheltzmm.indexing.expansions(order, scheme, 'nonpositive')]
    idx = 0
    for n in range(order + 1):
        for m in range(n + 1):
            assert nonpositive_components[idx] == (n, -m), f'Failed non-positive modes extraction for scheme {scheme} at order {order}, expected {(n, m)}, got {nonpositive_components[idx]}'
            idx += 1


@pytest.mark.parametrize('scheme', ['full', 'compact', 'linear'])
@pytest.mark.parametrize('order', [0, 1, 4])
def test_sectorial_indices(order, scheme):
    all_components = faheltzmm.indexing.expansions(order, scheme)
    sectorial_components = all_components[faheltzmm.indexing.expansions(order, scheme, 'sectorial')]
    idx = 0
    for n in range(order + 1):
        for m in [-n, n]:
            assert sectorial_components[idx] == (n, m), f'Failed sectorial_components modes extraction for scheme {scheme} at order {order}, expected {(n, m)}, got {sectorial_components[idx]}'
            idx += 1


@pytest.mark.parametrize('scheme', ['full', 'compact', 'linear'])
@pytest.mark.parametrize('order', [0, 1, 4])
def test_positive_sectorial_indices(order, scheme):
    all_components = faheltzmm.indexing.expansions(order, scheme)
    positive_sectorial_components = all_components[faheltzmm.indexing.expansions(order, scheme, 'positive sectorial')]
    idx = 0
    for n in range(1, order + 1):
        assert positive_sectorial_components[idx] == (n, n), f'Failed positive sectorial components modes extraction for scheme {scheme} at order {order}, expected {(n, n)}, got {positive_sectorial_components[idx]}'
        idx += 1


@pytest.mark.parametrize('scheme', ['full', 'compact', 'linear'])
@pytest.mark.parametrize('order', [0, 1, 4])
def test_negative_sectorial_indices(order, scheme):
    all_components = faheltzmm.indexing.expansions(order, scheme)
    negative_sectorial_components = all_components[faheltzmm.indexing.expansions(order, scheme, 'negative sectorial')]
    idx = 0
    for n in range(1, order + 1):
        assert negative_sectorial_components[idx] == (n, -n), f'Failed negative sectorial components modes extraction for scheme {scheme} at order {order}, expected {(n, -n)}, got {negative_sectorial_components[idx]}'
        idx += 1


@pytest.mark.parametrize('scheme', ['full', 'compact', 'linear'])
@pytest.mark.parametrize('order', [0, 1, 4])
def test_tesseral_indices(order, scheme):
    all_components = faheltzmm.indexing.expansions(order, scheme)
    tesseral_components = all_components[faheltzmm.indexing.expansions(order, scheme, 'tesseral')]
    idx = 0
    for n in range(order + 1):
        for m in range(-n + 1, n):
            if m == 0:
                continue
            assert tesseral_components[idx] == (n, m), f'Failed tesseral modes extraction for scheme {scheme} at order {order}, expected {(n, m)}, got {tesseral_components[idx]}'
            idx += 1


@pytest.mark.parametrize('scheme', ['full', 'compact', 'linear'])
@pytest.mark.parametrize('order', [0, 1, 4])
def test_positive_tesseral_indices(order, scheme):
    all_components = faheltzmm.indexing.expansions(order, scheme)
    positive_tesseral_components = all_components[faheltzmm.indexing.expansions(order, scheme, 'positive tesseral')]
    idx = 0
    for n in range(order + 1):
        for m in range(1, n):
            assert positive_tesseral_components[idx] == (n, m), f'Failed positive tesseral modes extraction for scheme {scheme} at order {order}, expected {(n, m)}, got {positive_tesseral_components[idx]}'
            idx += 1


@pytest.mark.parametrize('scheme', ['full', 'compact', 'linear'])
@pytest.mark.parametrize('order', [0, 1, 4])
def test_negative_tesseral_indices(order, scheme):
    all_components = faheltzmm.indexing.expansions(order, scheme)
    negative_tesseral_components = all_components[faheltzmm.indexing.expansions(order, scheme, 'negative tesseral')]
    idx = 0
    for n in range(order + 1):
        for m in range(1, n):
            assert negative_tesseral_components[idx] == (n, -m), f'Failed negative tesseral modes extraction for scheme {scheme} at order {order}, expected {(n, -m)}, got {negative_tesseral_components[idx]}'
            idx += 1