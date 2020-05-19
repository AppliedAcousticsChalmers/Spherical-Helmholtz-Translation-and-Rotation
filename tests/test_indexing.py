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
