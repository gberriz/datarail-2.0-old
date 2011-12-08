import numpy as np

def index2coords(i, shape, order='C'):
    """
>>> array345 = np.array([[[float((i*4 + j)*5 + k)
...     for k in range(5)] for j in range(4)] for i in range(3)])
>>> all([array345[index2coords(i, (3, 4, 5))] == i
...     for i in range(3 * 4 * 5)])
True
>>> all([array345[index2coords(i, (3, 4, 5), order='F')] == i
...     for i in range(3 * 4 * 5)])
False
>>> all([index2coords(coords2index(c, (3, 4, 5)), (3, 4, 5)) == c
...      for c in product(range(3), range(4), range(5))])
True
>>> all([index2coords(coords2index(c, (3, 4, 5), order='F'),
...                   (3, 4, 5), order='F') == c
...      for c in product(range(3), range(4), range(5))])
True
    """
    # the value of order keyword determines whether the last or first
    # index changes the fastest (order == 'C': last; order = 'F': first)
    if order not in ('C', 'F'):
        raise ValueError("order must be either 'C' or 'F'")

    l = len(shape)
    if not (hasattr(shape, '__iter__') and l > 0
            and all(ii > 0 for ii in shape)):
        raise TypeError('"shape" parameter must be a non-empty sequence '
                        'of positive integers')

    n = np.product(shape)
    if not -n <= i < n:
        raise ValueError('index %d exceeds size of shape %s array' %
                         (i, shape))

    i_ = n + i if i < 0 else i
    return (_i2c(i_, shape, l, n) if order is 'C' else
            tuple(reversed(_i2c(i_, tuple(reversed(shape)), l, n))))


def _i2c(i, s, l, n):
    # computes the coordinates for index according to the row-major
    # storage convention (i.e. order == 'C')
    assert l > 0
    s0 = s[0]
    if l == 1:
        assert i < n == s0
        return (i,)
    nn, ss = n//s0, s[1:]
    c, ii = divmod(i, nn)
    return (c,) + _i2c(ii, ss, l - 1, nn)


def coords2index(coords, shape, order='C'):
    # the value of order keyword determines whether the last or first
    # index changes the fastest (order == 'C': last; order = 'F': first)
    if order not in ('C', 'F'):
        raise ValueError("order must be either 'C' or 'F'")

    l = len(shape)
    if not (hasattr(shape, '__iter__') and l > 0
            and all(i > 0 for i in shape)):
        raise TypeError('"shape" parameter must be a non-empty sequence '
                        'of positive integers')

    if not (hasattr(coords, '__iter__') and len(coords) == l):
        raise TypeError('"coords" parameter must be a non-empty sequence '
                        'of integers of length %d' % l)

    negshape = tuple(-i for i in shape)
    if not negshape <= coords < shape:
        raise ValueError('coords %s exceed size of shape %s array' %
                         (coords, shape))

    coords_ = tuple(n + i if i < 0 else i for i, n in zip(coords, shape))
    return (_c2i(coords_, shape, l) if order is 'F'
            else _c2i(tuple(reversed(coords_)), tuple(reversed(shape)), l))


def _c2i(c, s, l):
    # computes the index for coordinates according to the column-major
    # storage convention (i.e. order == 'F')
    if l == 1:
        return c[0]
    else:
        return c[0] + s[0]*(_c2i(c[1:], s[1:], l - 1))


def _make_test_array(*shape):
    return reduce(lambda I, J: np.array([I + j*I.size for j in range(J)]),
                  reversed(shape), np.array(0))

if __name__ == '__main__':
    from itertools import product

    shp = 3, 4, 5
    array3d = _make_test_array(*shp)
    array1d = array3d.reshape((array3d.size,))
    irng = tuple(range(np.product(shp)))
    crng = tuple(product(*map(range, shp)))
    i2c = index2coords
    c2i = coords2index

    assert     all([array3d[i2c(i, shp)]      == i for i in irng])
    assert not all([array3d[i2c(i, shp, 'F')] == i for i in irng])
    assert     all([array1d[c2i(c, shp)]      == c2i(c, shp) for c in crng])
    assert not all([array1d[c2i(c, shp, 'F')] == c2i(c, shp) for c in crng])

    assert     all([i2c(c2i(c, shp), shp)           == c for c in crng])
    assert     all([i2c(c2i(c, shp, 'F'), shp, 'F') == c for c in crng])
    assert not all([i2c(c2i(c, shp, 'F'), shp, 'C') == c for c in crng])
    assert not all([i2c(c2i(c, shp, 'C'), shp, 'F') == c for c in crng])

    assert     all([c2i(i2c(i, shp), shp)           == i for i in irng])
    assert     all([c2i(i2c(i, shp, 'F'), shp, 'F') == i for i in irng])
    assert not all([c2i(i2c(i, shp, 'F'), shp, 'C') == i for i in irng])
    assert not all([c2i(i2c(i, shp, 'C'), shp, 'F') == i for i in irng])

    print 'ok'
