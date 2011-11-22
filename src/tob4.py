from math import log, ceil
from numbers import Integral
import numpy as np

def is_power_of_2(x):
    try:
        return x > 0 and 2**int(round(log(x, 2))) == x
    except:
        return False


def _tob2(z, minz, hz):
    c = minz + hz
    return (0, minz) if z < c else (1, c)


def _tob4((x, y), minx, miny, rng):
    assert 0 <= x - minx < rng
    assert 0 <= y - miny < rng
    if rng < 2:
        return 0
    hrng = rng/2
    xi, x0 = _tob2(x, minx, hrng)
    yi, y0 = _tob2(y, miny, hrng)
    return hrng*hrng*(2*yi + xi) + _tob4((x, y), x0, y0, hrng)


def tob4(xy):
    assert len(xy) == 2 and 0 <= min(xy)
    rng = 2**int(ceil(log(max(xy) + 1, 2)))
    return _tob4(xy, 0, 0, rng)


def _fromb2(z, minz, hz):
    return minz if z == 0 else minz + hz


def _fromb4(z, minx, miny, rng):
    assert 0 <= z < rng*rng
    if rng < 2:
        return (minx, miny)
    hrng = rng/2
    i, j = divmod(z, hrng*hrng)
    return _fromb4(j, _fromb2(i & 1, minx, hrng),
                   _fromb2(i & 2, miny, hrng), hrng)


def fromb4(z):
    rng = 2**int(ceil(log(z + 1, 4)))
    return _fromb4(z, 0, 0, rng)


def _toqt(ones, rng):
    lo = len(ones)
    if lo == 0:
        return 0
    if lo == rng:
        return 1
    qrng = rng/4
    ss = [set() for _ in 0, 1, 2, 3]
    for o in ones:
        i, j = divmod(o, qrng)
        ss[i].add(j)
    return tuple([_toqt(o, qrng) for o in ss])
    

def toqt(ones):
    assert 0 <= min(ones)
    rng = 4**int(ceil(log(max(ones) + 1, 4)))
    return _toqt(ones, rng)


def _fromqt(qt, rng, i):
    if qt == 0:
        return set()

    if qt == 1:
        offset = i * rng
        return set([offset + j for j in range(rng)])

    assert hasattr(qt, '__iter__') and len(qt) == 4
    hrng = rng/4
    base = 4 * i
    ret = set()
    lret = 0
    for j, sqt in enumerate(qt):
        s = _fromqt(sqt, hrng, base + j)
        t = ret.union(s)
        lt = len(t)
        # assertion: all additions to ret are disjoint
        assert len(s) + lret == lt
        ret = t
        lret = lt
    return ret
            

def fromqt(qt, rng):
    return _fromqt(qt, rng, 0)


def serializeqt(qt):
    if qt == 1 or qt == 0:
        return np.array([qt, qt])

    assert hasattr(qt, '__iter__') and len(qt) == 4

    return np.concatenate([np.array([1, 0])] + map(serializeqt, qt) +
                          [np.array([0, 1])])


def packqt(qt):
    return np.packbits(serializeqt(qt))
    

if __name__ == '__main__':
    import pickle
    import re

    # ones = list()
    # for line in open('PRIV/data/HCC1806_000_0_cellxy.txt'):
    #     if re.search(r'^#', line):
    #         continue
    #     xy = tuple(map(int, line.strip('\n').split(',')))
    #     ones.append(tob4(xy))

    # ones = set(ones)

    # out0 = open('PRIV/data/HCC1806_000_0_cell4b.pkl', 'w')
    # pickle.dump(ones, out0)
    # out0.close()

    # out1 = open('PRIV/data/HCC1806_000_0_cellqt.pkl', 'w')
    # qt = toqt(ones)
    # pickle.dump(qt, out1)
    # out1.close()

    # print "done building qt"

    # ones = pickle.load(open('PRIV/data/HCC1806_000_0_cell4b.pkl'))
    # print len(ones)
    qt = pickle.load(open('PRIV/data/HCC1806_000_0_cellqt.pkl'))
    # newones = fromqt(qt, 2**20)
    # print len(ones.intersection(newones))

    pqt = packqt(qt)
    out = open('PRIV/data/HCC1806_000_0_cellqt.int', 'w')
    print >>out, tuple(pqt)

    sqt = tuple(serializeqt(qt))
    out = open('PRIV/data/HCC1806_000_0_cellqt.sqt', 'w')
    print >>out, sqt
