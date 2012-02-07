from __future__ import division

import os
import os.path as op
import errno
import numpy as np
from numpy import arange, array
import fcntl

import yaml
import h5py

from operator import mul
from ordereddict import OrderedDict as ordd
import hdf
import hyperbrick as hb

from pdb import set_trace as ST

class PseudoLogger(object):
    import warnings
    def log(self, msg, _warnings=warnings):
        _warnings.warn(msg)

LOGGER = PseudoLogger()

class Hdf5File(h5py.File):

    # uncomment for debugging
    #
    # def __new__(cls, path, mode='r+', **kwargs):
    #     ret = super(Hdf5File, cls).__new__(cls, path, mode='r+', **kwargs)
    #     print cls, type(ret)
    #     try:
    #         print ret
    #     except Exception, e:
    #         print str(e)
    #     return ret

    def __init__(self, path, mode='r+', **kwargs):
        self._mode = mode
        self._path = path
        self._dummyhandle = None
        super(Hdf5File, self).__init__(path, mode, **kwargs)


    def __enter__(self):
        super(Hdf5File, self).__enter__()
        if self._mode != 'r':
            self._lock()
        self._register()
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        self._unregister()
        super(Hdf5File, self).__exit__()


    def __del__(self):
        self._unregister()
        sup = super(Hdf5File, self)
        if hasattr(sup, '__del__'):
            sup.__del__()


    def close(self):
        h = self._dummyhandle
        if h is not None:
            h.close()
            self._dummyhandle = None
        super(Hdf5File, self).close()


    def _lock(self):
        from fcntl import flock, LOCK_EX, LOCK_NB
        # LOGGER.log('locking %s...' % self._path)
        self._dummyhandle = open(self._path, 'r')
        try:
            flock(self._dummyhandle, LOCK_EX|LOCK_NB)
        except IOError, e:
            LOGGER.log("can't immediately write-lock the file "
                       "(%s), blocking ..." % e)
            flock(self._dummyhandle, LOCK_EX)

        
    REGISTRY = dict()


    def _register(self, registry=REGISTRY):
        registry[id(self)] = self


    def _unregister(self, registry=REGISTRY):
        # Note: LBYL-style below to avoid unnecessary output to stderr
        # (exceptions are ignored while this method executes, and
        # warnings are written to stderr instead)
        id_ = id(self)
        if id_ in registry:
            del registry[id_]
        

    @classmethod
    def shutdown(cls, registry=REGISTRY):
        for h in registry.values():
            try:
                h.close()
            except Exception, e:
                try:
                    LOGGER.log('at shutdown: %s' % str(e))
                except:
                    pass




import atexit
atexit.register(Hdf5File.shutdown)
del atexit


def mkdirp(h5path, components, **kwargs):
    with Hdf5File(h5path) as h5:
        return h5.mkdirp(components, **kwargs)


def rm(path):
    try:
        os.remove(path)
    except OSError, e:
        if e.errno != errno.ENOENT:
            raise


def ishdf5(path):
    try:
        with h5py.File(path, 'r'):
            return True
    except IOError, e:
        if str(e).lower().startswith('unable to open file'):
            return False
        raise


def createh5h(bn, ext=None, _default_ext='.h5'):
    if op.splitext(bn)[1] == _default_ext:
        path = bn
    else:
        if ext is None:
            ext = _default_ext
        path = bn + ext
    rm(path)
    # NOTE: at the moment, the call below *requires* explicitly
    # mentioning the path keyword; i.e. hdf.Hdf5(path) will cause
    # problems
    return hdf.Hdf5(path=path), path


def force_create_dataset(hdf5group, name, **kwargs):
    try:
        hdf5group.__delitem__(name)
    except KeyError:
        pass
    return hdf5group.create_dataset(name, **kwargs)


def add(h5h, dimspec, data, name=None):
    start = ordd((k, v[0]) for k, v in dimspec.items())

    dimnames = dimspec.keys()
    if name is None:
        name = '%dD_0' % len(dimnames)
    name = h5h.add_sdcube(dimnames, name=name)
    cube = h5h.get_sdcube(name)
    cube.create_dataset(dimspec)
    cube.set_data(start, data)

    
def mk_dimspec(dimnames, dimvals):
    return ordd(zip(dimnames, dimvals))


def prod(ls, init=1):
    return reduce(mul, ls, init)


def dump(item):
    def _dump(item):
        if isinstance(item, dict):
            return dict((_dump(k), _dump(v))
                        for k, v in item.items())
        elif hasattr(item, '__iter__'):
            if hasattr(item, '__len__'):
                return map(_dump, item)
        elif isinstance(item, int):
            return item
        else:
            try:
                return str(item)
            except:
                return unicode(item)

        raise TypeError('unsupported type: %s' % type(item))

    return yaml.dump(_dump(item), allow_unicode=True)


def load(item):
    def _load(item):
        if isinstance(item, dict):
            return dict((_load(k), _load(v))
                        for k, v in item.items())
        elif hasattr(item, '__iter__'):
            if hasattr(item, '__len__'):
                return tuple(map(_load, item))
        elif isinstance(item, int):
            return item
        else:
            return unicode(item)

        raise TypeError('unsupported type: %s' % type(item))

    return _load(yaml.load(item))


def write_hyperbrick(h5grp, brick):
    force_create_dataset(h5grp, 'labels', data=dump(brick.labels))
    force_create_dataset(h5grp, 'data', data=brick.data)
    return


def read_hyperbrick(h5grp):
    data = np.array(h5grp['data'])
    labels = load(h5grp['labels'].value)
    return hb.HyperBrick(data, labels)


def main(argv):
    from itertools import product
    from string import ascii_lowercase as lc, ascii_uppercase as uc

    from multikeydict import MultiKeyDict as mkd

    nd0 = 4
    nd1 = 1
    nd = nd0 + nd1
    dimnames0 = uc[:nd0]
    dimnames1 = lc[nd0:nd]
    dimnames = dimnames0 + dimnames1

    range_nd0 = range(nd0)
    #range_nd0 = (3, 5, 7, 11)
    dimlengths0 = tuple([nd0 + i for i in range_nd0])
    dimlengths1 = tuple([2] * nd1)
    dimlengths = dimlengths0 + dimlengths1

    assert len(dimnames) == len(dimlengths)
    def mk_dimvals(names, lengths):
        def fmt(p, i):
            return '%s%d' % (p, i)

        return [[fmt(c, k) for k in range(j)]
                for c, j in zip(names, lengths)]

    # def mk_dimvals(names, lengths, offset=0):
    #     def fmt(p, i):
    #         return '%s%d' % (p, i)

    #     def mk_iter(p, l, o):
    #         def _iter(p=p, l=l, o=o):
    #             b = 0
    #             while True:
    #                 for k in range(l):
    #                     yield fmt(p, k + b)
    #                 b += o
    #         return _iter()

    #     return [mk_iter(c, j, offset)
    #             for c, j in zip(names, lengths)]

    dimvals0 = mk_dimvals(dimnames0, dimlengths0)
    # dimvals0 = mk_dimvals(dimnames0, dimlengths0, 1)
    dimvals1 = mk_dimvals(dimnames1, dimlengths1)
    dimspec = mk_dimspec(dimnames, dimvals0 + dimvals1)

    data0 = range(prod(dimlengths0))
    data1 = array([(-1)**i * x for i, x in
                   enumerate(1/(2 + arange(prod(dimlengths1))))])
    data_mkd = mkd(maxdepth=nd0, noclobber=True)
    # def idx(i, l, *ps):
    #     if l == 0:
    #         return (i,)
    #     else:
    #         q, r = divmod(i, prod(ps))
    #         return (q,) + idx(r, l - 1, *ps[1:])

    # ps = list(reversed([3, 5, 7, 11]))
    # #ps = [3, 5, 7]
    # ps = [7, 3, 5]
    # print dimvals0
    # def to_rep(k, rxs, l):
    #     if l == 0:
    #         return (k,)
    #     q, r = divmod(k, rxs[-1])
    #     return to_rep(q, rxs[:-1], l - 1) + (r,)

    # valsets = [set() for _ in range(nd0)]
    for i, ks in enumerate(product(*dimvals0)):
        data_mkd.set(ks, data1 + data0[i])
        # continue

        # vs = list(to_rep(i, dimlengths0[1:], len(dimlengths0) - 1))
        # for j, u in enumerate(vs[:-1]):
        #     vs[j + 1] += u
        # ws = tuple(p + d for p, d in zip([s[0] for s in ks], map(str, vs)))
        # for s, w in zip(valsets, ws):
        #     s.add(w)
        # data_mkd.set(ws, data1 + data0[i])

    # print [tuple(sorted(vs, key=lambda w: (w[0], int(w[1:])))) for vs in data_mkd._dimvals()]
    # print [tuple(sorted(vs, key=lambda w: (w[0], int(w[1:])))) for vs in valsets]

    data = np.vstack(data_mkd.itervaluesmk()).reshape(dimlengths)

    if len(argv) > 1:
        bn = argv[1]
    else:
        bn = 'q_' + '-x-'.join(('x'.join(map(str, dimlengths0)),
                                'x'.join(map(str, dimlengths1))))
    h5h = createh5h(bn)[0]
    add(h5h, dimspec, data)
    return 0

if __name__ == '__main__':
    import sys
    exit(main(sys.argv))
