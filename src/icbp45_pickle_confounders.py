# -*- coding: utf-8 -*-
import os
from collections import namedtuple, defaultdict
# import pickle as serializer
# import cPickle as serializer
import numpy as np
from itertools import count

from multikeydict import MultiKeyDict as mkd
from ordereddict import OrderedDict as orddict
from orderedset import OrderedSet as ordset

import icbp45_utils
from h5helper import dump, load
import h5helper as h5h
#from factor_nset import min_shape
from factor_nset import get_labels
from factor_nset import get_factors
from factor_nset import get_feasible
from keymapper import KeyMapper

from pdb import set_trace as ST

class __param(object): pass
PARAM = __param()
del __param

__d = PARAM.__dict__
__d.update(
    {
      'debug': False,
      'encoding': 'utf-8',
      'sep': (u',\t,', u',', u'|', u'^'),
      'extra_dim_name': u'confounder',
      'maskval': 0,
      'inttype': 'int16',
    })
del __d

def _parseargs(argv):
    path_to_expmap = argv[1]
    path_to_outfile = argv[2]

    d = dict()
    l = locals()
    params = ('path_to_expmap path_to_outfile')
    for p in params.split():
        d[p] = l[p]
    _updateparams(d)


def _setparams(d):
    global PARAM
    try:
        pd = PARAM.__dict__
        pd.clear()
        pd.update(d)
    except NameError:
        class _param(object): pass
        PARAM = _param()
        _setparams(d)


def _updateparams(d):
    global PARAM
    try:
        PARAM.__dict__.update(d)
    except NameError:
        _setparams(d)


def convert(s):
    try:
        return float(s) if u'.' in s else int(s)
    except ValueError:
        return s


def parse_segment(segment, _sep=PARAM.sep[1]):
    return tuple(map(convert, segment.split(_sep)))


def parse_line(line, _sep=PARAM.sep[0], _enc=PARAM.encoding):
    return tuple(map(parse_segment, line.decode(_enc).strip().split(_sep)))


def get_subassay(subrecord):
    return icbp45_utils.get_subassay(subrecord.plate)


class IdSeq(mkd):
    class __ctr(object):
        def __init__(self, start=0):
            self._next = start


        def __call__(self):
            ret = self._next
            self._next += 1
            return ret


    def __init__(self, _factory_factory=__ctr):
        mkd.__init__(self, maxdepth=1, leaffactory=_factory_factory())
        

def get_repno(key, val, _lookup=mkd(1, IdSeq)):
    # NOTE: returns a singleton tuple (in the future, this repno
    # parameter may be a k-tuple for some k > 1)
    return (_lookup.get((key.cell_line, val.assay)),)


def needed_bits(n, _pow2_8=2**8, _pow2_16=2**16, _pow2_32=2**32):
    # n < _pow2_16 ?
    #     n < _pow2_8       ? 8  : 16
    #   : n < _pow2_32 ? 32 : 64

    return (n < _pow2_16) and \
               (n < _pow2_8 and 8 or 16) or \
               (n < _pow2_32 and 32 or 64)

    # return ((8, 16)[n < _pow2_8],
    #         (32, 64)[n < _pow2_32])[n < _pow2_16]

    # return (8 if n < _pow2_8 else 16) \
    #        if n < _pow2_16 else 32 if n < _pow2_32 else 64


def main(argv):
    _parseargs(argv)
    outpath = PARAM.path_to_outfile
    if os.path.exists(outpath):
        import sys
        print >> sys.stderr, 'warning: %s exists' % outpath

    global ValCoords, PickledCubes # globalized to enable pickling
    with open(PARAM.path_to_expmap) as fh:
        KeyCoords, ValCoords = [namedtuple(n, c)
                                for n, c in zip((u'KeyCoords', u'ValCoords'),
                                                parse_line(fh.next()))]

        OutputKeyCoords = namedtuple(u'OutputKeyCoords',
                                     KeyCoords._fields + (u'repno',))

        global Cube  # required for pickling
        class Cube(mkd):
            def __init__(self, *args, **kwargs):
                maxd = kwargs.get('maxdepth', len(OutputKeyCoords._fields))
                super(Cube, self).__init__(maxdepth=maxd, noclobber=True)
        cubes = mkd(1, Cube)

        nvals = len(ValCoords._fields)
        start = PARAM.maskval + 1
        vcmapper = KeyMapper(*([count(start)] * nvals)) # Sic! We want a
                                                        # single counter shared
                                                        # by all the component
                                                        # keymappers
        del nvals
        maxid = start
        del start

        debug = PARAM.debug
        recordcount = 0
        for line in fh:
            key, val = [clas(*tpl) for clas, tpl in
                        zip((KeyCoords, ValCoords), parse_line(line))]
            subassay = get_subassay(val)
            repno = get_repno(key, val)
            newkey = tuple(map(unicode, key + (repno,)))
            newval = vcmapper.getid(val)
            maxid = max(maxid, *newval)
            cubes.get((subassay,)).set(newkey, newval)
            if not debug:
                continue
            recordcount += 1
            if recordcount >= 10:
                break

    dtype = 'uint%d' % needed_bits(maxid)
    del maxid

    kcoords = tuple(map(unicode, OutputKeyCoords._fields))
    vcoords = tuple(map(unicode, ValCoords._fields))

    nonfactorial = set()

    for subassay, cube in cubes.items():
        keys_tuple = list(cube.sortedkeysmk())
        nonfactorial.update(get_feasible(keys_tuple)[0])

    if nonfactorial:
        subperms = map(tuple, (sorted(nonfactorial),
                               [i for i in range(len(kcoords))
                                if i not in nonfactorial]))
        del nonfactorial
        height = len(subperms[0])
        assert height > 1
        perm = sum(subperms, ())

        predn = [tuple([kcoords[i] for i in s]) for s in subperms]
        kcoords = (predn[0],) + predn[1]
        del predn
        for subassay, cube in cubes.items():
            cubes[subassay] = cube.permutekeys(perm).collapsekeys(height)
        del perm, height

    bricks = dict()
    for subassay, cube in cubes.items():
        keys_tuple = list(cube.sortedkeysmk())
        labels = get_labels(kcoords, keys_tuple) + \
                 ((PARAM.extra_dim_name, vcoords),)

        factors = tuple(kv[1] for kv in labels)
        shape = tuple(map(len, factors))
        npcube = np.ones(shape=shape, dtype=dtype) * PARAM.maskval
        for key in keys_tuple:
            npcube[cube.index(key)] = cube.get(key)

        bricks[subassay] = (labels, npcube)

    with h5h.Hdf5File(outpath, 'w') as h5:
        dir0 = h5.require_group('confounders')

        keymap = vcmapper.mappers
        h5h.force_create_dataset(dir0, 'keymap', data=dump(keymap))
        # reconstitute the above with:
        #     keymap = yaml.load(<H5>['confounders/keymap'].value)
        # ...where <H5> stands for some h5py.File instance

        for subassay, brick in bricks.items():
            subassay_dir = dir0.require_group(subassay)
            labels, data = brick

            subassay_dir.create_dataset('labels', data=dump(labels))
            # reconstitute the above with:
            #     labels = yaml.load(<H5>['confounders/<SUBASSAY>/labels'].value)
            # ...where <H5> stands for some h5py.File instance

            subassay_dir.create_dataset('data', data=data)

    # with h5h.Hdf5File(outpath, 'r') as h5:
    #     for subassay in bricks:
    #         labelsyaml = h5['confounders/%s/labels' % subassay].value
    #         factors = load(labelsyaml)
    #         for k, vs in factors:
    #             print k
    #             for v in vs:
    #                 print '  %s' % (v,)
    #             print
    #         print '\n%s\n\n' % labelsyaml

    return 0


if __name__ == '__main__':
    import sys
    exit(main(sys.argv))
