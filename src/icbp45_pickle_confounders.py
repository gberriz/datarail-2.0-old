# -*- coding: utf-8 -*-
import os
from collections import namedtuple, defaultdict
# import pickle
import cPickle as pickle
import numpy as np

from multikeydict import MultiKeyDict as mkd
import icbp45_utils
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
      'sep': (',\t,', ',', '|', '^'),
      'extra_dim_name': 'confounder',
    })
del __d

def _parseargs(argv):
    path_to_expmap = argv[1]
    path_to_pickle = argv[2]

    d = dict()
    l = locals()
    params = ('path_to_expmap path_to_pickle')
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
        return float(s) if '.' in s else int(s)
    except ValueError:
        return s.decode(PARAM.encoding)


def parse_segment(segment, _sep=PARAM.sep[1]):
    return tuple(map(convert, segment.split(_sep)))


def parse_line(line, _sep=PARAM.sep[0]):
    return tuple(map(parse_segment, line.strip().split(_sep)))


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


def main(argv):
    _parseargs(argv)
    outpath = PARAM.path_to_pickle
    if os.path.exists(outpath):
        import sys
        print >> sys.stderr, 'warning: %s exists' % outpath

    global ValCoords, PickledCubes # globalized to enable pickling
    with open(PARAM.path_to_expmap) as fh:
        KeyCoords, ValCoords = [namedtuple(n, c)
                                for n, c in zip(('KeyCoords', 'ValCoords'),
                                                parse_line(fh.next()))]

        OutputKeyCoords = namedtuple('OutputKeyCoords',
                                     KeyCoords._fields + ('repno',))

        global Cube  # required for pickling
        class Cube(mkd):
            def __init__(self, *args, **kwargs):
                maxd = kwargs.get('maxdepth', len(OutputKeyCoords._fields))
                super(Cube, self).__init__(maxdepth=maxd, noclobber=True)

        cubes = mkd(1, Cube)

        mapper = KeyMapper(*[KeyMapper(len(c._fields))
                             for c in OutputKeyCoords, ValCoords])
        debug = PARAM.debug
        count = 0
        for line in fh:
            key, val = [clas(*tpl) for clas, tpl in
                        zip((KeyCoords, ValCoords), parse_line(line))]
            subassay = get_subassay(val)
            repno = get_repno(key, val)
            newkey, newval = mapper.getid((key + (repno,), val))
            cubes.get((subassay,)).set(newkey, np.array(newval, dtype='int32'))
            if not debug:
                continue
            count += 1
            if count >= 10:
                break

    kcmapper, vcmapper = mapper.mappers
    nonfactorial = set()
    npcubes = dict()

    for subassay, cube in cubes.items():
        keys_tuple = list(cube.sortedkeysmk())
        nonfactorial.update(get_feasible(keys_tuple)[0])

    # prekeydims, prevaldims = [m.mappers for m in kcmapper, vcmapper]

    dimnames = OutputKeyCoords._fields
    if nonfactorial:
        subperms = map(tuple, (sorted(nonfactorial),
                               [i for i in range(len(dimnames))
                                if i not in nonfactorial]))
        height = len(subperms[0])
        assert height > 1
        perm = sum(subperms, ())
        predn = [tuple([dimnames[i] for i in s]) for s in subperms]
        dimnames = ((u'__'.join(predn[0]),) + predn[1])

        ms = kcmapper.mappers
        pkm = [tuple([ms[i] for i in s]) for s in subperms]
        kcmapper = KeyMapper(*((KeyMapper(*pkm[0]),) + pkm[1]))

    for subassay, cube in cubes.items():
        if nonfactorial:
            cube = (cube.permutekeys(perm).collapsekeys(height))
            for k, v in cube.sorteditemsmk():
                print kcmapper[k], vcmapper[map(int, v)]

    #     # does the cube below differ from the one produced by using
    #     # cube.itervaluesmk()?
    #     prekeydims, prevaldims = [m.mappers for m in keymapper, valmapper]
    #     newshape = tuple(map(len, prekeydims + (prevaldims,)))
    #     npcube = np.vstack(cube.sortedvaluesmk()).reshape(newshape)
    #     npcubes[subassay] = (npcube, prekeydims, prevaldims)

    # PickledCubes = namedtuple('PickledCubes', 'dimensions cubes')
    # pc = PickledCubes((OutputKeyCoords._fields, ValCoords._fields), cubes)
    # with open(outpath, 'w') as fh:
    #     pickle.dump(pc, fh)


if __name__ == '__main__':
    import sys
    exit(main(sys.argv))
