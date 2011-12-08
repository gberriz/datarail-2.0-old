# -*- coding: utf-8 -*-
import os.path as op
from collections import namedtuple

import numpy as np
import cPickle as pickle

import h5py

from multikeydict import MultiKeyDict as mkd
import icbp45_utils
import h5helper as h5h
from factor_nset import get_feasible

from pdb import set_trace as ST

from decruft import Decrufter
DECRUFTER = Decrufter(globals())

class __param(object): pass
PARAM = __param()
del __param

__d = PARAM.__dict__
__d.update(
    {
      'debug': False,
      'encoding': 'utf-8',
      'sep': (',\t,', ',', '|', '^'),
      'extra_dim': {'stat': ('mean', 'stddev')},
    })


def splitpath(pseudopath):
    comps = icbp45_utils.rsplit(pseudopath)
    path = ''
    for i, comp in enumerate(comps[:-1]):
        path = op.join(path, comp)
        if not op.exists(path):
            break
        if h5h.ishdf5(path):
            return path, comps[i:]
    raise ValueError("cannot resolve '%s'" % pseudopath)


def _parseargs(argv):
    path_to_dimspecs = argv[1]
    path_to_confounders = argv[2]
    path_to_precube = argv[3]
    output_path = argv[4]

    d = dict()
    l = locals()
    params = ('path_to_dimspecs path_to_confounders path_to_precube output_path ')
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

# from noclobberdict import NoClobberDict

# def verify_cubes(cubes, kvgen):
#     lookup0 = NoClobberDict()
#     lookup1 = NoClobberDict()
#     for cube in cubes:
#         for k0, v0 in cube.iteritemsmk():
#             k, v = kvgen(k0, v0)
#             lookup0[k] = v
#             lookup1[v] = k


# def extract_repno_lookup(keycoords, cubes):
#     i, j = [keycoords.index(f) for f in 'cell_line', 'repno']
#     # next line is optional...
#     verify_cubes(cubes, lambda k, v: ((k[i], k[j]), v.assay))

#     ijset = set((i, j))
#     perm = sorted(range(len(keycoords)), key=lambda h: h not in ijset)
#     lookup = NoClobberDict()
#     for cube in (c.permutekeys(perm) for c in cubes):
#         for (cell_line, repno), v in cube.iteritemsmk(2):
#             lookup[v.itervaluesmk().next().assay] = repno
#     return lookup


def unpickle_confounders(path_to_confounders):
    global ValCoords, PickledCubes, Cube # globalized to enable unpickling
    PickledCubes = namedtuple('PickledCubes', 'dimensions cubes')
    ValCoords = tuple # keeps pickle from whining
    Cube = mkd # keeps pickle from whining

    # unpickle for the dimensions tuple only
    with open(path_to_confounders, 'r') as fh:
        dims = pickle.load(fh).dimensions

    KeyCoords, ValCoords = [namedtuple(n, c) for n, c in
                            zip(('KeyCoords', 'ValCoords'), dims)]
    del dims

    # now a full unpickling, for keeps
    with open(path_to_confounders, 'r') as fh:
        confounders = pickle.load(fh)

    cubes = confounders.cubes
    return cubes, KeyCoords, ValCoords


def unpickle_precube(path_to_precube):
    with open(path_to_precube, 'r') as fh:
        return pickle.load(fh)


def main(argv):
    _parseargs(argv)

    confounders, KeyCoords, ValCoords = \
                 unpickle_confounders(PARAM.path_to_confounders)

    h5 = h5h.createh5h(PARAM.output_path)[0]

    assert h5.filename == PARAM.output_path

    with h5py.File(h5.filename, 'a') as h5addh:
        pickled = pickle.dumps(confounders)
        h5addh.create_dataset('metadata', data=pickled)

    # KeyCoords._fields: ('cell_line', 'ligand_name', 'ligand_concentration', 'time', 'signal', 'repno')
    # ValCoords._fields: ('assay', 'plate', 'well', 'channel', 'antibody')
    # confounders.keys(): [u'CK', u'GF']
    # confounders.iterkeysmk():
    #     CK, HCC1419, TNF-α, 1, 10, STAT3-r-647, (0,)
    #     CK, HCC1419, TNF-α, 1, 10, STAT3-r-488, (0,)
    #     ...

    # dimspecs = unpickle_dimspecs(PARAM.path_to_dimspecs)
    # print dimspecs

    # repno_lookup = extract_repno_lookup(KeyCoords._fields,
    #                                     confounders.values())

    nonfactorial = set()
    precube = unpickle_precube(PARAM.path_to_precube)
    output = mkd()
    for subassay, subassay_conf in confounders.sorteditemsmk(height=1):
        output[subassay] = output_datacube = mkd()
        dcube = precube[subassay]
        for k, v in subassay_conf.sorteditemsmk():
            output_datacube.set(k, dcube.get((v.assay,) + k[:-1]))

        keys_tuple = list(output_datacube.sortedkeysmk())
        nonfactorial.update(get_feasible(keys_tuple)[0])


    dimnames = KeyCoords._fields
    if nonfactorial:
        subperms = map(tuple, (sorted(nonfactorial),
                               [i for i in range(len(dimnames))
                                if i not in nonfactorial]))
        height = len(subperms[0])
        assert height > 1
        perm = sum(subperms, ())
        predn = [tuple([dimnames[i] for i in s]) for s in subperms]
        dimnames = ((u'__'.join(predn[0]),) + predn[1])


    for subassay, output_datacube in output.items():
        if nonfactorial:
            output_datacube = (output_datacube.permutekeys(perm).
                               collapsekeys(height))

        dimvals = output_datacube._dimvals()
        dimspec = h5h.mk_dimspec(dimnames, dimvals)
        dimspec.update(PARAM.extra_dim)
        dimlengths = map(len, dimspec.values())
        npcube = (np.vstack(output_datacube.sortedvaluesmk()).
                  reshape(dimlengths))
        h5h.add(h5, dimspec, npcube, name=subassay)

    return 0


DECRUFTER.instrument(globals())

if __name__ == '__main__':
    import sys
    DECRUFTER.exit(main(sys.argv)) # any output sent to sys.stderr by default
