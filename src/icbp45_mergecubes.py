# -*- coding: utf-8 -*-
import numpy as np
import cPickle as pickle
from itertools import product

from ordereddict import OrderedDict as orddict
from h5helper import dump, load
import h5helper as h5h
from factor_nset import get_feasible

from pdb import set_trace as ST

class __param(object): pass
PARAM = __param()
del __param

__d = PARAM.__dict__
__d.update(
    {
      'debug': False,
      'extra_dim': ('stat', ('mean', 'stddev')),
    })


def _parseargs(argv):
    path_to_precube = argv[1]
    path_to_h5 = argv[2]

    d = dict()
    l = locals()
    params = ('path_to_precube path_to_h5')
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


def unpickle_precube(path_to_precube):
    with open(path_to_precube, 'r') as fh:
        return pickle.load(fh)


def main(argv):
    _parseargs(argv)

    precube = unpickle_precube(PARAM.path_to_precube)

    output = dict()

    # └── from_IR
    #     │
    #     ├── GF
    #     │   ├── data
    #     │   └── labels
    #     │
    #     └── CK
    #         ├── data
    #         └── labels

    with h5h.Hdf5File(PARAM.path_to_confounders, 'r+') as h5:

        keymap = [dict((v, k) for k, v in d.items())
                  for d in load(h5['confounders/keymap'].value)]

        from_IR = h5.require_group('from_IR')

        for subassay, grp in h5['confounders'].items():
            if subassay not in precube: continue

            subassay_dir = from_IR.require_group(subassay)

            dcube = precube[subassay]
            confounders = grp['data']

            labels = orddict(load(grp['labels'].value))

            fnames, factors = labels.keys(), labels.values()

            confounder_index = fnames.index('confounder')

            dlabels = list(labels.items())
            dlabels[confounder_index] = PARAM.extra_dim

            h5h.force_create_dataset(subassay_dir, 'labels', data=dump(dlabels))

            shape = map(len, [kv[1] for kv in dlabels])
            output[subassay] = output_datacube = np.zeros(shape=shape)
            del shape

            assay_index = factors[confounder_index].index('assay')
            del confounder_index
            assay_dict = keymap[assay_index]

            for ii in product(*map(range, confounders.shape[:-1])):
                assay = assay_dict[confounders[ii][assay_index]]
                kk = tuple([f[i] for f, i in zip(labels.values(), ii)])
                output_datacube[ii] = dcube.get((assay, kk[0][0]) + kk[1:])

            h5h.force_create_dataset(subassay_dir, 'data', data=output_datacube)

    return 0


if __name__ == '__main__':
    import sys
    exit(main(sys.argv))
