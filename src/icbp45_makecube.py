# -*- coding: utf-8 -*-
import os.path as op
import collections as co
from glob import glob
import numpy as np

import memoized as me
import sdc_extract as sdcx
import h5helper as h5h

from pdb import set_trace as STOP

class __param(object): pass
PARAM = __param()
del __param

__d = PARAM.__dict__
__d.update(
    {
      'sep': (',\t,', ',', '|', '^'),
      'path_to_linkfarm': '/home/gfb2/IR/scans/linkfarm',
      'sdc_subdir_pat': '?.sdc',
      'hdf5_ext': '.h5',
      'sdc_basename': 'Data',
      'path_comp_attribs': 'assay plate well'.split(),
      'wanted_feature_types': 'Whole Nucleus Cyto'.split(),
      'antibody_class': co.namedtuple('Antibody', 'target species wavelength'),
      'require_nucleus_mean_to_cyto_mean_ratio': set((u'NF-κB',)),
    })

__d['wanted_templates'] = map(lambda s: '%s_w%%s (Mean)' % s,
                              PARAM.wanted_feature_types)
del __d

def _parseargs(argv):
    argv = tuple(unicode(a) for a in argv)
    path_to_h5 = argv[1]
    assay = argv[2]
    subassay = argv[3]

    d = dict()
    l = locals()
    params = 'path_to_h5 assay subassay '
    for p in params.strip().split():
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

def parse_antibody_spec(antibody_spec,
                        _cls=PARAM.antibody_class, _sep=PARAM.sep[2]):
    return _cls(*antibody_spec.split(_sep))

def get_target(subrecord):
    return parse_antibody_spec(subrecord.antibody).target

def get_data_path(subrecord,
                  _comps=PARAM.path_comp_attribs,
                  _path=PARAM.path_to_linkfarm):
    return op.join(_path,
                   *map(unicode,
                        (getattr(subrecord, a)
                         for a in _comps)))

def get_sdc_paths(data_path,
                  _basename=PARAM.sdc_basename + PARAM.hdf5_ext,
                  _pat=PARAM.sdc_subdir_pat):
    return sorted(glob(op.join(data_path, _pat, _basename)))

def get_wanted_features(channel):
    c = unicode(channel)
    return [t % c for t in PARAM.wanted_templates]

def get_rawdata(sdc_paths, wanted_features):
    d = sdcx._extract_wells_data(sdc_paths, wanted_features)
    return np.vstack([v for k, v in sorted(d.items())])

def maybe_reshape(d):
    return d.reshape((d.size, 1)) if len(d.shape) == 1 else d

def mean_and_stddev(d):
    dd = maybe_reshape(d)
    return np.hstack(zip(dd.mean(0, np.float64),
                         dd.std(0, np.float64)))

@me.memoized
def get_extractor(target):
    if target in PARAM.require_nucleus_mean_to_cyto_mean_ratio:
        def _cull_zeros(d, i):
            return d[d[:, i] > 0.]

        nidx, cidx = [PARAM.wanted_feature_types.index(f)
                      for f in 'Nucleus', 'Cyto']

        def _extract(rawdata):
            culled = _cull_zeros(rawdata, cidx)
            return culled[:, nidx]/culled[:, cidx]

    else:
        idx = PARAM.wanted_feature_types.index('Whole')
        def _extract(rawdata):
            return rawdata[:, idx]

    return _extract

def get_signal(rawdata, target):
    return get_extractor(target)(rawdata)

def _h5group(fh, path, *components):
    return fh[op.join(*((path,) + components))]

def partial_enumerate(nda, k):
    import itertools as it
    for ii in it.product(*[range(d) for d in nda.shape[:k]]):
        yield ii, nda[ii]

def project(data, index):
    def _project(slab, i=index):
        found = tuple(set(slab[..., i].ravel()))
        if len(found) == 1: return found[0]
        assert found
        raise ValueError, 'projection is not single-valued'
    return tuple(_project(slab) for slab in data)


def fetch_data(val):
    data_path = get_data_path(val)
    sdc_paths = get_sdc_paths(data_path)

    wanted_features = get_wanted_features(val.channel)
    rawdata = get_rawdata(sdc_paths, wanted_features)
    assert rawdata.size
    assert len(wanted_features) == rawdata.shape[1]

    target = get_target(val)
    signal = get_signal(rawdata, target)

    return mean_and_stddev(signal)


def run():
    subassay, assay = PARAM.subassay, PARAM.assay
    with h5h.Hdf5File(PARAM.path_to_h5, 'r') as h5:
        grp = h5['confounders']
        keymap = h5h.read_keymap(grp)
        confounders = h5h.read_hyperbrick(grp[subassay], grp)

    inverse_keymap = h5h.invert_keymap(keymap)

    assay_index = confounders.component_index(u'assay')
    assay_code = confounders.tocode('assay', assay)
    slab_index = project(confounders.data, assay_index).index(assay_code)

    slab = confounders.data[slab_index]

    buf = []
    for ii, vv in partial_enumerate(slab, -1):
        key = confounders.tokeycoords((slab_index,) + ii)
        buf.append(fetch_data(confounders.tovalcoords(vv)))

    result = np.array(buf)
    result.shape = slab.shape[:-1] + result.shape[-1:]

    # with h5h.Hdf5File(PARAM.path_to_h5, 'r+') as h5:
    #     dset = _h5group(h5, 'from_IR', subassay, 'data')
    #     dset[slab_index] = np.nan

    with h5h.Hdf5File(PARAM.path_to_h5, 'r+') as h5:
        dset = _h5group(h5, 'from_IR', subassay, 'data')
        existing = dset[slab_index]
        assert existing.shape == result.shape, '%s != %s' % (existing.shape, result.shape)
        assert np.isnan(existing.ravel()).all()
        dset[slab_index] = result

    with h5h.Hdf5File(PARAM.path_to_h5, 'r') as h5:
        dset = _h5group(h5, 'from_IR', subassay, 'data')
        assert not np.isnan(dset[slab_index].ravel()).any()

    return 0


def main(argv):
    _parseargs(argv)
    return run()


if __name__ == '__main__':
    import sys
    exit(main(sys.argv))
