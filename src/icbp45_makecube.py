# -*- coding: utf-8 -*-
import os.path as op
from collections import namedtuple
from glob import glob
import numpy as np
import re

from memoized import memoized
from sdc_extract import _extract_wells_data
from multikeydict import MultiKeyDict as mkd
import icbp45_utils

from pdb import set_trace as ST

class __param(object): pass
PARAM = __param()
del __param

__d = PARAM.__dict__
__d.update(
    {
      'encoding': 'utf-8',

      'sep': (',\t,', ',', '|', '^'),

      'path_to_linkfarm': '/home/gfb2/IR/scans/linkfarm',
      'sdc_subdir_pat': '?.sdc',
      'hdf5_ext': '.h5',
      'sdc_basename': 'Data',
      'path_comp_attribs': 'assay plate well'.split(),
      'wanted_feature_types': 'Whole Nucleus Cyto'.split(),
      'data_coords': namedtuple('DataCoords', 'mean stddev'),
      'antibody_class': namedtuple('Antibody', 'target species wavelength'),
      'require_nucleus_mean_to_cyto_mean_ratio': set((u'NF-κB',)),
      'extra_dim': {'stat': ('mean', 'stddev')},
    })


def quote_re_pattern(pat, _qpat=re.compile(r'([.^$+*?[\](){}])')):
    return _qpat.sub(r'\\\1', pat)

__d['wanted_templates'] = map(lambda s: '%s_w%%s (Mean)' % s,
                              PARAM.wanted_feature_types)
__d['strip_ext_re'] = re.compile(quote_re_pattern(PARAM.hdf5_ext) + '$')
del __d


def _parseargs(argv):
    path_to_expmap = argv[1]
    assay = argv[2]
    subassay = argv[3]
    output_basename = PARAM.strip_ext_re.sub('', argv[4])

    d = dict()
    l = locals()
    params = ('path_to_expmap assay subassay '
              'output_basename ')
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
    d = _extract_wells_data(sdc_paths, wanted_features)
    return np.vstack([v for k, v in sorted(d.items())])


def maybe_reshape(d):
    return d.reshape((d.size, 1)) if len(d.shape) == 1 else d


def mean_and_stddev(d, _cls=PARAM.data_coords):
    dd = maybe_reshape(d)
    return _cls(*np.hstack(zip(dd.mean(0, np.float64),
                               dd.std(0, np.float64))))


@memoized
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


def get_subassay(subrecord):
    return icbp45_utils.get_subassay(subrecord.plate)


def print_record(segments, _sep0=PARAM.sep[0], _sep1=PARAM.sep[1],
                 _enc=PARAM.encoding):
    print _sep0.join([_sep1.join(map(output_form, seg))
                      for seg in segments]).encode(_enc)

def main(argv):
    _parseargs(argv)

    path = PARAM.path_to_expmap

    def skip(key, val, assay=PARAM.assay, subassay=PARAM.subassay):
        return not (val.assay == assay and get_subassay(val) == subassay)

    with open(path) as fh:
        KeyCoords, ValCoords = [namedtuple(n, c)
                                for n, c in zip(('KeyCoords', 'ValCoords'),
                                                parse_line(fh.next()))]
        assert 'field' not in ValCoords._fields

        cube = mkd(len(KeyCoords._fields), noclobber=True)
        for line in fh:
            key, val = [clas(*tpl) for clas, tpl in
                        zip((KeyCoords, ValCoords),
                            parse_line(line))]

            if skip(key, val):
                continue

            data_path = get_data_path(val)
            sdc_paths = get_sdc_paths(data_path)

            wanted_features = get_wanted_features(val.channel)
            rawdata = get_rawdata(sdc_paths, wanted_features)
            assert rawdata.size
            assert len(wanted_features) == rawdata.shape[1]

            target = get_target(val)
            signal = get_signal(rawdata, target)
            data = mean_and_stddev(signal)
            cube.set(key, data)


    assert cube, 'empty cube'

    import h5helper as H5H
    dimvals = cube._dimvals()
    dimnames = KeyCoords._fields
    dimspec = H5H.mk_dimspec(dimnames, dimvals)
    dimspec.update(PARAM.extra_dim)
    dimlengths = map(len, dimspec.values())

    h5 = H5H.createh5h(PARAM.output_basename)[0]
    H5H.add(h5, dimspec, np.vstack(cube.itervaluesmk()).reshape(dimlengths))
    return 0


if __name__ == '__main__':
    import sys
    exit(main(sys.argv))
