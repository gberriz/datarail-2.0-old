# -*- coding: utf-8 -*-
import sys

import os.path as op
import random
import tempfile
import numpy as np
import csv
import re

import sdc_extract
from shell_utils import mkdirp
from find import find
from icbp45_utils import scrape_coords

from pdb import set_trace as ST

def _scrape_ncrmode(path, antibody, basename='.METADATA.csv'):
    with open(op.join(path, basename), 'r') as metadata:
        lines = metadata.read().splitlines()
        for i, line in enumerate(lines):
            m = re.match(r'#\s*column\s*(\d+)\s*:\s*%s' % antibody, line)
            if not m:
                continue
            colnum = int(m.group(1)) - 1
            for line in lines[i+1:]:
                if line.startswith('#'):
                    continue
                return line.split(',')[colnum].startswith('NF-κB')
        else:
            raise ValueError("can't find column for %s" % antibody)
     

def _parseargs(argv):
    nargs = len(argv)
    assert 2 < nargs < 5
    path = argv[1]

    assay, plate, well, _ = scrape_coords(path)
    extent = 'plate' if well is None else 'well'
    wavelength = argv[2]
    antibody = '%s_antibody' % wavelength

    if extent == 'plate':
        ncrmode = bool(int(argv[3])) if nargs > 3 else False
    else:
        ncrmode = _scrape_ncrmode(path, antibody)
        
    if ncrmode:
        features = ('Nucleus_w%(wavelength)s (Mean),'
                    'Cyto_w%(wavelength)s (Mean)' %
                    locals()).split(',')
        readout = 'ncratio'
        abbrev = 'ncr'
    else:
        features = ['Whole_w%s (Mean)' % wavelength]
        readout = 'wholecell'
        abbrev = 'whc'

    statpat = '%s (%%s)' % abbrev
    coord_headers = 'assay plate well field antibody'.split()

    d = dict()
    l = locals()
    params = ('path ncrmode features readout abbrev '
              'statpat wavelength antibody coord_headers '
              'assay plate well extent')

    for p in params.split():
        d[p] = l[p]
    _setparams(d)


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


def extractdata(paths, param):

    wanted = param.features
    ncrmode = param.ncrmode
    # TODO: this dependency on the "antibody" parameter is entirely
    # superfluous at this point, and should be factored out
    antibody = param.antibody

    rawdata = sdc_extract._extract_wells_data(paths, wanted)

    warnings = None
    if ncrmode:
        ks = rawdata.keys()
        vs = rawdata.values()

        def _cull_zeros(d):
            assert d.shape[1] == 2
            ret = d[d[:, 1] > 0.]
            return ret, len(d) - len(ret)

        cvs, nculled = zip(*map(_cull_zeros, vs))
        tot = sum(nculled)
        if tot > 0:
            ss = ' + '.join(map(str, nculled))
            warnings = ['data for %s = %d cells had to be culled '
                        'to prevent division by zero' % (ss, tot)]

        def _to_ncratio__trash(d):
            # the lambda function below casts an array with shape (n,)
            # to one with shape (n, 1); without this cast, hstack
            # produces an array of shape (2*n,), whereas we want one
            # with shape (n, 2).
            return np.hstack([(d[:, 0]/d[:, 1])[:, None],
                              (d[:, 2]/d[:, 3])[:, None]])

        nvs = [d[:, 0]/d[:, 1] for d in cvs]
        data = dict(zip(ks, nvs))
    else:
        data = rawdata

    return (dict([(k + (antibody,), v.reshape((v.size, 1)))
                  for k, v in data.items()]), warnings)


def makeheaders(headers):
    lines = []
    for i, w in enumerate(headers):
        lines.append('# column %d: %s' % (i + 1, w))
    return lines

def makepreamble(rawheaders, warnings=[]):
    rh = rawheaders
    preamble = dict([k, makeheaders(v)]
                    for k, v in rh.items())
    if warnings:
        info = []
        for w in warnings:
            info.append('### WARNING: %s' % w)

        for v in preamble.values():
            v.extend(info)

    return preamble


def process(data, param):
    statpat = param.statpat
    coord_headers = param.coord_headers

    def _tolist(d):
        return list(d) if hasattr(d, '__iter__') else [d]

    def _tofloat(d):
        return [map(float, _tolist(r)) for r in d]

    def _maybe_reshape(d):
        return d.reshape((d.size, 1)) if len(d.shape) == 1 else d

    def _mean_and_std(d):
        dd = _maybe_reshape(d)
        return np.hstack(zip(dd.mean(0, np.float64),
                             dd.std(0, np.float64)))

    # single-cell data
    cd = sum([[_tolist(coords) + row for row in _tofloat(arr)]
              for coords, arr in sorted(data.items())], [])

    ch = coord_headers + [statpat % 'cm']

    # per-field mean & std of cell means
    fd0 = np.vstack(map(_mean_and_std, data.values()))
    fd = [_tolist(coords) + _tofloat([arr])[0]
          for coords, arr in sorted(zip(data.keys(), fd0))]
    fh = coord_headers + [statpat % s for s in 'fm', 'fs']

    # per-well mean & std of cell means
    wd0 = _mean_and_std(np.vstack(data.values()))

    # per-well mean & std of field means
    wd1 = _mean_and_std(fd0.take([0], axis=1))

    wd = _tofloat([np.hstack([np.hstack([wd0, wd1])])])
    wh = [statpat % s for s in 'wcm wcs wfm wfs'.split()]

    return (dict(cell=cd, field=fd, well=wd),
            dict(cell=ch, field=fh, well=wh))


def _encode_ndarray(nd):
    for row in nd:
        yield [d.hex() if hasattr(d, 'hex') else str(d)
               for d in row]

def dump_csv(path, data, preamble):
    with open(path, 'w') as outfh:
        for line in preamble:
            print >> outfh, line
        writer = csv.writer(outfh)
        writer.writerows(_encode_ndarray(data))


def dump(basedir, todump):
    for level, v in todump.items():
        # print op.join(basedir, level + '.csv')
        dump_csv(op.join(basedir, level + '.csv'), **v)


def transpose_map(map_):
    ks = set([tuple(v.keys()) for v in map_.values()])
    assert len(ks) == 1
    ret = dict([(k, dict()) for k in ks.pop()])
    for k0, v0 in map_.items():
        for k1, v1 in v0.items():
            ret[k1][k0] = v1
    return ret
    
def getcontrols(path):
    ret = []
    with open(op.join(path, '.METADATA.csv'), 'r') as fh:
        for line in fh:
            if not line.startswith('# CONTROL'):
                continue
            c = re.sub(r'# CONTROL\s+', r'',
                       re.split(r'\s*:\s*', line)[0])
            ret.append(c)
        assert len(ret)
        return ret


def main(argv):
    _parseargs(argv)

    def want_h5(b, d, i):
        return b == 'Data.h5'

    subdir = '.DATA'
    extent, path, antibody, readout = [getattr(PARAM, a) for a in
                                       'extent path antibody readout'.split()]

    if extent == 'plate':
        q = [(sum([list(find(op.join(path, w), want_h5))
                   for w in c.split(',')], []),
              op.join(path, subdir, antibody, c, readout))
             for c in getcontrols(path)]
    else:
        assert extent == 'well'
        q = [(find(path, want_h5),
              op.join(path, subdir, antibody, readout))]

    for paths, basedir in q:
        data, warnings = extractdata(paths, PARAM)
        processed, rawheaders = process(data, PARAM)
        preamble = makepreamble(rawheaders, warnings)
        mkdirp(basedir)
        dump(basedir, transpose_map(dict(data=processed, preamble=preamble)))
        
    return 0


if __name__ == '__main__':
    exit(main(sys.argv))
