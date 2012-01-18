# -*- coding: utf-8 -*-
import sys

import os.path as op
import random
import tempfile
import numpy as np
import csv
import re
from itertools import product
from functools import partial
from types import StringTypes
from copy import deepcopy
from collections import defaultdict

import sdc_extract
from shell_utils import mkdirp, cmd_with_output
from icbp45_utils import is_valid_rc, rc_path_iter, scrape_coords
from noclobberdict import NoClobberDict
from multidict import MultiDict
from orderedset import OrderedSet

from pdb import set_trace as ST

DEBUG = False

REGION = {
  u'__base__': {
    u'GF':   u'-*-',
    u'CK-L': u'-*-6',
    u'CK-R': u'-*7-',
  },
}

ANTIBODY = {
    u'NF-κB-m-488': u'530',
    u'NF-κB-m-647': u'685',
    u'pAkt-m-488': u'530',
    u'pAkt-r-647': u'685',
    u'pErk-CK-m-488': u'530',
    u'pErk-CK-m-647': u'685',
    u'pErk-m-488': u'530',
    u'pErk-r-647': u'685',
    u'pJNK-m-488': u'530',
    u'pJNK-r-647': u'685',
    u'pP38-m-488': u'530',
    u'pP38-r-647': u'685',
    u'STAT1-r-488': u'530',
    u'STAT1-r-647': u'685',
    u'STAT3-r-488': u'530',
    u'STAT3-r-647': u'685',
}

CONTROL = {
  u'GF':                   [[u'F*-6', u'-*-\F*-6',
                              {u'z': [u'ligand_concentration=0',
                                      u'ligand_concentration=0*time=0',
                                      u'time=0',],
                               u'w': [u'time=0',],},],],
  u'CK':                   [[u'D*-6', u'-*-6\D*-6',
                              {u'z': [u'ligand_concentration=0',
                                      u'ligand_concentration=0*time=0',
                                      u'time=0',],
                               u'w': [u'time=0',],},],
                            [u'D*7-', u'-*7-\D*7-',
                              {u'z': [u'ligand_concentration=0',
                                      u'ligand_concentration=0*time=0',
                                      u'time=0',],
                               u'w': [u'time=0',],},],],
  u'20100925_HCC1806/GF1': [[u'F*1,2;C*10,9;F*5,6', u'-*-',
                              {u'z': [u'ligand_concentration=0',
                                      u'ligand_concentration=0*time=0',
                                      u'time=0',],
                               u'w': [u'time=0',],},],],
}

class Antibody(tuple):
    def __new__(cls, target, readout):
        return super(Antibody, cls).__new__(cls, (target, readout))

    def __init__(self, target, readout):
        #super(Antibody, self).__init__((target, readout))
        self.target = target
        self.readout = readout

    def __str__(self):
        return self.target

KLUGE = {
  u'ligand_name': {
    u'units': None,
    u'values': set([
      u'VEGFF',
      u'EGF',
      u'EPR',
      u'BTC',
      u'HRG',
      u'CTRL-GF-1',
      u'CTRL-GF-100',
      u'FGF1',
      u'FGF2',
      u'NGF',
      u'INS',
      u'IGF-1',
      u'IGF-2',
      u'SCF',
      u'HGF',
      u'PDGFBB',
      u'EFNA1',
      u'LPS',
      u'IL-1α',
      u'IL-6',
      u'CTRL-CK-L-1',
      u'CTRL-CK-L-100',
      u'CTRL-CK-R-1',
      u'CTRL-CK-R-100',
      u'IFN-α',
      u'IFN-γ',
      u'TNF-α',
      u'IL-2',
    ]),
  },
  u'ligand_concentration': {
    u'units': 'ng/ml',
    u'values': set([
      u'0',
      u'1',
      u'100',
    ]),
  },
  u'time': {
    u'units': 'min',
    u'values': set([
      u'0',
      u'10',
      u'30',
      u'90',
    ]),
  },
  u'530_antibody': {
    u'units': None,
    u'values': set([
      Antibody(u'pAkt-m-488', 'wholecell'),
      Antibody(u'pErk-m-488', 'wholecell'),
      Antibody(u'pJNK-m-488', 'wholecell'),
      Antibody(u'pP38-m-488', 'wholecell'),
      Antibody(u'NF-κB-m-488', 'ncratio'),
      Antibody(u'pErk-CK-m-488', 'wholecell'),
      Antibody(u'STAT1-r-488', 'wholecell'),
      Antibody(u'STAT3-r-488', 'wholecell'),
    ]),
  },
  u'685_antibody': {
    u'units': None,
    u'values': set([
      Antibody(u'pErk-r-647', 'wholecell'),
      Antibody(u'pAkt-r-647', 'wholecell'),
      Antibody(u'pP38-r-647', 'wholecell'),
      Antibody(u'pJNK-r-647', 'wholecell'),
      Antibody(u'STAT1-r-647', 'wholecell'),
      Antibody(u'STAT3-r-647', 'wholecell'),
      Antibody(u'NF-κB-m-647', 'ncratio'),
      Antibody(u'pErk-CK-m-647', 'wholecell'),
    ]),
  },
}

LAYER = {
  u'ligand_name': {
    u'GF': {
      u'VEGFF':   u'A*-6',
      u'EGF':     u'B*-6',
      u'EPR':     u'C*-6',
      u'BTC':     u'D*-6',
      u'HRG':     u'E*-6',
      u'CTRL-GF-1': u'F*-6-2',
      u'CTRL-GF-100': u'F*2-6-2',
      u'FGF1':    u'G*-6',
      u'FGF2':    u'H*-6',
      u'NGF':     u'A*7-',
      u'INS':     u'B*7-',
      u'IGF-1':   u'C*7-',
      u'IGF-2':   u'D*7-',
      u'SCF':     u'E*7-',
      u'HGF':     u'F*7-',
      u'PDGFBB':  u'G*7-',
      u'EFNA1':   u'H*7-',
    },
    u'CK': {
      u'LPS':       u'A*-',
      u'IL-1α':     u'B*-',
      u'IL-6':      u'C*-',
      u'CTRL-CK-L-1': u'D*-6-2',
      u'CTRL-CK-L-100': u'D*2-6-2',
      u'CTRL-CK-R-1': u'D*7--2',
      u'CTRL-CK-R-100': u'D*8--2',
      u'IFN-α':     u'E*-',
      u'IFN-γ':     u'F*-',
      u'TNF-α':     u'G*-',
      u'IL-2':      u'H*-',
    },
    u'20100925_HCC1806/GF1': {
      u'VEGFF':   u'A*1,2;H*10,9;A*5,6',
      u'EGF':     u'B*1,2;G*10,9;B*5,6',
      u'EPR':     u'C*1,2;F*10,9;C*5,6',
      u'BTC':     u'D*1,2;E*10,9;D*5,6',
      u'HRG':     u'E*1,2;D*10,9;E*5,6',
      u'CTRL-GF-1': u'F*1;C*10;F*5',
      u'CTRL-GF-100': u'F*2;C*9;F*6',
      u'FGF1':    u'G*1,2;B*10,9;G*5,6',
      u'FGF2':    u'H*1,2;A*10,9;H*5,6',
      u'NGF':     u'A*7,8;H*4,3;A*11,12',
      u'INS':     u'B*7,8;G*4,3;B*11,12',
      u'IGF-1':   u'C*7,8;F*4,3;C*11,12',
      u'IGF-2':   u'D*7,8;E*4,3;D*11,12',
      u'SCF':     u'E*7,8;D*4,3;E*11,12',
      u'HGF':     u'F*7,8;C*4,3;F*11,12',
      u'PDGFBB':  u'G*7,8;B*4,3;G*11,12',
      u'EFNA1':   u'H*7,8;A*4,3;H*11,12',
    },
  },
  u'ligand_concentration': {
    u'GF': {
      u'1':   u'-*--2\F*-6',
      u'100': u'-*2--2\F*-6',
      u'0':   u'F*-6',
    },
    u'CK': {
      u'1':   u'-C,E-*--2',
      u'100': u'-C,E-*2--2',
      u'0':   u'D*-',
    },
    u'20100925_HCC1806/GF1': {
      u'1':   u'-*1,4,5\F*1,5;-*7,10,11\C*10',
      u'100': u'-*2,3,6\F*2,6;-*8,9,12\C*9',
      u'0':   u'F*1,2;C*10,9;F*5,6',
    },
  },
  u'time': {
    u'*': {
      u'10': u'-*1,2,7,8',
      u'30': u'-*3,4,9,10',
      u'90': u'-*5,6,11,12',
    },
  },
  u'530_antibody': {
    u'GF': {
      u'1': {u'pAkt-m-488': u'-*-'},
      u'2': {u'pErk-m-488': u'-*-'},
      u'3': {u'pJNK-m-488': u'-*-'},
      u'4': {u'pP38-m-488': u'-*-'},
    },
    # NFKB is at
    # CK1 530 u'-*-6'
    # CK2 685 u'-*-6'
    u'CK': {
      # CK1,*01,NF-κB-m-488+STAT1-r-647
      # CK1,*07,pErk-CK-m-488+STAT3-r-647
      u'1': {
        u'NF-κB-m-488': u'-*-6',
        u'pErk-CK-m-488':  u'-*7-',
      },
      # CK2,*01,STAT1-r-488+NF-κB-m-647
      # CK2,*07,STAT3-r-488+pErk-CK-m-647
      u'2': {
        u'STAT1-r-488': u'-*-6',
        u'STAT3-r-488':  u'-*7-',
      },
    },
    u'20100925_HCC1806/GF1': { # same as for GF1
      u'1': {u'pAkt-m-488': u'-*-'},
    },
  },      
  u'685_antibody': {
    u'GF': {
      u'1': {u'pErk-r-647': u'-*-'},
      u'2': {u'pAkt-r-647': u'-*-'},
      u'3': {u'pP38-r-647': u'-*-'},
      u'4': {u'pJNK-r-647': u'-*-'},
    },
    u'CK': {
      # CK1,*01,NF-κB-m-488+STAT1-r-647
      # CK1,*07,pErk-CK-m-488+STAT3-r-647
      u'1': {
        u'STAT1-r-647': u'-*-6',
        u'STAT3-r-647':  u'-*7-',
      },
      # CK2,*01,STAT1-r-488+NF-κB-m-647
      # CK2,*07,STAT3-r-488+pErk-CK-m-647
      u'2': {
        u'NF-κB-m-647': u'-*-6',
        u'pErk-CK-m-647':  u'-*7-',
      },
    },
    u'20100925_HCC1806/GF1': { # same as for GF1
      u'1': {u'pErk-r-647': u'-*-'},
    },
  },
}


def _mkoutpath(dir_=None, subdir='.METADATA'):
    if dir_ is None:
        dir_ = PARAM.path
    p = op.join(dir_, subdir)
    return p


def _parseargs(argv):
    path = argv[1]
    # e.g. scans/linkfarm/20100924_HCC1187/CK2/H12
    assay, plate, well, _ = scrape_coords(path)
#     row = well[0]
#     col = int(well[1:])
#     rownum = ord(row) - ord('A')
#     colnum = col - 1

    d = dict()
    l = locals()
    params = (# 'row col rownum colnum '
              'path assay plate well')
              
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

def extractdata(path=None, wanted=None):
    if path is None:
        path = PARAM.path
    if wanted is None:
        wanted = PARAM.features
    rawdata = sdc_extract._extract_well_data(path, wanted)
    warnings = None
    if PARAM.ncrmode:
        ks = rawdata.keys()
        vs = rawdata.values()

        def _cull_zeros(d):
            assert d.shape[1] == 4
            keep = []
            nc = 0
            for i, r in enumerate(d):
                if r[1] > 0. and r[3] > 0.:
                    keep.append(i)
                else:
                    nc += 1
            return d.take(keep, axis=0), nc

        cvs, nculled = zip(*map(_cull_zeros, vs))
        tot = sum(nculled)
        if tot > 0:
            ss = ' + '.join(map(str, nculled))
            warnings = ['data for %s = %d cells had to be culled '
                        'to prevent division by zero' % (ss, tot)]

        def _to_ncratio(d):
            # the lambda function below casts an array with shape (n,)
            # to one with shape (n, 1); without this cast, hstack
            # produces an array of shape (2*n,), whereas we want one
            # with shape (n, 2).
            return np.hstack(map(lambda x: x.reshape((x.size, 1)),
                                 [d[:, 0]/d[:, 1], d[:, 2]/d[:, 3]]))

        nvs = [_to_ncratio(d) for d in cvs]
        data = dict(zip(ks, nvs))
    else:
        data = rawdata

    return data, warnings


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


# def _mean_and_std(col):
#     m = col.mean(0, np.float64)
#     sd = col.std(0, np.float64)
#     return np.hstack((m, sd))

def _mean_and_std(d):
    return np.hstack(zip(d.mean(0, np.float64), d.std(0, np.float64)))


def _tofloat(d):
    return [map(float, r) for r in d]


def process(data):
    cd = sum([[[k] + r for r in _tofloat(v)]
              for k, v in data.items()], [])

    statpat = PARAM.statpat

    ch = ['field'] + [p % '(cm)' for p in statpat]

    # per-field mean & std of cell means
    fd0 = np.vstack(map(_mean_and_std, data.values()))
    fd = [[k] + _tofloat([v])[0]
          for k, v in zip(data.keys(), fd0)]
    fh = ['field'] + sum(([p % '(fm)', p % '(fs)']
                          for p in statpat), [])


    # per-well mean & std of cell means
    wd0 = _mean_and_std(np.vstack(data.values()))

    # per-well mean & std of field means
    wd1 = _mean_and_std(fd0.take([0, 2], axis=1))

    wd = _tofloat([np.hstack([wd0, wd1])])
    wh = (sum(([p % '(wcm)', p % '(wcs)'] for p in statpat), []) +
          sum(([p % '(wfm)', p % '(wfs)'] for p in statpat), []))

    return (dict(cell=cd, field=fd, well=wd),
            dict(cell=ch, field=fh, well=wh))


def _encode_ndarray(nd):
    for row in nd:
        yield [d.hex() if hasattr(d, 'hex') else str(d)
               for d in row]

class Plate(object):
    __A_offset = ord('A')
    COLLEN = NROWS = 8
    ROWLEN = NCOLS = 12
    SIZE = NROWS * NCOLS

    @staticmethod
    def to_rownum(r):
        return ord(r.upper()) - Plate.__A_offset
    @staticmethod
    def to_colnum(c):
        return int(c) - 1
    @staticmethod
    def from_rownum(i):
        return chr(i + Plate.__A_offset)
    @staticmethod
    def from_colnum(i):
        return '%02d' % (i + 1)

    @staticmethod
    def to_index(*rc):
        try:
            if len(rc) == 1:
                if isinstance(rc[0], str):
                    rc = (rc[0][0], rc[0][1:])
                else:
                    return rc[0]
            assert len(rc) == 2
            if isinstance(rc[0], str):
                kl = Plate
                r, c = kl.to_rownum(rc[0]), kl.to_colnum(rc[1])
            else:
                r, c = rc
            # this should really be:
            # return icbp45_utils._default_rc2idx(r, c, Plate.NCOLS, Plate.NROWS)
            return (Plate.NROWS * c) + r
        except Exception, e:
            print e
            raise TypeError('rc must be a valid well specification '
                            'of type (str,), (str, str), (int,) or '
                            '(int, int)')

    @staticmethod
    def from_index(i):
        kl = Plate
        if is_valid_rc(i):
            i = kl.to_index(i)
        c, r = divmod(i, Plate.NROWS)
        return (kl.from_rownum(r), kl.from_colnum(c))

    @staticmethod
    def getrow(i):
        return Plate.from_index(i)[0]

    @staticmethod
    def getrownum(i):
        return Plate.to_index(i) % Plate.NROWS

    @staticmethod
    def getcol(i):
        return Plate.from_index(i)[1]

    @staticmethod
    def getcolnum(i):
        return Plate.to_index(i)//Plate.NROWS
    

#class Region(set):
class Region(OrderedSet):
    def __init__(self, wells):
        ws = map(Plate.to_index, wells)
        super(Region, self).__init__(ws)
        self.size = sz = len(wells)
        assert sz == len(self)

    # The parsing done by method below is rudimentary and inadequate
    # for the long run; the parsing should be done by a bonafide
    # parser, capable of recognizing (parenthesized) recursive
    # subexpressions.  E.g. at the moment, this code can't handle an
    # expression to indicate "all wells minus A01-A03 and B04-B06";
    # this would require the expression r'-*-\(A*1-3;B*4-6)', which
    # this simpleminded implementation can't handle.
    @staticmethod
    def _expand_product_region(spec, exclude=set()):
        def _expand_range(x, max_, conv):
            # allowing x to be an empty string, to make it easier to
            # to define "empty ranges"
            if len(x) == 0:
                return []
            if not '-' in x:
                return [conv(x)]
            bes = x.split('-')
            n = len(bes)
            assert n < 4
            b = 0 if bes[0] == '' else conv(bes[0])
            e = max_ if bes[1] == '' else conv(bes[1]) + 1
            s = 1 if n == 2 or bes[2] == '' else int(bes[2])
            return range(max_).__getitem__(slice(b, e, s))

        pl = Plate
        def _expand_rrange(rr, exp=_expand_range, conv=pl.to_rownum):
            return exp(rr, Plate.NROWS, conv)

        def _expand_crange(cr, exp=_expand_range, conv=pl.to_colnum):
            return exp(cr, Plate.NCOLS, conv)

        assert ';' not in spec
        try:
            pspec, nspec = spec.split(u'\\')
        except ValueError, e:
            msg = str(e)
            if msg == 'too many values to unpack':
                raise ValueError('invalid region spec: %s' % spec)
            elif msg != 'need more than 1 value to unpack':
                raise
        else:
            me = Region._expand_product_region
            return me(pspec, exclude=set(me(nspec)))

        assert u'\\' not in spec

        rs, cs = [s.split(',') for s in spec.split('*')]
        return [w for w in [pl.to_index(r, c) for r, c in
                            product(sum(map(_expand_rrange, rs), []),
                                    sum(map(_expand_crange, cs), []))]
                if w not in exclude]

    @staticmethod
    def mkregion(spec):
        return Region(sum(map(Region._expand_product_region,
                              spec.split(';')), []))


    def to_rows(self):
        rows = [[] for _ in xrange(Plate.NROWS)]
        for w in self:
            rows[Plate.getrownum(w)].append(w)
        return rows
        
    def to_cols(self):
        cols = [[] for _ in xrange(Plate.NCOLS)]
        for w in self:
            cols[Plate.getcolnum(w)].append(w)
        return cols

    def show(self, in_=u'X', out=u''):
        width = max(map(len, (in_, out, '00')))
        def ctr(s):
            return unicode.center(unicode(s), width, u' ')
        rn = Plate.from_rownum
        cn = Plate.from_colnum
        colnums = range(Plate.NCOLS)
        rownums = range(Plate.NROWS)
        hrow = u'|'.join([u' ' * len(rn(0))] +
                         [ctr(cn(c)) for c in colnums] +
                         [u''])

        hdiv = u'-' * len(hrow)
        for line in hdiv, hrow, hdiv:
            print line
        idx = Plate.to_index
        for r, rname in enumerate(map(rn, rownums)):
            print u'|'.join([rname] +
                            [ctr(in_ if idx(r, c) in self else out)
                             for c in colnums] +
                            [u''])
            print hdiv

    def __str__(self):
        return ','.join([''.join(Plate.from_index(w)) for w in self])


class Control(object):
    def __init__(self, wells, zone, tval_specs):
        if not wells:
            raise ValueError('wells param must be a non-empty region')
        if not (wells.issubset(zone) or wells.isdisjoint(zone)):
            raise ValueError('wells must be contained in or disjoint from zone')

        regs = dict()
        self.wells = w = regs['w'] = Region(wells)
        self.zone = z = regs['z'] = Region(zone.difference(wells))
        if not z:
            raise ValueError('zone has no non-control wells')

        self.region = r = regs['r'] = Region.union(w, z)

        self.tval_specs = tvs = []

        for code, ss in tval_specs.items():
            if code == 'z':
                reg = self.zone
            elif code == 'r':
                reg = self.region
            elif code == 'w':
                reg = self.wells
            else:
                raise ValueError('unknown control region code: "%s"' % code)

            for s in ss:
                tvs.append((s, reg))

    def __str__(self):
        return str(self.wells)
                                


class Layer(object):
    def __init__(self, category=None, units=None, spec={}):
        self.category = category
        self.units = units
        self._origspec = spec
        self.spec = s = dict()
        self.w2v = t = [None for _ in range(Plate.SIZE)]
        self.v2ws = u = dict()
        self.values = spec.keys()
        allws = []
        for k, v in spec.items():
            s[k] = ws = Region.mkregion(v)
            u[k] = map(Plate.from_index, ws)
            for i in ws:
                t[i] = k
            allws.extend(ws)
        self.size = sz = len(allws)

        allws = set(allws)
        # ensure no overlaps:
        assert sz == len(allws)
        self.allws = allws

    def value(self, well):
        try:
            ret = self.w2v[Plate.to_index(well)]
        except IndexError:
            ret = None
        if ret is None:
            raise ValueError('well %s not within layer "%s"' %
                             (well, self.category))
        return ret

    def inmask(self, mask):
        return not self.allws.difference(mask)

    def fillsmask(self, mask):
        return not mask.difference(self.allws)

    def crop(self, mask):
        ret = type(self).__new__(type(self))
        ret.category = self.category
        ret.spec = s = dict()
        ret.w2v = t = [None for _ in range(Plate.SIZE)]
        ret.v2ws = u = dict()
        allws = set()
        for k, v in self.spec.items():
            s[k] = ws = filter(lambda w: w in mask, v)
            u[k] = map(Plate.from_index, ws)
            for i in ws:
                t[i] = k
            allws.update(ws)
        ret.size = sz = len(allws)
        ret.allws = allws
        return ret

    def __str__(self):
        return self.category
        

class Layout(object):
    def __init__(self, tlayers, rlayers, controls):
        tl, rl = [tlayers[:], rlayers[:]]
        mask = Region.mkregion('-*-')

        layer = dict()
        for l in tlayers + rlayers:
            k = l.category
            if k in layer:
                raise ValueError('multiple "%s" layers' % k)
            layer[k] = l
        self.layer = layer

        for layer in tl + rl:
            if not layer.inmask(mask):
                raise ValueError('"%s" layer does not lie '
                                 'within mask' % layer.category)

            if not layer.fillsmask(mask):
                raise ValueError('"%s" layer does not fill mask' %
                                 layer.category)

        if any([c.region.difference(mask) for c in controls]):
            raise ValueError('control region not within mask')
            
        self.tlayers = tl
        self.rlayers = rl
        self.controls = controls
        self.mask = mask

        self._check_controls()

    @property
    def _orphans(self):
        try:
            ret = self._memo_orphans
        except AttributeError:
            tl = self.tlayers
            ret = self._memo_orphans = \
                  [x.difference(y) for x, y in
                   zip(map(set, [l.values for l in tl]),
                       map(set, zip(*[self.values(w, tl)
                                      for w in sorted(self.mask)])))]
        return ret

    def tvals(self):
        tls = self.tlayers
        rls = self.rlayers
        mask = self.mask
        ret = defaultdict(MultiDict)

        template = ['' for _ in rls]
        for w in sorted(mask):
            tvals = self.values(w, tls)
            rvals = tuple(zip([l.category for l in rls],
                              self.values(w, rls)))
            ret[tvals][rvals] = ''.join(Plate.from_index(w))

        return dict([(k, dict(v)) for k, v in ret.items()])
        

    def implicit_tvals(self):
        tvals = dict()
        ret = defaultdict(NoClobberDict)

        tls = self.tlayers
        rls = self.rlayers

        tl_n = [t.category for t in tls]

        def _parse_spec(s):
            c, v = s.split('=')
            return (c, v.split(','))

        for c in self.controls:
            tvs = []
            for specs, region in c.tval_specs:
                wsd = MultiDict()
                for w in region:
                    # wsd[self.values(w, rls)] = w
                    rv = tuple(zip([l.category for l in rls],
                                   self.values(w, rls)))
                    wsd[rv] = w


                for rl, ws in wsd.items():
                    tl_v = map(list,
                               map(set, zip(*[self.values(w, tls)
                                              for w in sorted(ws)])))

                    assert len(tl_n) == len(tl_v)

                    available = zip(tl_n, tl_v)
                    tmp = dict(available +
                               map(_parse_spec, specs.split('*')))

                    chunk = [(tl, rl) for tl in
                             list(product(*[tmp[t] for t in tl_n]))]
                    tvs.append(chunk)

            from itertools import combinations
            assert 0 == sum(map(len, [set.intersection(set(x), set(y))
                                      for x, y in list(combinations(tvs, 2))]))

            for tv, rv in sorted(set(sum(tvs, []))):
                ret[tv][rv] = c

        return dict([(k, dict(v)) for k, v in ret.items()])


    def all_tvals(self):
        ret = defaultdict(NoClobberDict)
        for d in self.tvals(), self.implicit_tvals():
            for k, v in d.items():
                ret[k].update(v)

        return dict([(k, dict(v)) for k, v in ret.items()])


    def _check_controls(self):
        #controlled_region = set.union(*[c.region for c in self.controls])
        controlled_region = OrderedSet.union(*[c.region for c in self.controls])
        if len(controlled_region) != Plate.SIZE:
            assert len(controlled_region) < Plate.SIZE
            raise ValueError('layout contains uncontrolled wells')
        del controlled_region
        

    def _checkwell(self, well):
        mask = self.mask
        if mask:
            wi = Plate.to_index(well)
            if wi not in mask:
                raise ValueError('well %s not in layout' % well)

    def value(self, well, category):
        self._checkwell(well)
        if not category in self.layer:
            raise ValueError('unrecognized layer: "%s"' % category)
        return self._value(well, category)

    def _value(self, well, category):
        return self.layer[category].value(well)


    def values(self, well, layers=None):
        self._checkwell(well)
        if layers is None:
            layers = self.tlayers + self.rlayers
        return tuple([ly.value(well) for ly in layers])

    def plate_preamble(self):
        # only a stub for now
        return []

    def plate_metadata(self):
        ret = []
        for c in self.controls:
            wells, zone = map(str, (c.wells, c.zone))
            ret.append('# CONTROL %s : %s' % (wells, zone))
        return ret


    def well_preamble(self, well, wanted=None):
        w = Plate.to_index(well)
        return [u'# column %d: %s' % (i + 1, l.category)
                for i, l in enumerate(self.tlayers + self.rlayers)
                if wanted is None or l.category in wanted]


    def well_metadata(self, well, wanted=None):
        w = Plate.to_index(well)
        return [l.value(w)
                for l in self.tlayers + self.rlayers
                if wanted is None or l.category in wanted]


    def dump_well(self, well, wanted=None):
        w = Plate.to_index(well)
        vs = [l.value(w)
              for l in self.tlayers
              if wanted is None or l.category in wanted]

        wanted_rlayers = [l for l in self.rlayers
                          if wanted is None or l.category in wanted]

        if not wanted_rlayers:
            print '|'.join(vs)
        else:
            for l in wanted_rlayers:
                print '|'.join(vs + [l.value(w)])


    def dump(self, wanted=None, width=None, twidth=None, vdiv=u'|'):

        if not (width is None or twidth is None):
            raise ValueError('at most one of width and twidth can be '
                             'different from None')

        if width is None and twidth is None:
            from os import environ
            twidth = int(environ.get('COLUMNS', 89)) - 2

        def _mkctr(w):
            def ctr(s):
                return unicode.center(unicode(s), w, u' ')
            return ctr

        def _mklj(w):
            def lj(s):
                return unicode.ljust(unicode(s), w, u' ')
            return lj

        wellrows = self.mask.to_rows()
        cols = sorted(set(sum(map(lambda row: map(Plate.getcol, row),
                                  wellrows), [])))

        rowh = [unicode(Plate.from_rownum(i)) for i in range(Plate.NROWS)]
        lvdiv = len(vdiv)

        rowhw = 2*lvdiv + max(map(len, rowh))
        def rj(s):
            return unicode.rjust(unicode(s), rowhw - 2*lvdiv, u' ')

        rowh = map(rj, rowh)
        topleft = blankrowhdr = rj(u'')

        if twidth is not None:
            assert width == None
            # (twidth - rowhw)/ncols == width + lvdiv >= minwidth + lvdiv
            # twidth - rowhw == (width + lvdiv)*ncols >= (minwidth + lvdiv)*ncols
            # twidth == (width + lvdiv)*ncols + rowhw >= (minwidth + lvdiv)*ncols + rowhw == mintwidth
            minwidth = 1
            mintwidth = (minwidth + lvdiv)*Plate.NCOLS + rowhw
            if twidth < mintwidth:
                raise ValueError('twidth must be at least %d' % mintwidth)
            # (twidth - rowhw)//ncols - lvdiv == maxwidth
            maxwidth = ((twidth - rowhw)//Plate.NCOLS) - lvdiv
            assert maxwidth > 0

        ls = self.tlayers + self.rlayers

        if wanted is None:
            wanted = set([l.category for l in ls])
        else:
            wanted = set(wanted)

        for l in ls:
            if l.category not in wanted:
                continue

            rows = map(lambda row: map(lambda well: l.value(well), row),
                       wellrows)

            if width is None:
                w = min([maxwidth,
                         max(map(len, cols + map(unicode, _flatten(rows, 1))))])
            else:
                w = width

            vs = map(partial(strs2cols, width=w),
                     [[unicode(c) for c in r] for r in rows])
            mr = max(map(len, sum(vs, [])))
            for r in vs:
                for c in r:
                    padlns(c, mr)

            ctr = _mkctr(w)
            lj = _mklj(w)
            rowheight = mr
            for i, row in enumerate(vs):
                if i == 0:
                    hrow = vdiv.join([u'', topleft] + map(ctr, cols) + [u''])
                    top = bottom = u'-' * len(hrow)
                    hdiv = re.sub(r'[^|]', '-', hrow)
                    print top
                    print hrow

                print hdiv
                for j in range(rowheight):
                    r = rowh[i] if j == 0 else blankrowhdr
                    print vdiv.join([u'', r] + [lj(c[j]) for c in row] + [u''])

            print bottom

            print
            if rowheight > 1:
                print


def _flatten(ll, levels=None):
    def __notiterable(x):
        return not hasattr(x, '__iter__')

    def __r(ll, levels=None):
        if __notiterable(ll):
            msg = ('argument 1 to _flatten must support iteration'
                   if levels is None else
                   'specified levels parameter exceeds depth of object')
            raise TypeError(msg)

        if levels > 0:
            r = partial(__r, levels=levels-1)
        elif levels is None and not any(map(__notiterable, ll)):
            r = __r
        else:
            assert levels <= 0 or any(map(__notiterable, ll))
            return ll
        return sum(map(r, ll), [])

    if levels is not None:
        try:
            levels - 0
        except TypeError:
            raise TypeError('argument 2 to _flatten must be an integer')
    try:
        return __r(ll, levels)
    except Exception, e:
        raise type(e)(*e.args)
                                                           

def wrap(s, width):
    p = -len(s) % width
    sp = s + '\0' * p
    ret = [''.join(u) for u in zip(*[iter(sp)]*width)]
    if p > 0:
        ret[-1] = ret[-1][:-p]
    return ret


def padlns(ss, n):
    m = n - len(ss)
    if m > 0:
        ss.extend([''] * m)
    else:
        assert m == 0


def strs2cols(ss, width):
    return map(partial(wrap, width=width), ss)


def default_layouts():
    # TODO: this function a desperate kluge, top to bottom; it needs
    # to be rewritten from scratch.
    def convert(layermap, category=None):
        ks = layermap.keys()
        for k in ks:
            v = layermap[k]
            if isinstance(v.values()[0], StringTypes):
                kluge = KLUGE[category]
                #vs = kluge['values']
                #assert not any([vv not in vs for vv in v.keys()])
                #layermap[k] = Layer(category, kluge['units'], v)
                dok = dict((unicode(x), x) for x in kluge['values'])
                dov = dict((dok[k], v[k]) for k in v.keys())
                layermap[k] = Layer(category, kluge['units'], dov)
            else:
                c = k if category is None else category
                convert(v, category=category or k)

    layer = deepcopy(LAYER)
    convert(layer)

    layout = dict()

    # KLUGE
    pfxs = ['GF', 'CK', '20100925_HCC1806/GF1']
    idxs = map(range, (4, 2, 1))

    for pfx, idx in zip(pfxs, idxs):
        def _mkctrl(args):
            well, zone = map(Region.mkregion, args[:2])
            return Control(well, zone, args[2])

        controls = [_mkctrl(cc) for cc in CONTROL[pfx]]
                    
        for i in idx:
            j = '%d' % (i + 1)
            if pfx == '20100925_HCC1806/GF1':
                platename = pfx
            else:
                platename = pfx + j 

            tls = []
            rls = []
            ks = ('ligand_name ligand_concentration time '
                  '530_antibody 685_antibody'.split())

            for category in ks:
                specmap = layer[category]
                if category.endswith('antibody'):
                    rls.append(specmap[pfx][j])
                else:
                    try:
                        l = specmap[platename]
                    except KeyError:
                        try:
                            l = specmap[pfx]
                        except KeyError:
                            l = specmap['*']
                    tls.append(l)
            layout[platename] = Layout(tls, rls, controls)

    return layout
        
DEFAULT_LAYOUTS = default_layouts()

def dump_default_layout():
    layout = DEFAULT_LAYOUTS
    l0 = layout.values()[0]
    for c in [x.category for x in l0.tlayers + l0.rlayers]:
        for name, lo in sorted(layout.items()):
            print name
            lo.dump(wanted=[c])
            print


def getlayout(assay, plate, layoutmap=None):
    lm = DEFAULT_LAYOUTS if layoutmap is None else layoutmap
    subpath = op.join(assay, plate)
    return lm.get(subpath, lm[plate])


def _encode_rows(rows):
    for row in rows:
        yield [(d if isinstance(d, unicode)
                else unicode(d)).encode('utf-8') for d in row]


def dump_well_metadata(path, data, preamble):
    with open(path, 'w') as outfh:
        for line in preamble:
            print >> outfh, line.encode('utf-8')
        writer = csv.writer(outfh)
        writer.writerows(_encode_rows(data))


def dump_plate_metadata(path, data, preamble):
    with open(path, 'w') as outfh:
        for line in preamble:
            print >> outfh, line.encode('utf-8')
        for line in data:
            print >> outfh, line.encode('utf-8')


def dump(basename='.METADATA.csv'):
    layout = getlayout(PARAM.assay, PARAM.plate)
    plate_path = PARAM.path

    for p in rc_path_iter(plate_path):
        rc = op.basename(p)
        #print op.join(p, basename)
        #continue
        dump_well_metadata(op.join(p, basename),
                           [layout.well_metadata(rc)],
                           layout.well_preamble(rc))

    #print op.join(plate_path, basename)
    #return
    dump_plate_metadata(op.join(plate_path, basename),
                        layout.plate_metadata(),
                        layout.plate_preamble())

#     path = _mkoutpath()
#     with open(path, 'w') as outfh:
#         layout.dumpmetadata(outfh, PARAM.well)


def transpose_map(map_):
    ks = set([tuple(v.keys()) for v in map_.values()])
    assert len(ks) == 1
    ret = dict([(k, dict()) for k in ks.pop()])
    for k0, v0 in map_.items():
        for k1, v1 in v0.items():
            ret[k1][k0] = v1
    return ret
    
def main(argv):
    _parseargs(argv)
    dump()
    return 0


if __name__ == '__main__':
    exit(main(sys.argv))
