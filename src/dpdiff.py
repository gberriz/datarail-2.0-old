# -*- coding: utf-8 -*-

from pdb import set_trace as ST

import datapflex
from kyub import Kyub
from collect_dpf import Column

def diff(path_a, path_b):
    def _getkyub(path):
        return Kyub(*datapflex.read_datapflex(path))

    diffs, dkyub = Kyub.diff(*map(_getkyub, (path_a, path_b)))
    return '\n'.join(diffs), dkyub
#     for ch in dkyub.readouts:
#         m = str(max(dkyub.get_column(ch)))
#         if m != '0.0':
#             print '%s: %s' % (ch, str(m))
#     return diffs, dkyub

if __name__ == '__main__DISCARD':
    from glob import glob
    import os.path as op

    irdir = '/home/gfb2/IR'
    dpdirs0, dpdirs1 = dpdirs = [op.join(irdir, d)
                                 for d in 'DATAPFLEX', 'DATAPFLEX_111018T']

    expected_template = {
        u'GF': u"""
Factors differ in details.
coord(s) found only in first kyub:
  %(assay)s,CTRL,0,0
  %(assay)s,CTRL,0,10
  %(assay)s,CTRL,0,30
  %(assay)s,CTRL,0,90
coord(s) found only in second kyub:
  %(assay)s,CTRL-GF-1,0,10
  %(assay)s,CTRL-GF-1,0,30
  %(assay)s,CTRL-GF-1,0,90
  %(assay)s,CTRL-GF-100,0,10
  %(assay)s,CTRL-GF-100,0,30
  %(assay)s,CTRL-GF-100,0,90
  """.strip(),
        u'CK': u"""
Factors differ in details.
readout(s) found only in first kyub:
  NF-κB-m-488
  NF-κB-m-488=stdev
  NF-κB-m-647
  NF-κB-m-647=stdev
readout(s) found only in second kyub:
  NF-κB-m-488 (ncr)
  NF-κB-m-488 (ncr)=stdev
  NF-κB-m-647 (ncr)
  NF-κB-m-647 (ncr)=stdev
coord(s) found only in first kyub:
  %(assay)s,CTRL,0,0
  %(assay)s,CTRL,0,10
  %(assay)s,CTRL,0,30
  %(assay)s,CTRL,0,90
coord(s) found only in second kyub:
  %(assay)s,CTRL-CK-L-1,0,10
  %(assay)s,CTRL-CK-L-1,0,30
  %(assay)s,CTRL-CK-L-1,0,90
  %(assay)s,CTRL-CK-L-100,0,10
  %(assay)s,CTRL-CK-L-100,0,30
  %(assay)s,CTRL-CK-L-100,0,90
  %(assay)s,CTRL-CK-R-1,0,10
  %(assay)s,CTRL-CK-R-1,0,30
  %(assay)s,CTRL-CK-R-1,0,90
  %(assay)s,CTRL-CK-R-100,0,10
  %(assay)s,CTRL-CK-R-100,0,30
  %(assay)s,CTRL-CK-R-100,0,90
  """.strip(),
        }

    for p in sorted(glob('scans/linkfarm/*')):
        assay = op.basename(p)
        for z in ('GF', 'CK'):
            f0, f1 = fs = [op.join(d, '%s_%s.csv' % (assay, z)) for d in dpdirs]
            if not op.exists(f1):
                continue
            diffs, dkyub = diff(f0, f1)
            out = []
            for ch in dkyub.readouts:
                m = max(dkyub.get_column(ch))
                if str(m) != '0.0' and m > 1e-6:
                    out.append((u'%s: %s' % (ch, str(m))).encode('utf-8'))

            expected = expected_template[z] % locals()
            if diffs != expected:
                lcp = op.commonprefix(diffs, expected)
                lls = lc, la, lb = map(len, (lcp, diffs, expected))
                ST()
                sfxs = [s[lc:] if len(s) > lc else None for s in (diffs, expected)]
                ST()
                print sfxs
            if out:
                print '%s %s' % (assay, z)
                print '\n'.join(out)
                print
        #exit(0)

if __name__ == '__main__':
    from glob import glob
    import os.path as op
    def _getkyub(path):
        return Kyub(*datapflex.read_datapflex(path))

#     irdir = '/home/gfb2/IR'
#     dpdir = op.join(irdir, 'DATAPFLEX_111018T')
    import sys
    dpdir = sys.argv[1]
    outpath = op.join(dpdir, 'ALL.csv')

    ks = map(_getkyub,
             [p for p in [op.join(dpdir, '%s_%s.csv' % (op.basename(sp), z))
                          for sp in sorted(glob('scans/linkfarm/*'))
                          for z in ('GF', 'CK')]
              if op.exists(p) and p != outpath])

    k = Kyub.merge(*ks)
    def _mkTC(hu):
        p = hu.split('=')
        h = p[0]
        u = p[1] if len(p) > 1 else None
        return Column(h, iterable=k.get_treatment_column(hu), units=u)

    def _mkDC(h):
        return Column(h, iterable=k.get_column(h))

#     def _mkDC(h):
#         m = k.get_column(h)
#         s = k.get_column(h + '=stdev')
#         return MSColumn(h, iterable=zip(m, s))

    tc = [_mkTC(f) for f in k.factors]
    dc = [_mkDC(f) for f in k.readouts]
    #dc = [_mkDC(f) for f in k.readouts if not f.endswith('=stdev')]

    datapflex.write_datapflex(outpath, tc, dc)

#     for p in sorted(glob('scans/linkfarm/*')):
#         assay = op.basename(p)
#         for z in ('GF', 'CK'):
#             f0, f1 = fs = [op.join(d, '%s_%s.csv' % (assay, z)) for d in dpdirs]
#             if not op.exists(f1):
#                 continue
#             diffs, dkyub = diff(f0, f1)
#             out = []
