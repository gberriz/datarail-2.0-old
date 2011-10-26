# -*- coding: utf-8 -*-

from pdb import set_trace as ST

import datapflex
from kyub import Kyub
from collect_dpf import Column

def _getkyub(path):
    return Kyub(*datapflex.read_datapflex(path))

def _merge(ks, outpath):
    k = Kyub.merge(*ks)
    def _mkTC(hu):
        p = hu.split('=')
        h = p[0]
        u = p[1] if len(p) > 1 else None
        return Column(h, iterable=k.get_treatment_column(hu), units=u)

    def _mkDC(h):
        return Column(h, iterable=k.get_column(h))

    tc = [_mkTC(f) for f in k.factors]
    dc = [_mkDC(f) for f in k.readouts]

    datapflex.write_datapflex(outpath, tc, dc)


def main(argv):
    outpath = argv[1]
    dpfiles = argv[2:]

    _merge(map(_getkyub, dpfiles), outpath)
    return 0

if __name__ == '__main__':
    import sys
    exit(main(sys.argv))
