import sys
import os.path as op
from glob import glob
from collections import defaultdict

from dump_well_metadata import DEFAULT_LAYOUTS, Control
from multidict import MultiDict
from noclobberdict import NoClobberDict
from icbp45_utils import scrape_coords

from pdb import set_trace as ST

def _parseargs(argv):
    path = argv[1]
    mode = argv[2].lower() if len(argv) > 2 else ''

    assay, _, _, _ = scrape_coords(path)

    d = dict()
    l = locals()
    params = ('path mode assay')
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


def jglob(*args):
    return sorted(glob(op.join(*args)))


def main(argv):
    _parseargs(argv)
    d, b = op.split(PARAM.path)
    mode = PARAM.mode
    _, assay = op.split(d)
    tvals = defaultdict(NoClobberDict)
    for p in jglob(d, b + '?'):
        _, plate = op.split(p)
        layout = DEFAULT_LAYOUTS.get(op.join(assay, plate),
                                     DEFAULT_LAYOUTS.get(plate))

        '''
          1         2         3         4         5         6         7         8       
*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*1234567
'''
        if mode == 't':
            print plate
            layout.dump()

        '''
                                                                                                    1                        
          1         2         3         4         5         6         7         8         9         0         1         2    
*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*1234
'''
        if mode == 'c':
            print plate
            for c in layout.controls:
                print c
                for rn in 'wells', 'zone', 'region':
                    reg = getattr(c, rn)
                    print rn
                    reg.show()
                    print
                print
                print
            print

        for k, v in layout.all_tvals().items():
            for rvals, wells in v.items():
                tvals[k][rvals] = (plate, wells)

    if mode == '':
        for tv, v in sorted(tvals.items()):
            for rv, pwc in sorted(v.items()):
                plate, wc = pwc
                if type(wc) == Control:
                    print plate, tv, rv, wc
                elif len(wc) != 1:
                    print 'skipped:', plate, tv, rv, wc
                else:
                    print plate, tv, rv, wc[0]

    return 0


if __name__ == '__main__':
    exit(main(sys.argv))
