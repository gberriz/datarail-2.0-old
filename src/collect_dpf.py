import sys
import os.path as op
from glob import glob
from collections import defaultdict

from dump_well_metadata import DEFAULT_LAYOUTS
from multidict import MultiDict
from noclobberdict import NoClobberDict

from pdb import set_trace as ST

def _parseargs(argv):
    path = argv[1]
    mode = argv[2]

    d = dict()
    l = locals()
    params = ('path mode')
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
    #print DEFAULT_LAYOUTS.keys()
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
            layout.dump(width=None, twidth=87)

        '''
0                                                                                                   1                                                                                                   2                                                                                                   3
0         1         2         3         4         5         6         7         8         9         0         1         2         3         4         5         6         7         8         9         0         1         2         3         4         5         6         7         8         9         0
*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*123456789*
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

#     for tval, v in sorted(tvals.items()):
#         for rval, wells in sorted(v.items()):
#             print tval, rval, wells

#     for c in getcontrols():
#         data, warnings = extractdata(sum([list(find(getpath(w),
#                                                     lambda b, d, i:
#                                                     b == 'Data.h5'))
#                                           for w in c.split(',')], []))

#         processed, rawheaders = process(data)
#         preamble = makepreamble(rawheaders, warnings)

#         path, wavelength, readout = [getattr(PARAM, a) for a in
#                                      'path wavelength readout'.split()]

#         basedir = op.join(path, '.DATA', wavelength, c, readout)
#         mkdirp(basedir)
#         dump(basedir, transpose_map(dict(data=processed,
#                                          preamble=preamble)))

    return 0


if __name__ == '__main__':
    exit(main(sys.argv))
