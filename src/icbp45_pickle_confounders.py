# -*- coding: utf-8 -*-
import os
from collections import namedtuple, defaultdict
# import pickle
import cPickle as pickle

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
    })
del __d

def _parseargs(argv):
    path_to_expmap = argv[1]
    path_to_pickle = argv[2]

    d = dict()
    l = locals()
    params = ('path_to_expmap path_to_pickle')
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


def output_form(x):
    s = x.hex() if hasattr(x, 'hex') else x
    return unicode(s)


def get_subassay(subrecord):
    return icbp45_utils.get_subassay(subrecord.plate)


def main(argv):
    _parseargs(argv)
    outpath = PARAM.path_to_pickle
    if os.path.exists(outpath):
        import sys
        print >> sys.stderr, 'warning: %s exists' % outpath

    global KeyCoords, ValCoords # globalized to enable pickling

    with open(PARAM.path_to_expmap) as fh:
        KeyCoords, ValCoords = [namedtuple(n, c)
                                for n, c in zip(('KeyCoords', 'ValCoords'),
                                                parse_line(fh.next()))]

        class Cube(mkd):
            def __init__(self, *args, **kwargs):
                super(Cube, self).__init__(len(KeyCoords._fields),
                                           noclobber=True)

        cubes = mkd(2, Cube)

        count = 0
        for line in fh:
            key, val = [clas(*tpl) for clas, tpl in
                        zip((KeyCoords, ValCoords), parse_line(line))]
            subassay = get_subassay(val)
            cubes.get((val.assay, subassay)).set(key, val)

            continue
            count += 1
            if count >= 10:
                break


    with open(outpath, 'w') as fh:
        pickle.dump(cubes.todict(), fh)

if __name__ == '__main__':
    import sys
    exit(main(sys.argv))
