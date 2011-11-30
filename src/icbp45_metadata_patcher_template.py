# -*- coding: utf-8 -*-
from collections import namedtuple

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
    path = argv[1]

    d = dict()
    l = locals()
    params = ('path')
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


def print_record(segments, _sep0=PARAM.sep[0], _sep1=PARAM.sep[1],
                 _enc=PARAM.encoding):
    print _sep0.join([_sep1.join(map(output_form, seg))
                      for seg in segments]).encode(_enc)

def main(argv):
    _parseargs(argv)

    path = PARAM.path

    with open(path) as fh:
        KeyCoords, ValCoords = [namedtuple(n, c)
                                for n, c in zip(('KeyCoords', 'ValCoords'),
                                                parse_line(fh.next()))]

        # modify headers?
        print_record([nt._fields for nt in KeyCoords, ValCoords])

        for line in fh:
            key, val = [clas(*tpl) for clas, tpl in
                        zip((KeyCoords, ValCoords), parse_line(line))]

            # modify key or val
            print_record((key, val))


if __name__ == '__main__':
    import sys
    exit(main(sys.argv))
