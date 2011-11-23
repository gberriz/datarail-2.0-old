# -*- coding: utf-8 -*-
from collections import namedtuple

from multikeydict import MultiKeyDict as mkd

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

    d = dict()
    l = locals()
    params = ('path_to_expmap')
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

    path = PARAM.path_to_expmap

    with open(path) as fh:
        KeyCoords, ValCoords = [namedtuple(n, c)
                                for n, c in zip(('KeyCoords', 'ValCoords'),
                                                parse_line(fh.next()))]

        _ii = set([KeyCoords._fields.index('ligand_name'),
                   KeyCoords._fields.index('ligand_concentration')])
        _is = [i for i in range(len(KeyCoords._fields)) if i not in _ii]
        _j = ValCoords._fields.index('plate')
        _k = ValCoords._fields.index('well')
        del _ii
        def _reduced_kv(key, val):
            return tuple([key[i] for i in _is] + [val[_j], val[_k][1:]])

        def _delete_field(tuple_, _i=ValCoords._fields.index('field')):
            return tuple_[:_i] + tuple_[_i + 1:]

        control_conc = mkd(len(_reduced_kv(KeyCoords._fields,
                                           ValCoords._fields)), noclobber=True)

        OutputValCoords = namedtuple('OutputValCoords',
                                     _delete_field(ValCoords._fields))

        print_record([nt._fields for nt in KeyCoords, OutputValCoords])

        already_processed = set()
        for line in fh:
            key, val = [clas(*tpl) for clas, tpl in
                        zip((KeyCoords, ValCoords), parse_line(line))]

            idx = _delete_field(val)
            if idx in already_processed:
                continue
            already_processed.add(idx)

            rk = _reduced_kv(key, val)
            if key.ligand_name == 'CTRL':
                key = key._replace(ligand_concentration=control_conc[rk])
            else:
                control_conc[rk] = key.ligand_concentration
            print_record((key, idx))


if __name__ == '__main__':
    import sys
    exit(main(sys.argv))
