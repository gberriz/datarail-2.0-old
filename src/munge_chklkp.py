import sys
from collections import namedtuple

path = sys.argv[1]

KeyCoords = namedtuple('KeyCoords',
                       'cell_line ligand_name ligand_concentration time signal')

ValCoords = namedtuple('ValCoords',
                       'assay plate well field channel antibody')

def convert(s):
    return s.decode('utf-8')

with open(path) as fh:
    assert 'cell_line' in fh.next()
    print ','.join(KeyCoords._fields + ('\t',) + ValCoords._fields)

    for line in fh:
        h, t = line.strip().split('\t')
        k = KeyCoords(*map(convert, h.split(','))[:5])
        v = ValCoords(*map(convert, t.split(',')))
        print u','.join(map(unicode, k) + [u'\t'] + map(unicode, v)).encode('utf-8')
