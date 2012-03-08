# -*- coding: utf-8 -*-
import os.path as op
import re
from types import StringTypes
from collections import namedtuple
from itertools import imap, product
import codecs
import csv

from find import find

OrderedSet = list
# TODO: implement OrderedSet.index
# TODO: get rid of the optional argument kluge in is_valid_rc, etc.
# from orderedset import OrderedSet


CELL_LINE_ASSAY_NAMES = OrderedSet('''
20100924_HCC1187 20100925_HCC1806 20100928_CAMA1 20101004_HCC1954
20101005_AU565 20101006_HCC1569 20101007_BT20 20101008_HCC38
20101012_MCF7 20101018_HCC70 20101021_RA_Rob 20101022_N_Rob
20101025_HCC1419 20101122_HCC1937 20101202_SKBR3 20101206_MCF10A
20101210_SKBR3 20101213_HCC1395 20101215_ZR751 20101216_HCC202
20101221_HCC1428 20101222_MDAMB231 20110128_ZR7530 20110201_SKBR3
20110210_MCF7 20110308_MDAMB175 20110310_HCC1500 20110311_MDAMB453
20110317_MDAMB231 20110318_Hs578T 20110321_T47D 20110322_BT549
20110324_MDAMB361 20110325_MDAMB157 20110330_UACC893 20110414_MCF10F
20110415_MCF12A 20110418_BT474 20110420_UACC812 20110421_184B5
20110421_MDAMB415 20110427_MDAMB436 20110502_BT-483 20110517_MDAMB134
'''.strip().split())

PLATE_NAMES = OrderedSet('GF1 GF2 GF3 GF4 CK1 CK2'.split())

ROW_NAMES = OrderedSet('ABCDEFGH')
COL_NAMES = OrderedSet(['%02d' % (i + 1) for i in range(12)])
WELL_NAMES = OrderedSet([''.join(p) for p in product(ROW_NAMES, COL_NAMES)])
FIELD_NAMES = OrderedSet('1234')
COMPONENTS = [CELL_LINE_ASSAY_NAMES, PLATE_NAMES, WELL_NAMES,
              FIELD_NAMES]

def is_valid_rc(rc, WELL_NAMES=set(WELL_NAMES)):
    return rc in WELL_NAMES

def is_valid_platename(platename, PLATE_NAMES=set(PLATE_NAMES)):
    return platename in PLATE_NAMES

def is_valid_cell_line_assay(string, CELL_LINE_ASSAY_NAMES=set(CELL_LINE_ASSAY_NAMES)):
    return string in CELL_LINE_ASSAY_NAMES

def is_valid_fieldname(fieldname, FIELD_NAMES=set(FIELD_NAMES)):
    return fieldname in FIELD_NAMES

def rc2idx(rc):
    """
    Convert a row-column well descriptor into a well index.

    The argument may be a row-column sequence or a string.  In the
    latter case, the string will be canonicalized by splitting it into
    a row-column sequence where the row is the first character in the
    string and the column is the remainder of the string.

    The row must be a one-character string.  The column must be either
    an integer, or a string convertible to an integer.

    >>> rc2idx('C06')
    42
    >>> rc2idx('C06')
    42
    >>> rc2idx(('C', '06'))
    42
    >>> rc2idx(['C', 6])
    42
    >>> rc2idx('C6')
    42
    >>> rc2idx('c06')
    42
    >>> rc2idx('C006')
    42
    >>> rc2idx(['C06'])
    Traceback (most recent call last):
        ...
    TypeError: argument must be a string or a sequence of length 2 (got "['C06']")
    >>> rc2idx('I01')
    Traceback (most recent call last):
        ...
    ValueError: invalid rc: 'I01'
    >>> rc2idx(('H', 0))
    Traceback (most recent call last):
        ...
    ValueError: invalid rc: ('H', 0)
    >>> rc2idx(('C0', '6'))
    Traceback (most recent call last):
        ...
    ValueError: invalid rc: ('C0', '6')
    >>> rc2idx(('', 'C06'))
    Traceback (most recent call last):
        ...
    ValueError: invalid rc: ('', 'C06')
    >>> rc2idx('spam')
    Traceback (most recent call last):
        ...
    ValueError: invalid rc: 'spam'
    >>> rc2idx('C6.0')
    Traceback (most recent call last):
        ...
    ValueError: invalid rc: 'C6.0'
    >>> rc2idx('C')
    Traceback (most recent call last):
        ...
    ValueError: invalid rc: 'C'
    >>> rc2idx('')
    Traceback (most recent call last):
        ...
    ValueError: invalid rc: ''

    >>> rc2idx('C', 6)
    Traceback (most recent call last):
        ...
    TypeError: rc2idx() takes exactly 1 argument (2 given)
    """
    return _rc2idx(rc)

def _checkrc(ridx, cidx, rowsize, colsize):
    if not (-1 < ridx < colsize and -1 < cidx < rowsize):
        raise ValueError

# lambda r, c, rr, cc: r + c * cc
def __default_rc2idx(ridx, cidx, IGNORED, colsize):
    return ridx + cidx * colsize


def _default_rc2idx(ridx, cidx, rowsize=12, colsize=8):
    _checkrc(ridx, cidx, rowsize, colsize)
    return __default_rc2idx(ridx, cidx, _rowsize, _colsize)

def _rc2idx(rc,
            _base=ord('A'),
            _rowsize=12,
            _colsize=8,
            _convert=__default_rc2idx):

    if isinstance(rc, StringTypes):
        c0 = rc[1:]
    elif hasattr(rc, '__iter__') and len(rc) == 2:
        c0 = rc[1]
    else:
        raise TypeError('argument must be a string '
                        'or a sequence of length 2 '
                        '(got "%r")' % (rc,))
    try:
        try:
            ridx = ord(rc[0].upper()) - _base
        except (IndexError, TypeError):
            raise ValueError
        cidx = int(c0) - 1
        _checkrc(ridx, cidx, _rowsize, _colsize)
        return _convert(ridx, cidx, _rowsize, _colsize)
    except ValueError:
        raise ValueError('invalid rc: %r' % (rc,))


def idx2rc(idx):
    """
    Convert a well index (in range(96)) into a (row, column) tuple.

    >>> idx2rc(42)
    ('C', '06')
    >>> idx2rc(96)
    Traceback (most recent call last):
        ...
    ValueError: argument must be a non-negative integer less than 96
    """
    return _idx2rc(idx)


def _idx2rc(idx,
            _row_names=tuple('ABCDEFGH'),
            _col_names=tuple(['%02d' % (i + 1) for i in xrange(12)])):
    col_idx, row_idx = divmod(idx, 8)
    max_ = len(_row_names) * len(_col_names)
    if -1 < idx < max_:
        return _row_names[row_idx], _col_names[col_idx]
    raise ValueError('argument must be a non-negative integer '
                     'less than %d' % max_)


def idx2plate(idx):
    """
    Convert a plate index to a plate name.

    >>> idx2plate(4)
    'CK1'
    >>> idx2plate(8)
    Traceback (most recent call last):
        ...
    ValueError: invalid index: 8
    """

    return _idx2plate(idx)


def _idx2plate(idx,
               _plate_names=tuple('GF1 GF2 GF3 '
                                  'GF4 CK1 CK2'.split())):
    try:
        return _plate_names[idx]
    except IndexError:
        raise ValueError('invalid index: %d' % idx)
    except:
        raise TypeError('invalid index: %d' % idx)


def assay2cellline(assay):
    if not is_valid_cell_line_assay(assay):
        raise TypeError('invalid assay name: "%s"' % assay)
    return assay.split('_', 1)[1]


def plate2idx(platename):
    """
    Convert a plate name to a plate index.

    >>> plate2idx('GF3')
    2
    >>> plate2idx('CK3')
    Traceback (most recent call last):
        ...
    ValueError: invalid plate name: 'CK3'
    """
    return _plate2idx(platename)


def parse_platename(platename):
    return platename[:2], platename[2:]


def get_subassay(platename):
    return parse_platename(platename)[0]


def _plate2idx(platename):
    try:

        pfx, sfx = parse_platename(platename)

        if pfx == 'GF':
            i = -1
        elif pfx == 'CK':
            i = 3
        else:
            raise ValueError

        idx = int(sfx) + i
        assert platename == idx2plate(idx)
        return idx
    except (ValueError, IndexError):
        raise ValueError('invalid plate name: %r' % platename)


def fieldno2idx(fieldno):
    """
    Convert a field number to a field index.

    >>> fieldno2idx('3')
    2
    >>> fieldno2idx(3)
    2
    >>> fieldno2idx('0')
    Traceback (most recent call last):
        ...
    ValueError: invalid fieldno: '0'
    """
    return _fieldno2idx(fieldno)


def _fieldno2idx(fieldno,
                 _valid=set('1234')):
    if str(fieldno) in _valid:
        return int(fieldno) - 1
    raise ValueError('invalid fieldno: %r' % fieldno)


def _decode(s, _encoding='utf-8'):
    return s.decode(_encoding) if isinstance(s, str) else s
    

def _canonicalize_header(s,
                         _sub=re.compile(ur'[^_a-zA-Z0-9]'),
                         _encoding='utf-8'):
    return _sub.sub('_', _decode(s, _encoding)).lstrip('_')


def _csv_reader(fh):
    for line in fh:
        yield tuple(line.rstrip('\n').split(','))


def read_exp_design(path):
    with open(path) as fh:
        reader = csv.reader(fh)
        record = namedtuple('_record', map(_canonicalize_header,
                                           reader.next()))
        for r in reader:
            yield record(*map(_decode, r))


def _parse_antibody_string(antibody_string,
                           encoding,
                           _ab_name_re=re.compile(ur'^([^-]+)-α-(.*?)'
                                                  ur'(?:\s+(\d+))?$')):
    m = _ab_name_re.search(antibody_string.decode(encoding)
                           if isinstance(antibody_string, str)
                           else antibody_string)
    if m:
        parts = m.group(2, 1, 3)
        return parts[:-1] if parts[-1] is None else parts
    else:
        return None


def parse_antibody_string(antibody_string, encoding='utf-8'):
    return _parse_antibody_string(antibody_string, encoding)


def rsplit(path):
    if len(path) == 0:
        return []
    if path == '/':
        return [path]
    p, b = op.split(path)
    return rsplit(p) + [b]


def scrape_coords(path):
    tests = [is_valid_cell_line_assay,
             is_valid_platename,
             is_valid_rc,
             is_valid_fieldname,]

#     screens = [lambda s: not (s is None or test(s))
#                for test in tests]

    n = len(tests)
    parts = ([None] * n +
             [op.splitext(p)[0] for p in rsplit(path)] +
             [None] * n)
    maxlen = 0
    ret = ()
    for i in range(len(parts) - n):
    #for i in xrange(len(parts) - n):
        pts = parts[i:i+n]
#         nix = [s(p) for s, p in zip(screens, pts)]
#         print i, pts, nix
#         if any(s(p) for s, p in zip(screens, pts)):
#             continue

        matches = [1 if t(p) else 0
                   for t, p in zip(tests, pts)]

        score = sum(matches)

        if score >= maxlen:
            maxlen = score
            ret = tuple(pts)

        # print matches, score

    if maxlen == 0:
        raise ValueError('path "%s" has no ICBP45 coordinates' % path)

    return ret


def path_to_coords(path):
    parts = [op.splitext(p)[0] for p in rsplit(path)]
    for i in xrange(len(parts) - 3, 0, -1):
        if (is_valid_fieldname(parts[i + 2]) and
            is_valid_rc(parts[i + 1]) and
            is_valid_platename(parts[i])):
            return parts[i - 1:i + 3]
    raise ValueError('"%s" is not a valid path' % path)


def coords_to_field_number(coords, components=COMPONENTS):
    iv = coords_to_index_vector(coords, components)
    shape = map(len, components)
    return index_vector_to_field_number(iv, shape)


def coords_to_index_vector(coords, components):
    assert len(coords) == len(components)
    return [p.index(c) for p, c in zip(components, coords)]


def index_vector_to_field_number(iv, shape):
    n = len(iv)
    assert n == len(shape)
    if n == 0:
        return 0
    else:
        return ((shape[-1] *
                 index_vector_to_field_number(iv[:-1],
                                              shape[:-1]))
                + iv[-1])


def _path_iter(path, test):
    def wanted(basename, dirname, isdir):
        return isdir and test(basename)
    return find(path, wanted)

def is_valid_rc(rc, WELL_NAMES=set(WELL_NAMES)):
    return rc in WELL_NAMES

def is_valid_platename(platename, PLATE_NAMES=set(PLATE_NAMES)):
    return platename in PLATE_NAMES

def is_valid_cell_line_assay(string, CELL_LINE_ASSAY_NAMES=set(CELL_LINE_ASSAY_NAMES)):
    return string in CELL_LINE_ASSAY_NAMES

def is_valid_fieldname(fieldname, FIELD_NAMES=set(FIELD_NAMES)):
    return fieldname in FIELD_NAMES

GLOBALS = globals()
for root in 'rc platename cell_line_assay fieldname'.split():
    testfunc = GLOBALS['is_valid_%s' % root]
    def func(path, _test=testfunc):
        return _path_iter(path, _test)
    GLOBALS['%s_path_iter' % root] = func

# def rc_path_iter(path):
#     return _path_iter(path, is_valid_rc)

# def platename_path_iter(path):
#     return _path_iter(path, is_valid_platename)

# def cell_line_assay_path_iter(path):
#     return _path_iter(path, cell_line_assay_platename)

# def fieldname_path_iter(path):
#     return _path_iter(path, is_valid_fieldname)


def greek_to_english(s, map_={u'\u03b1': u'a', u'\u03b3': u'g',
                              u'\u03ba': u'k'}):
    return u''.join([map_.get(c, c) for c in list(s)])

if __name__ == '__main__':
    import doctest
    doctest.testmod()
