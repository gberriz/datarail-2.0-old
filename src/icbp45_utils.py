# -*- coding: utf-8 -*-
import re
from collections import namedtuple
from itertools import imap
import codecs
import csv

def is_valid_rc(rc):
    return _is_valid_rc(rc)


def _is_valid_rc(rc,
                 _re=re.compile(r'^[A-H](?:0[1-9]|1[0-2])$')):
    try:
        return bool(_re.search(rc))
    except:
        return False
    

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
    TypeError: argument must be a string or a sequence of length 2
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


def _rc2idx(rc,
            _base=ord('A'),
            _rowsize=12,
            _colsize=8,
            _convert=lambda r, c, rr, cc: r + c * cc):

    if isinstance(rc, str):
        c0 = rc[1:]
    elif hasattr(rc, '__iter__') and len(rc) == 2:
        c0 = rc[1]
    else:
        raise TypeError('argument must be a string '
                        'or a sequence of length 2')
    try:
        try:
            ridx = ord(rc[0].upper()) - _base
        except (IndexError, TypeError):
            raise ValueError
        cidx = int(c0) - 1
        if not (-1 < ridx < _colsize and -1 < cidx < _rowsize):
            raise ValueError
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


def _plate2idx(platename):
    try:
        pfx, sfx = platename[:2], platename[2:]
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

if __name__ == '__main__':
    import doctest
    doctest.testmod()
