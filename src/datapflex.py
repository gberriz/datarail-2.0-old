import sys
import csv
from collections import defaultdict
import re
from pdb import set_trace as ST

from noclobberdict import NoClobberDict

class Column(list):
    def __new__(cls, name, iterable=[], units=None):
        if not hasattr(iterable, '__iter__'):
            raise TypeError('argument 2 to %s constructor '
                            'must be an iterable' % cls.__name__)
        if not hasattr(iterable, '__len__'):
            raise TypeError('argument 2 to %s constructor '
                            'must have a length' % cls.__name__)
        return super(Column, cls).__new__(cls, iterable)

    def __init__(self, name, iterable=[], units=None):
        super(Column, self).__init__(iterable)
        self.name = name
        self.units = units

    def __getslice__(self, i, j):
        iterable = super(Column, self).__getslice__(i, j)
        ret = type(self)(self.name, iterable=iterable)
        ret.__dict__.update(self.__dict__)
        return ret

    def __repr__(self):
        sig = u', '.join([(u'%r' % self.name),
                         (u'iterable=%s' % super(Column, self).__repr__()),
                         (u'units=%r' % self.units)])
        return (u'%s(%s)' % (type(self).__name__, sig)).encode('utf-8')


class MSColumn(Column):
    def __init__(self, name, iterable=[]):
        self._check_contents(*iterable)
        super(MSColumn, self).__init__(name, iterable)
        self.__dict__.pop('units')


    @staticmethod
    def _is_invalid_elem(e, n):
        return not (hasattr(e, '__iter__') and
                    hasattr(e, '__len__') and
                    len(e) == n)

    @classmethod
    def _width(cls):
        return 2

    @classmethod
    def _check_contents(cls, *els):
        w = cls._width()
        for e in els:
            if MSColumn._is_invalid_elem(e, w):
                raise TypeError('elements of "%s" object must be '
                                'iterables of length %d' %
                                (cls.__name__, w))

    @property
    def mean(self):
        return self.__dict__.setdefault('_mean',
                                        Column(self.name,
                                               zip(*self)[0]))

    @property
    def stdev(self):
        return self.__dict__.setdefault('_stdev',
                                        Column(self.name,
                                               zip(*self)[1]))
                                               
                

    def append(self, e):
        self._check_contents(e)
        super(MSColumn, self).append(e)


    def extend(self, els):
        self._check_contents(*els)
        super(MSColumn, self).extend(els)


    def __setitem__(self, i, e):
        self._check_contents(e)
        super(MSColumn, self).__setitem__(i, e)


    def __setslice__(self, i, j, els):
        self._check_contents(*els)
        super(MSColumn, self).__setslice__(self, i, j, els)



def greekToEnglish(s, map_={u'\u03b1': u'alpha', u'\u03b3': u'gamma',
                            u'\u03ba': u'kappa'}):
    return u''.join([map_.get(c, c) for c in list(s)])
                       

def encode_record(r):
    return [greekToEnglish(unicode(c)).encode('utf-8') for c in r]


def print_table(fh, headers, row_iterator, lineterminator='\r\n'):
    writer = csv.writer(fh, lineterminator=lineterminator)
    writer.writerow(encode_record(headers))
    wrapped_iterator = (encode_record(r) for r in row_iterator)
    writer.writerows(wrapped_iterator)


def write_datapflex(path, treatment_columns, data_columns, info_columns=()):

    treatment_columns = tuple(treatment_columns)
    data_columns = tuple(data_columns)
    info_columns = tuple(info_columns)

    if len(treatment_columns) == 0:
        raise TypeError('there must be at least one treatment column')

    if len(data_columns) == 0:
        raise TypeError('there must be at least one data column')
        
    if 1 != len(set(map(len,
                        treatment_columns + data_columns + info_columns))):
        raise TypeError('columns passed to write_datapflex must '
                        'all have the same length')

    nrows = len(treatment_columns[0])
    noheader_columns = tuple([Column(u'', i)
                              for i in info_columns + ([u''] * nrows,)])

    data_subcolumns = tuple(sum([(Column(c.name, c.mean),
                                  Column(u'%s=stdev' % c.name, c.stdev))
                                 if isinstance(c, MSColumn) else (c,)
                                 for c in data_columns], ()))

    def col_header(c):
        return c.name if c.units is None else u'%s=%s' % (c.name, c.units)

    all_colsets = treatment_columns, noheader_columns, data_subcolumns
    all_columns = sum(all_colsets, ())
    headers = tuple(col_header(c) for c in all_columns)

    def ith_subrecord(i, colset):
        return tuple(c[i] for c in colset)

    def ith_record(i):
        return sum([ith_subrecord(i, cs) for cs in all_colsets], ())

    row_iterator = (ith_record(i) for i in xrange(nrows))

    if path is None:
        print_table(sys.stdout, headers, row_iterator, '\n')
    else:
        with open(path, 'w') as outfh:
            print_table(outfh, headers, row_iterator)


def unique(iterable):
    return 1 == len(set(iterable))


def nodups(iterable):
    return len(iterable) == len(set(iterable))


def read_datapflex(path):
    with open(path, 'r') as inh:
        reader = csv.reader(inh)
        def _normalize_header(h):
            ret = h
            ret = ret.replace('pErk-CK', 'pErk')
            ret = ret.replace(' STDDEV', '=stdev')
            ret = re.sub(r' \((\d+)\)', r'-\1', ret)
            return ret

        def decode(s):
            return s.decode('utf-8')

        headers = map(decode, reader.next())

        nheaders = len(headers)
        skip_idxs = [i for i, l in enumerate(headers) if len(l) == 0]
        blank_idx = skip_idxs[-1]
        assert blank_idx - skip_idxs[0] + 1 == len(skip_idxs)
        first_data_idx = blank_idx + 1
        treatment_idxs = slice(0, skip_idxs[0])
        data_idxs = slice(first_data_idx, nheaders)

        dhdrs = map(_normalize_header, headers[data_idxs])
        assert nodups(dhdrs)
        tmp_table = defaultdict(NoClobberDict)
        for record in (map(decode, r) for r in reader):
            data = NoClobberDict((dhdrs[i], v) for i, v in
                                 enumerate(record[data_idxs])
                                 if len(v) > 0)
            keytuple = tuple(record[treatment_idxs])
            tmp_table[keytuple].update(data)

#         assert unique(tuple(sorted(v.keys()))
#                       for v in tmp_table.values())
        tmplt = dict((h, '') for h in dhdrs)
        table = dict((k, tmplt.copy()) for k in tmp_table.keys())
        for k in tmp_table.keys():
            table[k].update(tmp_table[k])

        return headers[treatment_idxs], dhdrs, table

