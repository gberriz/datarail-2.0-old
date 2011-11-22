import sys
import os.path as op
from glob import glob
from collections import defaultdict
from itertools import ifilter
import re
import csv

from dump_well_metadata import DEFAULT_LAYOUTS, Control, KLUGE
from multidict import MultiDict
from noclobberdict import NoClobberDict
from icbp45_utils import scrape_coords

from pdb import set_trace as ST

def _parseargs(argv):
    path = argv[1]
    outdir = argv[2] if len(argv) > 2 else None
    mode = argv[3].lower() if len(argv) > 3 else ''

    assay, _, _, _ = scrape_coords(path)

    d = dict()
    l = locals()
    params = ('path mode assay outdir')
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


class DataColumn(Column):
    def __new__(cls, name, iterable=[], units=None, err=None):
        return super(DataColumn, cls).__new__(cls, iterable)

    def __init__(self, name, iterable=[], units=None, err=None):
        super(DataColumn, self).__init__(name, iterable, units)
        self.err = err

    def append(self, *args):
        nargs = len(args)
        if self.err is None:
            if nargs != 1:
                raise TypeError('append expects exactly one argument')
        else:
            if nargs != 2:
                raise TypeError('append expects exactly two arguments')
            self.err.append(args[1])
        super(Column, self).append(args[0])

    def __getslice__(self, i, j):
        iterable = super(DataColumn, self).__getslice__(i, j)
        ret = type(self)(self.name, iterable=iterable)
        ret.__dict__.update(self.__dict__)
        myerr = self.err
        ret.err = None if myerr is None else myerr[i:j]
        return ret

#     def __setslice__(self, i, j, v=None, verr=None):
#         assert self.err is not None
#         assert not (v is None and verr is None)
#         for w in v, verr:
#             assert w is None or (hasattr(w, '__iter__') and
#                                  hasattr(w, '__len__'))

#         if v is not None:
#             if verr is None:
#                 assert len(v) == j - i
#             else:
#                 assert len(v) == len(verr)
#             super(DataColumn, self).__setslice__(i, j, v)

#         if verr is not None:
#             if v is None:
#                 assert len(verr) == j - i
#             self.err[i:j] = verr


# class Kyub(object):
#     def __init__(self, kcols, dcols):
#         if set(map(len, kcols + dcols)) != 1:
#             raise TypeError('arguments to constructor must comprise '
#                             'sequences of uniform length')
#         self.kcols = kcols
#         self.dcols = dcols

#     def pivot(self, n):
#         kcols = self.kcols
#         nkcols = len(kcols)
#         if n >= nkcols:
#             raise TypeError('argument must be less than %d' % nkcols)

#         newkcols = (s[:0] for s in kcols)
#         dcolsd = defaultdict(Column)
#         for t in zip(*(kcols + dcols)):
            


def psort(master, *mutable_sequences, **kwargs):
    if any([not hasattr(ms, '__setitem__')
            for ms in mutable_sequences]):
        raise TypeError('arguments 2, 3,... to psort must be mutable '
                        'sequences')
    nms = len(mutable_sequences)
    if nms < 1:
        raise TypeError('psort requires at least one mutable sequence '
                        '(got %d)' % nms)
    try:
        if len(set(map(len, (master,) + mutable_sequences))) != 1:
            raise TypeError()
    except:
        raise TypeError('arguments to psort must have equal '
                        '(finite) lengths')

    key = kwargs.pop('key', None)
    invalid = ', '.join(['"%s"' % k for k in kwargs.keys()])
    if invalid:
        raise TypeError('unsupported parameters: %s' % invalid)

    if key is None:
        key_ = lambda p: p[1]
    else:
        key_ = lambda p: key(p[1])

    jj = [x[0] for x in sorted(enumerate(master), key=key_)]
    for ms in mutable_sequences:
        ms[:] = [ms[j] for j in jj]


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
    headers = map(col_header, all_columns)

    def ith_subrecord(i, colset):
        return tuple(c[i] for c in colset)

    def ith_record(i):
        return sum([ith_subrecord(i, cs) for cs in all_colsets], ())
                    
    def greekToEnglish(s, map_={u'\u03b1': u'alpha', u'\u03b3': u'gamma',
                                u'\u03ba': u'kappa'}):
        return u''.join([map_.get(c, c) for c in list(s)])
                           

    def encode_record(r):
        return [greekToEnglish(unicode(c)).encode('utf-8') for c in r]

    def print_table(fh, lineterminator='\r\n'):
        writer = csv.writer(fh, lineterminator=lineterminator)
        writer.writerow(encode_record(headers))
        writer.writerows(encode_record(ith_record(i)) for i in xrange(nrows))

    if path is None:
        print_table(sys.stdout, '\n')
    else:
        with open(path, 'w') as outfh:
            print_table(outfh)
            

def main(argv):
    _parseargs(argv)
    dname, gfck = op.split(PARAM.path)
    mode = PARAM.mode
    _, assay = op.split(dname)
    tvals = defaultdict(NoClobberDict)
    treatment_columns = None
    vcol = Column('readout')
    #dcol = DataColumn('data', [], err=[])
    dcol = MSColumn('data')
    for p in jglob(dname, gfck + '?'):
        _, plate = op.split(p)
        layout = DEFAULT_LAYOUTS.get(op.join(assay, plate),
                                     DEFAULT_LAYOUTS.get(plate))
        if treatment_columns is None:
            treatment_columns = [Column(str(l), units=l.units)
                                 for l in layout.tlayers]
        else:
            assert not any([str(l) != c.name or l.units != c.units
                            for l, c in zip(layout.tlayers,
                                            treatment_columns)])
        
        if mode == 't':
            print plate
            layout.dump()

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

    if mode == 'c':
        return 0

    def skip_rows(row, _regex=re.compile(r'^\s*(?:#|--|$)')):
        return not _regex.search(row)

    if mode == '':
        path0 = op.dirname(PARAM.path)
        collect0 = defaultdict(NoClobberDict)
        collect1 = defaultdict(NoClobberDict)
        dhs = set()
        for tv, v in sorted(tvals.items()):
            for rv, pwc in sorted(v.items()):
                plate, wc = pwc
                path1 = op.join(path0, plate)

                for antibody, target in rv:
                    if type(wc) == Control:
                        path2 = op.join(path1, '.DATA', antibody, str(wc))
                        path3 = op.join(path2, target.readout, 'well.csv')
                    elif len(wc) != 1:
                        continue
                    else:
                        path2 = op.join(path1, wc[0], '.DATA', antibody)
                        tmp = jglob(path2, '*', 'well.csv')
                        assert len(tmp) == 1
                        path3 = tmp[0]


                    headers = []
                    def skip_rows(row, _regex=re.compile(r'^\s*(?:#|--|$)'),
                                  save=headers):
                        ret = _regex.search(row)
                        if ret and not 'WARNING' in row:
                            save.append(re.sub(r'^.*:\s*', '', row.strip()))
                        return not ret

                    with open(path3) as in_:
                        for record in csv.reader(ifilter(skip_rows, in_)):
                            tgt = (u'%s (ncr)' % (target,)
                                   if 'ncr' in headers[0]
                                   else unicode(target))
                            dhs.add(tgt)
                            mean_stdev = tuple(map(float.fromhex, record[:2]))
                            dcol.append(mean_stdev)
                            vcol.append(tgt)
                            collect0[tv][tgt] = collect1[tgt][tv] = mean_stdev


        dcolnames = sorted(set(sum([d.keys() for d in collect0.values()], [])))
        table = [(
                  tv,
                  tuple(dd.get(cn, (u'', u'')) for cn in dcolnames)
                 )
                 for tv, dd in sorted(collect0.items())]

        tmp = zip(*[r[0] for r in table])
        assert len(tmp) == len(treatment_columns)
        for tc, c in zip(treatment_columns, tmp): tc.extend(c)

        assay = PARAM.assay

        # prepend a cell_line column, with a constant value ########
        kluge = Column('cell_line', [assay] * len(treatment_columns[0]))
        treatment_columns = [kluge] + treatment_columns
        del kluge
        ############################################################

        data_columns = [MSColumn(n, i) for n, i in
                       zip(dcolnames, zip(*[r[1] for r in table]))]

        outdir = PARAM.outdir
        outpath = (None if outdir is None
                   else op.join(outdir, '%s_%s.csv' % (assay, gfck)))
        write_datapflex(outpath, treatment_columns, data_columns)

    return 0


if __name__ == '__main__':
    exit(main(sys.argv))
