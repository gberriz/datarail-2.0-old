import sys
import collections as co
import itertools as it

import pandas as pd

class Bag(object):
    # the use of ____ instead of 'self' as the first argument in __init__'s
    # signature is meant to avoid conflicts with a possible 'self' keyword in
    # kwargs (a very poor solution to the problem).
    def __init__(____, **kwargs):
        ____.__dict__.update(**kwargs)


def read_int(s):
    return unicode(int(round(float(s))))

def read_float(s):
    return unicode(float(s))

def metadata_reader(type_, _gl=globals()):
    return _gl.get('read_%s' % type_, unicode)

OPTS = Bag(iterator_chunksize=10000000)

def _sizeofrec(rec):
    return sum(map(sys.getsizeof, rec))


def df_from_iter(iterator, chunksize=None, **kwargs):
    first = next(iterator)
    stride = max((OPTS.iterator_chunksize if chunksize is None
                  else chunksize)//sum(map(sys.getsizeof, first)), 1)
    chunks = []
    batch = [first] + list(it.islice(iterator, stride - 1))
    # totsize = 0
    nread = 0

    while batch:
        # bsize = sum(map(_sizeofrec, batch))
        # print stride, bsize
        chunks.append(pd.DataFrame(batch, **kwargs))

        # totsize += bsize
        nread += stride
        # print nread
        # stride = max((nread * chunksize)//totsize, 1)

        batch = list(it.islice(iterator, stride))

    return pd.concat(chunks).reset_index(drop=True)

def mkprocessor(converters, delim=u'\t'):
    def process_line(line):
        return tuple(c(s.strip()) for c, s in
                     zip(converters,
                         unicode(line[:-1]).split(delim)))
    return process_line

def mkiter(fh, converters):
    process_line = mkprocessor(converters)
    return (process_line(l) for l in fh)


def _parse_hline(line):
    _hrecord = co.namedtuple(u'_hrecord', u'name type category')
    _w = len(_hrecord._fields)
    def _(line):
        return _hrecord(*unicode(line).split()[:_w])
    global parse_hline
    _parse_hline = _
    return _parse_hline(line)


def _parse_headers(fh, _column=co.namedtuple(u'_column', u'name converter')):
    for line in fh:
        if line.startswith(u'#'): continue
        if line.startswith(u'*'): break

        hrec = _parse_hline(line)

        if hrec.category == u'metadata':
            proc = metadata_reader(hrec.type)
        else:
            assert hrec.category == u'data'
            proc = eval(hrec.type)

        yield _column(hrec.name, proc)


def read_dataframe(path):

    with open(path) as fh:
        columns, converters = zip(*_parse_headers(fh))
        return df_from_iter(mkiter(fh, converters), columns=columns)
