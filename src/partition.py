import itertools as it

def partition(data, pred):
    """
    >>> from partition import partition
    >>> pred = lambda x: int(x) % 3 == 2
    >>> data = map(str, range(15))
    >>> partition(data, pred)
    [('2', '5', '8', '11', '14'),
     ('0', '1', '3', '4', '6', '7', '9', '10', '12', '13')]
    """

    t, f = [], []
    for d in data:
        if pred(d): t.append(d)
        else: f.append(d)
    return [tuple(t), tuple(f)]


def ipartition(data, pred):
    """
    >>> from partition import ipartition
    >>> pred = lambda x: int(x) % 3 == 2
    >>> data = imap(str, xrange(15))
    >>> ipartition(data, pred)
    [<itertools.ifilter at 0x33193d0>, <itertools.ifilterfalse at 0x3319a10>]
    >>> map(tuple, ipartition(data, pred))
    [('2', '5', '8', '11', '14'),
     ('0', '1', '3', '4', '6', '7', '9', '10', '12', '13')]
    """
    t1, t2 = it.tee(data)
    return [it.ifilter(pred, t1), it.ifilterfalse(pred, t2)]
