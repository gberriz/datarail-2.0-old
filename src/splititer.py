from itertools import izip, tee, islice

def mergesplit_demo(n, times, _minpreflen=3, _maxpreflen=8):
    from itertools import count, islice
    from random import randint

    # first, some helper functions
    def _take(iter_, n):
        return tuple(islice(iter_, n))

    def _randomprefix(iter_):
        return _take(iter_, randint(_minpreflen, _maxpreflen))

    def _printmerged(merged, hdng, _pfx='  '):
        print
        print '\n  '.join([hdng] + map(str, _take(merged, _minpreflen)))

    def _printsplit(split, hdng):
        print
        print '\n  '.join([hdng] + map(str, map(_randomprefix, split)))

    split = tuple(count(1000 * i) for i in range(1, n + 1))
    for i in range(times):
        if i == 0:
            _printsplit(split, 'original %d iterators '
                          '(random-length prefixes):' % n)

        merged = mergeiters(split)
        _printmerged(merged, 'first %d items after '
                    'merging %d time(s):' % (_minpreflen, (i + 1)))

        split = splititer(merged, n)
        _printsplit(split, 'random-length prefixes after '
                      'splitting %d time(s):' % (i + 1))


def mux(iters):
    """
    Multiplex the iterators in iters.

    iters is a (finite) sequence of iterators

    Returns a single iterator that repeatedly cycles over all the
    iterators in iters.

    >>> from itertools import count, islice
    >>> def take(iter_, n): return tuple(islice(iter_, n))
    ... 
    >>> from random import randint
    >>> def randomprefix(iter_): return take(iter_, randint(3, 8))
    ... 
    >>> orig = tuple(count(1000 * i) for i in range(1, 3 + 1))
    >>> muxed = mux(orig)
    >>> take(muxed, 10)
    (1000, 2000, 3000, 1001, 2001, 3001, 1002, 2002, 3002, 1003)
    >>> demuxed = demux(muxed, 5)
    >>> print '\n'.join(map(str, map(randomprefix, demuxed)))
    (2003, 3003, 1004)
    (2003, 3003, 1004)
    (2003, 3003, 1004, 2004, 3004, 1005, 2005)
    (2003, 3003, 1004, 2004, 3004, 1005, 2005, 3005)
    (2003, 3003, 1004, 2004, 3004, 1005, 2005)
    >>> muxed = mux(demuxed)
    >>> take(muxed, 10)
    (2004, 2004, 3005, 1006, 3005, 3004, 3004, 1006, 2006, 1006)
    >>> demuxed = demux(muxed, 4)
    >>> print '\n'.join(map(str, map(randomprefix, demuxed)))
    (1005, 1005, 2006, 3006, 2006, 2005, 2005)
    (1005, 1005, 2006, 3006, 2006, 2005)
    (1005, 1005, 2006)
    (1005, 1005, 2006)
    """
    while True:
        for i in iters:
            yield next(i)


# NOTE: demux is identical to itertools.tee; provided here just for
# symetry with mux.
demux = tee

def mergeiters(iters):
    """
    Make an iterator of sequences from a sequence of iterators.

    iters is a (finite) sequence of iterators

    Returns a single iterator that produces sequences of length
    len(iters), obtained from sequentially polling (once per
    iteration) all the iterators in iters.

    mergeiters is a lightweight wrapper for itertools.izip.  All it
    does is to change izip's signature slightly (mergeiters(iters) is
    identical to izip(*iters)).
    """
    return izip(*iters)


def splititer(seqiter, n):
    """
    Split sequence iterator seqiter into n iterators.

    seqiter is an iterator that returns 'sequences' (i.e. any
    container, such as a list or tuple, that can be addressed with a
    numerical index) of length n or greater.

    n must be either a sequence of integers or a non-negative integer
    (the cases n=0 and n=1 are pointless but permissible).  The latter
    alternative is equivalent to passing range(n) as the second
    argument.  For this reason, in what follows I will assume that n
    is a sequence of integers, without loss of generality.

    Returns a tuple of r== len(n) independent iterators, representing
    positions n[0], n[1], ..., n[r] along the sequences returned by
    seqiter.  (Hence, the integers in the n parameter must all be
    valid indices for the sequences produced by seqiter.)

    WARNING: once an iterator is passed as argument to splititer, it
    is no longer safe to use it independently.

    Functions splititer and mergeiters may be regarded as inverses of
    each other (as demonstrated by the mergesplit_demo function
    below), as long as one keeps in mind the proviso written in the
    previous paragraph.

    def mergesplit_demo(n, times, _minpreflen=3, _maxpreflen=8):
        from itertools import count, islice
        from random import randint

        # first, some helper functions
        def _take(iter_, n):
            return tuple(islice(iter_, n))

        def _randomprefix(iter_):
            return _take(iter_, randint(_minpreflen, _maxpreflen))

        def _printmerged(merged, hdng, _pfx='  '):
            print
            print '\n  '.join([hdng] + map(str, _take(merged, _minpreflen)))

        def _printsplit(split, hdng):
            print
            print '\n  '.join([hdng] + map(str, map(_randomprefix, split)))

        split = tuple(count(1000 * i) for i in range(1, n + 1))
        for i in range(times):
            if i == 0:
                _printsplit(split, 'original %d iterators '
                              '(random-length prefixes):' % n)

            merged = mergeiters(split)
            _printmerged(merged, 'first %d items after '
                        'merging %d time(s):' % (_minpreflen, (i + 1)))

            split = splititer(merged, n)
            _printsplit(split, 'random-length prefixes after '
                          'splitting %d time(s):' % (i + 1))

    >>> # showtime!
    >>> mergesplit_demo(3, 3)

    original 3 iterators (random-length prefixes):
        (1000, 1001, 1002, 1003, 1004, 1005)
        (2000, 2001, 2002, 2003, 2004, 2005, 2006)
        (3000, 3001, 3002, 3003, 3004, 3005, 3006)

    first 3 items after merging 1 time(s):
        (1006, 2007, 3007)
        (1007, 2008, 3008)
        (1008, 2009, 3009)

    random-length prefixes after splitting 1 time(s):
        (1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016)
        (2010, 2011, 2012)
        (3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017)

    first 3 items after merging 2 time(s):
        (1017, 2013, 3018)
        (1018, 2014, 3019)
        (1019, 2015, 3020)

    random-length prefixes after splitting 2 time(s):
        (1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027)
        (2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023)
        (3021, 3022, 3023, 3024, 3025, 3026, 3027)

    first 3 items after merging 3 time(s):
        (1028, 2024, 3028)
        (1029, 2025, 3029)
        (1030, 2026, 3030)

    random-length prefixes after splitting 3 time(s):
        (1031, 1032, 1033, 1034)
        (2027, 2028, 2029, 2030, 2031, 2032, 2033)
        (3031, 3032, 3033, 3034, 3035)
    """

    if hasattr(n, '__iter__'):
        n = tuple(n)
    else:
        n = range(n)
    r = len(n)
    n_seqiter_clones = tee(seqiter, r)
    return tuple([_ith(clone, i) for i, clone in zip(n, n_seqiter_clones)])
                  

def _ith(seqiter, i):
    """
    Produce the 'i-th projection iterator' from an iterator of sequences.

    See the splititer's documentation for a description of the seqiter
    parameter.

    i should be a valid index for the sequences generated by seqiter.

    Returns a generator that produces the ith-component of each
    sequence produced by seqiter.
    """
    return (seq[i] for seq in seqiter)
