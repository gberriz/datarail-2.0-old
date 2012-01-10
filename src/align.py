def align(s0, s1, _blank=None):
    """
    >>> align(['d', 'g', 'e', 'f', 'c'], ['e', 'd', 'i', 'g', 'a', 'f'])
    (['d', 'g', None, 'e', None, 'f', 'c'], ['d', 'g', 'i', 'e', 'a', 'f', None])

    in:
      ['d', 'g', 'e', 'f', 'c']
      ['e', 'd', 'i', 'g', 'a', 'f']
    out:
      ['d', 'g', None, 'e', None, 'f', 'c']
      ['d', 'g', 'i', 'e', 'a', 'f', None]

    in:
      ['z', 'd', 'g', 'e', 'f', 'c', 'j']
      ['e', 'd', 'i', 'g', 'a', 'f']
    out:
      ['z', 'd', 'g', None, 'e', None, 'f', 'c', 'j']
      [None, 'd', 'g', 'i', 'e', 'a', 'f', None, None]
    """

    ss0 = set(s0)
    if len(ss0) < len(s0) or len(set(s1)) < len(s1):
        raise TypeError, 'some argument has repeated values'

    order = dict((v, i) for i, v in enumerate(s0))
    p = filter(lambda ix: ix[1] in order, enumerate(s1))
    q = [pr[0] for pr in sorted(p, key=lambda iv: order[iv[1]])]
    r = range(len(s1))
    for i, pr in enumerate(p):
        r[pr[0]] = q[i]
    t0 = s0[:]
    t1 = [s1[i] for i in r]
    del p, q, r

    u0, u1 = [], []
    i = j = 0
    lim0, lim1 = len(t0), len(t1)
    while i < lim0 or j < lim1:
        v = t0[i] if i < lim0 else None
        w = t1[j] if j < lim1 else None
        di = dj = 1
        if v != w:
          if w not in ss0 and j < lim1:
              v = _blank
              di = 0
          else:
              w = _blank
              dj = 0

        i += di
        j += dj
        u0.append(v)
        u1.append(w)

    return u0, u1


if __name__ == '__main__':
    import random as rn
    s0 = list('cdefg')
    seed = 1
    rn.seed(seed)
    rn.shuffle(s0)
    s1 = rn.sample(list('abcdefghi'), 6)
    print 'in:'
    print '  ', s0
    print '  ', s1
    print
    print 'out:'
    for s in align(s0, s1):
        print '  ', s
