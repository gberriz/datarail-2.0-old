from pdb import set_trace as ST

def align(s0, s1, key=None, pad=None):
    """
    >>> align(('d', 'g', 'e', 'f', 'c'), ('e', 'd', 'i', 'g', 'a', 'f'))
    (('d', 'g', None, 'e', None, 'f', 'c'), ('d', 'g', 'i', 'e', 'a', 'f', None))

    in:
      ('d', 'g', 'e', 'f', 'c')
      ('e', 'd', 'i', 'g', 'a', 'f')
    out:
      ('d', 'g', None, 'e', None, 'f', 'c')
      ('d', 'g', 'i', 'e', 'a', 'f', None)

    in:
      ('z', 'd', 'g', 'e', 'f', 'c', 'j')
      ('e', 'd', 'i', 'g', 'a', 'f')
    out:
      ('z', 'd', 'g', None, 'e', None, 'f', 'c', 'j')
      (None, 'd', 'g', 'i', 'e', 'a', 'f', None, None)


    With pad = lambda d: (d[0], (None,)) and key = lambda t: t[0]
    
    in:
      (('D', ('a',)), ('G', ('b', 'c', 'd', 'e')), ('E', ('f', 'g', 'h',
       'i', 'j')), ('F', ('k',)), ('C', ('l', 'm', 'n', 'o', 'p')))
      (('E', ('q',)), ('D', ('r',)), ('I', ('s',)), ('G', ('t', 'u',
       'v')), ('A', ('w',)), ('F', ('x', 'y', 'z')))

    out:
      (('D', ('a',)), ('G', ('b', 'c', 'd', 'e')), ('I', (None,)), ('E',
       ('f', 'g', 'h', 'i', 'j')), ('A', (None,)), ('F', ('k',)), ('C',
       ('l', 'm', 'n', 'o', 'p')))
      (('D', ('r',)), ('G', ('t', 'u', 'v')), ('I', ('s',)), ('E',
       ('q',)), ('A', ('w',)), ('F', ('x', 'y', 'z')), ('C', (None,)))


    With pad = lambda d: (d[0], 1) and key = lambda t: t[0]

    in:
      (('D', 1), ('G', 4), ('E', 5), ('F', 1), ('C', 5))
      (('E', 1), ('D', 1), ('I', 1), ('G', 3), ('A', 1), ('F', 3))
    
    out:
      (('D', 1), ('G', 4), ('I', 1), ('E', 5), ('A', 1), ('F', 1), ('C',
       5))
      (('D', 1), ('G', 3), ('I', 1), ('E', 1), ('A', 1), ('F', 3), ('C',
       1))
    """

    if key is None:
        m0, m1 = s0, s1
    else:
        m0, m1 = map(key, s0), map(key, s1)

    mm0 = set(m0)
    if len(mm0) < len(m0) or len(set(m1)) < len(m1):
        raise TypeError, 'some argument has repeated values'

    order = dict((v, i) for i, v in enumerate(m0))

    p = filter(lambda ix: ix[1] in order, enumerate(m1))
    q = [pr[0] for pr in sorted(p, key=lambda iv: order[iv[1]])]
    r = range(len(m1))
    for i, pr in enumerate(p):
        r[pr[0]] = q[i]

    t0 = zip(m0, s0)
    t1 = [(m1[i], s1[i]) for i in r]

    del p, q, r

    if not callable(pad):
        blank = pad
        pad = lambda x: blank

    u0, u1 = [], []
    i = j = 0
    lim0, lim1 = len(t0), len(t1)
    null = object()

    while i < lim0 or j < lim1:
        vv, v = t0[i] if i < lim0 else (null, pad(t1[j][1]))
        ww, w = t1[j] if j < lim1 else (null, pad(t0[i][1]))

        di = dj = 1
        if vv != ww:
          if ww not in mm0 and j < lim1:
              v = pad(w)
              di = 0
          else:
              w = pad(v)
              dj = 0

        i += di
        j += dj
        u0.append(v)
        u1.append(w)

    return tuple(u0), tuple(u1)


if __name__ == '__main__':
    import random as rn
    import string as st

    s0 = list('CDEFG')
    seed = 1
    rn.seed(seed)
    rn.shuffle(s0)
    s1 = rn.sample(list('ABCDEFGHI'), 6)
    print 'in:'
    print '  ', s0
    print '  ', s1
    print
    print 'out:'
    for s in align(s0, s1):
        print '  ', s

    print '\n'

    ns0, ns1 = len(s0), len(s1)

    lc = list(st.ascii_lowercase)
    nlc = len(lc)
    import numpy as np
    h = np.array([0] +
                 [1 + x for x in
                  sorted(rn.sample(range(nlc), ns0 + ns1 - 1))] +
                 [nlc])

    levels = [tuple(lc[start:stop]) for start, stop in zip(h[:-1], h[1:])]
    di0, di1 = zip(s0, levels[:ns0]), zip(s1, levels[ns0:])

    import textwrap as tw

    print 'in:'
    for di in di0, di1:
        print tw.fill(str(di), initial_indent='  ', subsequent_indent='   ')

    nulldim = lambda d: (d[0], (None,))
    first = lambda t: t[0]

    print
    print 'out:'
    for di in align(di0, di1, key=first, pad=nulldim):
        print tw.fill(str(di), initial_indent='  ', subsequent_indent='   ')


    sh0, sh1 = [map(lambda d: (d[0], len(d[1])), di) for di in di0, di1]

    print 'in:'
    for sh in sh0, sh1:
        print tw.fill(str(sh), initial_indent='  ', subsequent_indent='   ')

    nulldim = lambda d: (d[0], 1)
    first = lambda t: t[0]

    print
    print 'out:'
    for sh in align(sh0, sh1, key=first, pad=nulldim):
        print tw.fill(str(sh), initial_indent='  ', subsequent_indent='   ')
