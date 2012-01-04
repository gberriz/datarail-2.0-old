from itertools import repeat
from functools import partial

def comp(fns, x):
    """
    Evaluate the composition of single-argument functions at x.

    comp((f0, f1, f2, f3), x) ==> f3(f2(f1(f0(x))))

    Note that component functions are applied in the order they appear
    in fns.

    All the component functions must be callable with a single
    argument, and must return a value that is a valid argument for the
    next function in fns.
    
    comp((), x) ==> x

    comp((f, f, f), x) will produce the same output as nest(f, 3, x)
    """
    return _comp(reversed(fns), x)


def _comp(fns, x):
    try:
        return next(fns)(_comp(fns, x))
    except StopIteration:
        return x


# ccomp == curried comp
def ccomp(*fns):
    return partial(comp, fns)


def complist(fns, x):
    return _complist(reversed(fns), x)


def _complist(fns, x):
    try:
        lastfn = next(fns)
    except StopIteration:
        return (x,)
    head = _complist(fns, x)
    return head + (lastfn(head[-1]),)


def ccomplist(*fns):
    return partial(complist, fns)


def nest(func, n, init):
    return _comp(repeat(func, n), init)

def cnest(func, n):
    return partial(nest, func, n)


def nestlist(func, n, init):
    return _complist(repeat(func, n), init)


def cnestlist(func, n):
    return partial(nestlist, func, n)


def reducelist(function, sequence, *init):
    """
    >>> reduce(lambda x, y: sum((x, y)), iter(range(11)))
    55
    >>> fn.reducelist(lambda x, y: x + y, iter(range(11)))
    (0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55)

    The first argument, function, must be callable with two arguments.
    Therefore, the following will not work:

    >>> reducelist(sum, iter(range(11)))
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: 'int' object is not iterable

    This restriction is not serious, since it can always be remedied
    with a lambda expression, as shown above, or more generally:
    
    >>> fn.reducelist(lambda x, y: takes_a_sequence((x, y)),
    ...               iter(range(11)))
    """
    ret = []
    ls = len(init)
    if ls > 1:
        raise TypeError, 'reducelist() takes 2 or 3 arguments ' \
                         '(%d given)' % ls + 2
    if ls:
        init0 = init[0]
        ret.append(init0)
        sequence = (init0,) + tuple(sequence)
    else:
        sequence = iter(sequence)
        ret.append(next(sequence))

    for g in sequence:
        cur = ret[-1]
        ret.append(function(cur, g))

    return tuple(ret)
    

if __name__ == '__main__':
    from string import lowercase as atoz

    sq = lambda x: x*x
    log2 = lambda x: 0 if x == 1 else 1 + log2(x>>1)

    huge = nest(sq, 8, 2)
    print huge
    print nestlist(log2, 3, huge)[1:]

    squarer = cnestlist(sq, 7)
    print squarer(2)

    frobozz = ccomp(list, len, sq)
    print frobozz(atoz)
    lfrobozz = ccomplist(list, len, sq)
    print lfrobozz(atoz)

    # from multikeydict import MultiKeyDict
    # from itertools import product
    # mkd = MultiKeyDict()
    # for tupl in product(atoz, repeat=3):
    #     mkd.set(tupl, ''.join(tupl))

    # tlb = lambda x: next(x.itervalues())
    # walker = cnestlist(tlb, 2)
    # print map(ccomp(dict.keys, len), walker(mkd))
