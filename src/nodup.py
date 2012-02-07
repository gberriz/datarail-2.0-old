_NODUPERR = TypeError('Duplicates are not allowed')


class NoDup(list):

    def __new__(cls, seq=[]):
        return list.__new__(cls)


    # mutators

    def __init__(self, seq=[]):
        """
        >>> NoDup([1, 2, 2, 1])
        NoDup([1, 2])
        """
        self.__set = set()
        self.extend(seq)


    def append(self, item):
        """
        >>> x = NoDup([1, 2]); x.append(1); x == NoDup([1, 2])
        True
        """

        if not item in self.__set:
            super(NoDup, self).append(item)
            self.__set.add(item)
        # self._consistency_check()


    def __setitem__(self, i, item):
        """
        >>> x = NoDup([1, 2])
        >>> x[1] = 3; x
        NoDup([1, 3])
        >>> x[1] = 1
        Traceback (most recent call last):
        ...
        TypeError: Duplicates are not allowed
        >>> x[0] = 1; x
        NoDup([1, 3])
        """

        # FIXME: implement support for the case where i is a SLICE

        if item in self.__set:
            if item != self[i]: raise _NODUPERR
        else:
            # goners = self[i] if isinstance(i, slice) else [self[i]]
            goner = self[i]
            super(NoDup, self).__setitem__(i, item)
            s = self.__set
            s.remove(goner)
            s.add(item)
        # self._consistency_check()


    def __delitem__(self, i):
        """
        >>> x = NoDup([1, 2, 3])
        >>> del x[1]; x == NoDup([1, 3])
        True
        """

        goners = self[i] if isinstance(i, slice) else [self[i]]
        super(NoDup, self).__delitem__(i)
        for g in goners:
            self.__set.remove(g)
        # self._consistency_check()


    def insert(self, i, item):
        """
        >>> x = NoDup([1, 2])
        >>> x.insert(1, 0); x
        NoDup([1, 0, 2])
        >>> x.insert(2, 1)
        Traceback (most recent call last):
        ...
        TypeError: Duplicates are not allowed
        >>> x._NoDup__set
        set([0, 1, 2])
        """

        if item in self.__set: raise _NODUPERR
        super(NoDup, self).insert(i, item)
        self.__set.add(item)
        # self._consistency_check()


    def extend(self, other):
        """
	>>> x = NoDup([1, 2, 3]); x.extend([5, 3, 2, 5, 4, 1]); x
	NoDup([1, 2, 3, 5, 4])
        """

        for o in other: self.append(o)
        # self._consistency_check()


    def remove(self, item):
        """
        >>> x = NoDup([1, 2, 3]); x.remove(2); x
        NoDup([1, 3])
        """

        super(NoDup, self).remove(item)
        self.__set.remove(item)
        # self._consistency_check()


    def __setslice__(self, i, j, other):
        """
        >>> x = NoDup(range(10)); x
        NoDup([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> x[4:8] = [10, 11]; x
        NoDup([0, 1, 2, 3, 10, 11, 8, 9])
        >>> x[3:6] = [0]
        Traceback (most recent call last):
        ...
        TypeError: Duplicates are not allowed
        """

        s = self.__set
        d = set(self[i:j])
        if (s - d).intersection(set(other)): raise _NODUPERR
        super(NoDup, self).__setslice__(i, j, other)
        s.difference_update(d)
        s.update(other)
        # self._consistency_check()


    def __delslice__(self, i, j):
        """
        >>> x = NoDup(range(10))
        >>> del x[4:8]; x
        NoDup([0, 1, 2, 3, 8, 9])
        """

        print '__delslice__ called'
        self.__set.difference_update(self[i:j])
        super(NoDup, self).__delslice__(i, j)
        # self._consistency_check()


    def pop(self, i=-1):
        """
	>>> x = NoDup([1, 2, 3]); x.pop()
	3
	>>> x = NoDup([1, 2, 3]); x.pop(1)
	2
        >>> x._NoDup__set
        set([1, 3])
        """

        ret = super(NoDup, self).pop(i)
        self.__set.remove(ret)
        # self._consistency_check()
        return ret


    def __iadd__(self, other):
        """
        >>> x = NoDup([1, 2, 3]); x += [2, 5, 3, 4]; x
        NoDup([1, 2, 3, 5, 4])
        """

        self.extend(other)
        # self._consistency_check()
        return self


    def __imul__(self, n):
        """
        >>> x = NoDup([1, 2]); x *= 0; x
        NoDup([])
        >>> x = NoDup([1, 2]); x *= 1; x
        NoDup([1, 2])
        >>> x = NoDup([1, 2]); x *= 2; x
        Traceback (most recent call last):
        ...
        TypeError: Duplicates are not allowed
        """

        if n > 1: raise _NODUPERR
        if n < 1:
            super(NoDup, self).__imul__(n)
            # self[:] = []
            self.__set.clear()
        # self._consistency_check()
        return self


    # read-only methods

    def __contains__(self, item):
        """
        >>> x = NoDup([1, 2]); 1 in x; 3 in x
        True
        False
        """

        return self and item in self.__set


    def __add__(self, other):
        """
        >>> NoDup([1, 2, 3]) + NoDup([3, 4, 2, 4]) == \\
        ... NoDup([1, 2, 3] + [3, 4, 2, 4])
        True
        >>> NoDup([1, 2, 3]) + NoDup([3, 4, 2, 4])
        NoDup([1, 2, 3, 4])
        """

        cls = type(self)
        o = other if isinstance(other, cls) else cls(other)
        return cls(super(NoDup, self).__add__(o))


    # def __radd__(self, other):
    #     """
    #     >>> [1, 2, 3] + NoDup([3, 4, 2, 4])
    #     NoDup([1, 2, 3, 4])
    #     """

    #     return self.__class__(super(NoDup, self).__radd__(other))


    def __mul__(self, n):
        """
        >>> 0 * NoDup([1, 2])
        NoDup([])
        >>> 1 * NoDup([1, 2])
        NoDup([1, 2])
        >>> 2 * NoDup([1, 2])
        Traceback (most recent call last):
        ...
        TypeError: Duplicates are not allowed
        """

        if n > 1: raise _NODUPERR
        return self.__class__(super(NoDup, self).__mul__(n))


    __rmul__ = __mul__


    def __getslice__(self, i, j):
        """
        >>> NoDup(range(10))[2:7]
        NoDup([2, 3, 4, 5, 6])
        """

        return self.__class__(super(NoDup, self).__getslice__(i, j))


    def __str__(self):
        """
        >>> str(NoDup([1, 2]))
        '[1, 2]'
        >>> '%s' % NoDup([1, 2])
        '[1, 2]'
        """

        # as strange as it may look, NoDup.__str__ uses list.__repr__
        # while NoDup.__repr__ uses NoDup.__str__ (which in turn uses
        # list.__repr__); IOW, list.__str__ is never used here,
        # neither directly nor indirectly; this is because (as far as
        # I can tell) list.__str__ calls its argument's __repr__
        # method, and therefore it is necessary to avoid it if one
        # wants NoDup.__str__ and NoDup.__repr__ to return different
        # strings.
        return super(NoDup, self).__repr__()
        

    def __repr__(self):
        """
        >>> repr(NoDup([1, 2]))
        'NoDup([1, 2])'
        >>> '%r' % NoDup([1, 2])
        'NoDup([1, 2])'
        """

        # SEE COMMENTS in __str__ above
        return '%s(%s)' % (self.__class__.__name__, self)


"""

    def __eq__(self, other):
        print '__eq__ called'
        return super(NoDup, self).__eq__(other)
        # return self.data == self.__cast(other)

    def __cast(self, other):
        return other
    def __lt__(self, other):
        print '__lt__ called'
        return list(self) <  self.__cast(other)
    def __le__(self, other):
        print '__le__ called'
        return list(self) <= self.__cast(other)
    def __eq__(self, other):
        print '__eq__ called'
        return list(self) == self.__cast(other)
    def __ne__(self, other):
        print '__ne__ called'
        return list(self) != self.__cast(other)
    def __gt__(self, other):
        print '__gt__ called'
        return list(self) >  self.__cast(other)
    def __ge__(self, other):
        print '__ge__ called'
        return list(self) >= self.__cast(other)
    def __cmp__(self, other):
        print '__cmp__ called'
        return cmp(list(self), self.__cast(other))


    # def __cast(self, other):
    #     if isinstance(other, ): return other.data
    #     else: return other

    # def _consistency_check(self):
    #     assert set(self) == self.__set

    # __hash__ = None # Mutable sequence, so not hashable
    __hash__ = 0
"""


def _test():
    # class AllEq:
    #     def __eq__(self, other):
    #         return True
    #     __hash__ = None

    # print AllEq() in []
    # print AllEq() in NoDup()
    # print AllEq() in [1]
    # print AllEq() in NoDup([1])

    a = NoDup([0,1,2,3,4])
    del a[::2]
