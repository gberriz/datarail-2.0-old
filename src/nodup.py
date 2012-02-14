from pdb import set_trace as ST

_NODUPERR = TypeError('Duplicates are not allowed')

class NoDup(list):

    # mutators

    def __init__(self, seq=[]):
        """
        >>> NoDup([1, 2, 2, 1])
        NoDup([1, 2])
        """

        super(NoDup, self).__setitem__(slice(0, len(self)), ())
        self.__set = set()
        self.extend(seq)
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
        >>> x = NoDup(range(5))
        >>> x[1:3] = [5, 6, 7]; x
        NoDup([0, 5, 6, 7, 3, 4])
        >>> x[1:3] = [8, 8]
        Traceback (most recent call last):
        ...
        TypeError: Duplicates are not allowed
        >>> x[1:3] = [0, 8]
        Traceback (most recent call last):
        ...
        TypeError: Duplicates are not allowed
        >>> x
        NoDup([0, 5, 6, 7, 3, 4])
        """

        selfset = self.__set

        if isinstance(i, slice):
            newitems = set(item)
            goners = set(self[i])

            # the first test below, namely,
            #
            #   len(newitems) < len(item)
            #
            # detects duplicates in item; the second one, namely,
            # whether the set
            #
            # (selfset - goners).intersection(newitems)
            #
            # is null, detects overlaps (i.e. a non-null intersection)
            # between the elements in item (now in the set newitem)
            # and the remaining elements of self (i.e. those that
            # remain after removing from the current elements in self,
            # as given in selfset, those elements that currently
            # occupy the positions specified by the slice i, which now
            # are also those in the goners set; a true value for
            # either test represents a violation of the uniqueness
            # constraint, so an exception is raised.

            if ((len(newitems) < len(item)) or
                (selfset - goners).intersection(newitems)):
                raise _NODUPERR
            super(NoDup, self).__setitem__(i, item)
            selfset.difference_update(goners)
            selfset.update(newitems)
        else:
            if item in self and item != self[i]:
                raise _NODUPERR
            goner = self[i]
            super(NoDup, self).__setitem__(i, item)
            selfset.remove(goner)
            selfset.add(item)

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


    def __setslice__(self, i, j, other):
        """
        >>> a = NoDup([0, 1]); a[-3:-1] = [3]; a
        NoDup([3, 1])
        >>> a = NoDup([0, 1]); a[-3:-1] = [2, 3]; a
        NoDup([2, 3, 1])
        >>> a = NoDup([0, 1]); a[0:1] = [2, 3]; a
        NoDup([2, 3, 1])
        >>> a = NoDup([0, 1]); a[0:2] = [2, 3]; a
        NoDup([2, 3])
        >>> a = NoDup([0, 1]); a[1:2] = [2, 3]; a
        NoDup([0, 2, 3])
        >>> a = NoDup([0, 1]); a[-1:2] = [2, 3]; a
        NoDup([0, 2, 3])
        >>> a = NoDup([0, 1]); a[-3:2] = [2, 3]; a
        NoDup([2, 3])
        >>> a = NoDup([0, 1]); a[:] = [2, 3]; a
        NoDup([2, 3])
        """

        i = i if i > 0 else 0
        j = j if j > 0 else 0
        self.__setitem__(slice(i, j), other)
        # self._consistency_check()


    def __delslice__(self, i, j):
        """
	>>> a = NoDup([0, 1]); del a[0:1]; a
	NoDup([1])
	>>> a = NoDup([0, 1]); del a[-3:-1]; a
	NoDup([1])
	>>> a = NoDup([0, 1]); del a[0:2]; a
	NoDup([])
	>>> a = NoDup([0, 1]); del a[1:2]; a
	NoDup([0])
	>>> a = NoDup([0, 1]); del a[-1:2]; a
	NoDup([0])
	>>> a = NoDup([0, 1]); del a[-3:2]; a
	NoDup([])
	>>> a = NoDup([0, 1]); del a[:]; a
	NoDup([])
	>>>
        """

        i = i if i > 0 else 0
        j = j if j > 0 else 0
        self.__delitem__(slice(i, j))
        # self._consistency_check()
        

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


    def append(self, item):
        """
        >>> x = NoDup([1, 2]); x.append(1); x == NoDup([1, 2])
        True
        """

        if not item in self:
            super(NoDup, self).append(item)
            self.__set.add(item)

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

        if item in self: raise _NODUPERR
        super(NoDup, self).insert(i, item)
        self.__set.add(item)
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


    def remove(self, item):
        """
        >>> x = NoDup([1, 2, 3]); x.remove(2); x
        NoDup([1, 3])
        """

        super(NoDup, self).remove(item)
        self.__set.remove(item)
        # self._consistency_check()


    def sort(self, *args, **kwargs):
        """
        >>> a = NoDup([2, 0, 1]); a.sort(); a
        NoDup([0, 1, 2])
        >>> ii = [0]
        >>> def evilcmp(x, y, ii=ii):
        ...   a.append(ii[0]); ii[0] += 1; return cmp(x, y)
        ... 
        >>> a = NoDup(['C', 'A', 'B']); a
        NoDup(['C', 'A', 'B'])
        >>> a.sort(evilcmp)
        Traceback (most recent call last):
        ...
        ValueError: list modified during sort
        >>> a
        NoDup(['A', 'B', 'C'])
        """

        # super(NoDup, self).sort does not work perfectly with the
        # rest of the current NoDup implementation; in particular, it
        # fails to raise an exception when it receives a cmp function
        # that (perversely) modifies the list being sorted; the reason
        # for this is that, at least in some cases, super(NoDup,
        # self).sort clears the original list (with "x[:] = []", or
        # something like it), apparently to more easily detect any
        # modification by cmp (or key?) on the list (or attempts
        # thereof); these modifications, whether they are detected
        # (thereby triggering an exception) or not, do not seem to
        # affect the final result; therefore, the only objection to
        # inheriting super(NoDup, self).sort directly is that some
        # perverse/inept, but probably harmless, code would go
        # undetected;

        # super(NoDup, self).sort(*args, **kwargs)

        hold_set, self.__set = self.__set, set()
        try:
            super(NoDup, self).sort(*args, **kwargs)
        finally:
            self.__set = hold_set
        # self._consistency_check()


    def extend(self, other):
        """
        >>> x = NoDup([1, 2, 3]); x.extend([5, 3, 2, 5, 4, 1]); x
        NoDup([1, 2, 3, 5, 4])
        """

        for o in other: self.append(o)
        # self._consistency_check()


    # read-only methods

    def __repr__(self):
        """
        >>> repr(NoDup([1, 2]))
        'NoDup([1, 2])'
        >>> '%r' % NoDup([1, 2])
        'NoDup([1, 2])'
        """

        # SEE COMMENTS in __str__ below
        return '%s(%s)' % (self.__class__.__name__, self)


    def __contains__(self, item):
        """
        >>> x = NoDup([1, 2]); 1 in x; 3 in x
        True
        False
	>>> class hashless:
	...     __hash__ = None
	... 
	>>> hashless() in NoDup([0])
	False
	>>> class hashless(object):
	...     __hash__ = None
	... 
	>>> hashless() in NoDup([0])
	False
        """

        if not self: return False
        try:
            return item in self.__set
        except TypeError, e:
            msg = str(e)
            # attempt to recover item is "unhashable" for some reason
            if ('unhashable' in msg or
                "'NoneType' object is not callable" in msg):
                return super(NoDup, self).__contains__(item)
            raise


    def __getitem__(self, i):
        """
        >>> NoDup([0, 1])[-1]
        1
        >>> NoDup([0, 1])[::-1]
        NoDup([1, 0])
        """

        ret = super(NoDup, self).__getitem__(i)
        return type(self)(ret) if isinstance(i, slice) else ret


    def __getslice__(self, i, j):
        """
        >>> NoDup([0, 1])[:]
        NoDup([0, 1])
        >>> NoDup([0, 1])[0:1]
        NoDup([0])
        >>> NoDup([0, 1])[-3:-1]
        NoDup([0])
        >>> NoDup([0, 1])[0:2]
        NoDup([0, 1])
        >>> NoDup([0, 1])[1:2]
        NoDup([1])
        >>> NoDup([0, 1])[-1:2]
        NoDup([1])
        >>> NoDup([0, 1])[-3:2]
        NoDup([0, 1])
        """

        i = i if i > 0 else 0
        j = j if j > 0 else 0
        return self.__getitem__(slice(i, j))


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
        

    def __copy__(self):
        """
        >>> import copy
        >>> a = NoDup([0, 1]); b = copy.copy(a)
        >>> id(a) != id(b)
        True
        """

        return self[:]



    def __deepcopy__(self, ignored):
        """
        >>> import copy
        >>> a = NoDup([0, 1]); b = copy.deepcopy(a)
        >>> id(a) != id(b)
        True
        """

        # this method emit a warning letting the user know that,
        # despite its name, it returns a shallow copy; a deepcopy does
        # not make sense for a container of immutable objects, like
        # this one; the same thing happens with, e.g., set:
        #
        # >>> x = set(tuple([i]) for i in range(3))
        # >>> x
        # set([(2,), (0,), (1,)])
        # >>> [id(y) for y in x]
        # [140479126058768, 140479126143248, 140479126117136]
        # >>> [id(y) for y in cp.copy(x)]
        # [140479126058768, 140479126143248, 140479126117136]
        # >>> [id(y) for y in cp.deepcopy(x)]
        # [140479126058768, 140479126143248, 140479126117136]

        return self.__copy__()


    # Uncomment to get the test test_getitemoverwriteiter to pass
    # def __iter__(self):
    #     class __nodupiterator(object):
    #         def __init__(self, _nodupobj):
    #             self.__counter = -1
    #             self.__nodupobj = _nodupobj
    #             self.__done = False
    #         def next(self):
    #             if not self.__done:
    #                 c = self.__counter
    #                 self.__counter = c = c + 1
    #                 nd = self.__nodupobj
    #                 if c < len(nd): return nd[c]
    #                 self.__done = True
    #             raise StopIteration
    #         def __iter__(self):
    #             return self
    #     return __nodupiterator(self)


    def _consistency_check(self):
        assert set(self) == self.__set
