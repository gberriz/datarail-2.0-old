# adapted from http://code.activestate.com/recipes/576694-orderedset on
# 110302W

# TODO: implement OrderedSet.index
import re

import traceback as tb
from pdb import set_trace as ST

import collections as co

try:
    from thread import get_ident
except ImportError:
    from dummy_thread import get_ident

def _recursive_repr(dots=lambda s: '%s(...)' % type(s).__name__):
    if not callable(dots):
        dots = lambda s: dots

    def _decorator(user_function):    
        'Decorator to make a repr function return "..." for a recursive call'
        repr_running = set()

        def wrapper(self):
            key = id(self), get_ident()
            if key in repr_running:
                return dots(self)
            repr_running.add(key)
            try:
                result = user_function(self)
            finally:
                repr_running.discard(key)
            return result

        # Can't use functools.wraps() here because of bootstrap issues
        wrapper.__module__ = getattr(user_function, '__module__')
        wrapper.__doc__ = getattr(user_function, '__doc__')
        wrapper.__name__ = getattr(user_function, '__name__')
        return wrapper

    return _decorator


KEY, PREV, NEXT = range(3)

class OrderedSet(co.Set):
    def __init__(self, iterable=None):
        self._debug = False
        self._reset([], {})
        if iterable is not None:
            self.update(iterable)

    def add(self, key):
        """Add an element."""

        if key not in self.mapping:
            end = self.end
            curr = end[PREV]
            curr[NEXT] = end[PREV] = self.mapping[key] = [key, curr, end]

    def clear(self):
        if not self: return

        for curr in self.__baseiter():
            curr[:] = []

        assert self.end == [None, [], []]
        self._reset()

    def copy(self):
        return type(self)(self)

    def difference(self, *other):
        return type(self)(filter(lambda x: x not in set().union(*other), self))

    def difference_update(self, *args):
        for other in args:
            for v in other:
                self.discard(v)

    def discard(self, key):
        """Remove an element.  Do not raise an exception if absent."""

        try:
            self.remove(key)
        except KeyError:
            pass

    def intersection(self, *args):
        l = len(args)
        other = (() if l == 0 else
                 set(args[0]).intersection(*args[1:]))
        return type(self)(filter(lambda x: x in other, self))

    def intersection_update(self, *args):
        l = len(args)
        other = (() if l == 0 else
                 set(args[0]).intersection(*args[1:]))
        for v in self:
            if not v in other:
                self.discard(v)

    def isdisjoint(self, other):
        short, long_ = (self, other) if len(self) < len(other) else (other, self)
        for item in short:
            if item in long_:
                return False
        return True

    def issubset(self, other):
        return all([o in other for o in self])

    def issuperset(self, other):
        return all([o in self for o in other])

    def first(self):
        self._assert_not_empty()
        return self.end[NEXT][KEY]

    def last(self):
        self._assert_not_empty()
        return self.end[PREV][KEY]

    def pop(self, last=True):
        """Return the popped value.  Raise KeyError if empty."""

        self._assert_not_empty(KeyError)
        key = next(reversed(self)) if last else next(iter(self))
        self.remove(key)
        return key

    def remove(self, key):
        """Remove an element. If not a member, raise a KeyError."""

        self._assert_not_empty(KeyError, key)
        origkey, key = key, _maybefreeze(key)
        try:
            popped = self.mapping.pop(key)
        except KeyError:
            raise KeyError(origkey)

        prev, n_xt = popped[PREV], popped[NEXT]
        popped[:] = []
        # bye-bye popped...
        n_xt[PREV], prev[NEXT] = prev, n_xt

    def symmetric_difference(self, other):
        tmp = (other if hasattr(other, 'difference') else
               type(self)(other)).difference(self)
        return self.difference(other).union(tmp)

    def symmetric_difference_update(self, other):
        tmp = (other if hasattr(other, 'difference') else
               type(self)(other)).difference(self)
        self.difference_update(other)
        self.update(tmp)

    def union(self, *others):
        ret = self.copy()
        ret.update(*others)
        return ret

    def update(self, *others):
        for other in others:
            for item in other:
                self.add(item)

    # the NEXT keyword avoids an exception that may occur
    # *occasionally* at shutdown if the call to __del__ results in a
    # call to __baseiter (this is a case of "defensive programming",
    # since __baseiter is not invoked in the current implementation
    # of OrderedSet)
    def __baseiter(self, NEXT=NEXT):
        end = self.end
        curr = end[NEXT]
        while curr is not end:
            n_xt = curr[NEXT]
            yield curr
            curr = n_xt

    # re PREV keyword: see comment before definition of __baseiter.
    def __basereviter(self, PREV=PREV):
        end = self.end
        curr = end[PREV]
        while curr is not end:
            prev = curr[PREV]
            yield curr
            curr = prev

    def __cmp__(self, other, _subre=re.compile(r'\bset(s?\b)')):
        cl = type(self)
        cn = cl.__name__
        try:
            return set.__cmp__(set(self),
                               set(other) if isinstance(other, cl)
                               else other)
        except Exception, e:
            msg = re.sub(_subre, cn + r'\1', str(e))
            raise type(e), msg

    def __contains__(self, key):
        return _maybefreeze(key) in self.mapping

    def __del__(self):
        self.clear()                    # remove circular references

    def __eq__(self, other):
        if len(self) != len(other): return False
        if isinstance(other, OrderedSet):
            return list(self) == list(other)
        else:
            return hasattr(other, '__iter__') and set(self) == set(other)

    # re KEY keyword: see comment before definition of __baseiter.
    def __iter__(self, KEY=KEY):
        for curr in self.__baseiter():
            yield curr[KEY]

    def __len__(self):
        return len(self.mapping)

    def __reduce_ex__(self, proto):
        # FIXME: don't ignore proto
        state = {}
        stdkeys = _stdkeys()
        for k, v in self.__dict__.items():
            if k not in stdkeys:
                state[k] = v
        return (_unpickle, (list(self),), state)

    @_recursive_repr()
    def __repr__(self):
        if not self:
            return '%s()' % (type(self).__name__,)
        return '%s(%r)' % (type(self).__name__, list(self))


    def __reversed__(self):
        for curr in self.__basereviter():
            yield curr[KEY]

    def __difference(self, other, update=False):
        _assert_supported_operand_types('-' if not update else '-=',
                                        self, other)
        if update:
            self.difference_update(other)
            return self
        else:
            return self.difference(other)

    def __intersection(self, other, update=False, flip=False):
        assert not (flip and update)
        op = '&' if not update else '&='
        args = (self, other) if not flip else (other, self)
        _assert_supported_operand_types(op, *args)
        del op, args

        if update:
            self.intersection_update(other)
            return self
        elif flip:
            return type(self)(other).intersection(self)
        else:
            return self.intersection(other)

    def __symmetric_difference(self, other, update=False, flip=False):
        assert not (flip and update)
        op = '^' if not update else '^='
        args = (self, other) if not flip else (other, self)
        _assert_supported_operand_types(op, *args)
        del op, args

        if update:
            self.symmetric_difference_update(other)
            return self
        elif flip:
            return type(self)(other).symmetric_difference(self)
        else:
            return self.symmetric_difference(other)

    def __union(self, other, update=False, flip=False, op='|'):
        if update:
            assert not flip
            op += '='
        args = (self, other) if not flip else (other, self)
        _assert_supported_operand_types(op, *args)
        del op, args

        if update:
            self.update(other)
            return self
        elif flip:
            return type(self)(other).union(self)
        else:
            return self.union(other)

    def __add__(self, other):
        # non-standard!
        return self.__union(other, op='+')

    def __and__(self, other):
        return self.__intersection(other)


    def __iadd__(self, other):
        return self.__union(other, update=True, op='+')

    def __iand__(self, other):
        return self.__intersection(other, update=True)

    def __ior__(self, other):
        return self.__union(other, update=True)

    def __isub__(self, other):
        return self.__difference(other, update=True)

    def __ixor__(self, other):
        return self.__symmetric_difference(other, update=True)


    def __or__(self, other):
        return self.__union(other)

    def __radd__(self, other):
        # non-standard!
        return self.__union(other, flip=True, op='+')

    def __rand__(self, other):
        return self.__intersection(other, flip=True)

    def __ror__(self, other):
        return self.__union(other, flip=True)

    def __rxor__(self, other):
        return self.__symmetric_difference(other, flip=True)

    def __sub__(self, other):
        return self.__difference(other)

    def __xor__(self, other):
        return self.__symmetric_difference(other)


    def _reset(self, end=None, mapping=None):
        if end is None: end = self.end
        else: self.end = end

        # sentinel node for doubly linked list
        end[:] = [None, end, end]

        # key --> [key, prev, n_xt]
        if mapping is None: mapping = self.mapping
        else: self.mapping = mapping

        mapping.clear()

    def _assert_not_empty(self, _exctype=ValueError, _msg='set is empty'):
        if not self: raise _exctype, _msg


def _unpickle(iterable):
    return OrderedSet(iterable)

def _stdkeys(_ret=OrderedSet().__dict__.keys()):
    return _ret

def _assert_supported_operand_types(op, *operands):
    assert 0 < len(operands) < 3
    if all(hasattr(o, '__iter__') for o in operands):
        return True
    types = ' and '.join("'%s'" % type(o).__name__ for o in operands)
    raise TypeError, 'unsupported operand type(s) for %s: %s' % (op, types)


def _maybefreeze(key):
    # to match the special-casing of set-type arguments by
    # set.discard, set.remove, and set.__contains__; see
    # http://docs.python.org/library/stdtypes.html#set-types-set-frozenset
    return key if getattr(key, '__hash__', None) or \
                  not isinstance(key, (set, OrderedSet)) \
               else frozenset(key)


if __name__ == '__main__':
    # print(OrderedSet('abracadaba'))
    # print(OrderedSet('simsalabim'))
    class SC(OrderedSet):
        pass

    word = 'simsalabim'
    otherword = 'madagascar'
    s = SC(word)
    o = tuple(otherword)

    if True:
        ST()
        k = o ^ s
        for c in (word + otherword):
            if (c in word) ^ (c in otherword):
                assert c in k
            else:
                assert c not in k
