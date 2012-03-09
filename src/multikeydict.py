"""
The interaction below shows that a MultiKeyDict takes much less space
than the corresponding dictionary of tuples.

>>> dot = dict()
>>> mkd = MultiKeyDict()
>>> string.uppercase
'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
>>> for tupl in itertools.product(string.uppercase, repeat=3):
...     v = ''.join(tupl)
...     dot[tupl] = v
...     mkd.set(tupl, v)
... 
>>> mkd.get(('M', 'M', 'M'))
'MMM'
>>> dot.get(('M', 'M', 'M'))
'MMM'
>>> len(dot.keys())
17576
>>> len(mkd.keys())
26
>>> len(next(mkd.itervalues()).keys())
26
>>> len(next(next(mkd.itervalues()).itervalues()).keys())
26
>>> sys.getsizeof(dot)
786712
>>> sys.getsizeof(mkd)
3352
"""

from collections import defaultdict
from copy import deepcopy as _deepcopy

from orderedset import OrderedSet

from pdb import set_trace as ST
# INDENT = 0
class MultiKeyDict(defaultdict):
    '''
    Class to implement nested dictionaries of arbitrary depth.

    This class is most useful (over simply using tuples as keys) when
    there are opportunities for saving time by caching
    sub-dictionaries, especially when the composite keys are
    manipulated programmatically.  In contrast, interactive use of
    this class (as in the illustrative examples below) is expected to
    be cumbersome.

    Examples:
    >>> d = MultiKeyDict()
    >>> d.set((1, 2, 3, 4), "x")
    >>> d.get((1, 2, 3, 4))
    'x'
    >>> d.get((1, 2, 3))
    defaultdict(<class 'multikeydict.MultiKeyDict'>, {4: 'x'})
    >>> d.set((1, 2, 3, 4), "y")
    >>> d.get((1, 2, 3, 4))
    'y'
    >>> d.set((1, 2, 3, 40), "z")
    >>> d.get((1, 2, 3, 40))
    'z'

    Some assignments that implicitly would lead to the deletion of
    subtrees are vetoed.  The user must explicitly delete any subtree
    (e.g. using the delete or popmk method) before carrying out the
    assignment.

    >>> d = MultiKeyDict()
    >>> d.set((1, 2, 3, 4), "x")
    >>> d.set((1,), 5)
    Traceback (most recent call last):
    ...
    TypeError: invalid multikey (must first delete item for multikey [1])
    >>> d.set((1, 2, 3, 4, 5, 6, 7), 8)
    Traceback (most recent call last):
    ...
    TypeError: invalid multikey (must first delete item for multikey [1, 2, 3, 4])
    >>> d.delete((1, 2, 3, 4))
    >>> d.set((1, 2, 3, 4, 5, 6, 7), 8)
    '''

    NIL = object()

    def __init__(self, maxdepth=None, leaffactory=None, noclobber=False):
        md = maxdepth
        lf = leaffactory
        cls = type(self)
        nc = bool(noclobber)

        if md is None:
            assert lf is None
            t = cls
            self.height = None
        else:
            if lf is None:
                lf = dict
            assert md > 0
            if md == 1:
                t = lf
            else:
                t = _subclass_factory(cls, md - 1, leaffactory=lf, noclobber=nc)

            self.height = md
            if isinstance(t, type):
                if issubclass(t, MultiKeyDict):
                    h = t().height
                    self.height = h if h is None else h + 1
                elif issubclass(t, dict):
                    self.height += 1
        super(MultiKeyDict, self).__init__(t)

        self.maxdepth = md
        self.noclobber = nc
        self._keyorder = OrderedSet()


    def __getitem__(self, key):
        md = self.maxdepth
        if md is None or md > 1:
            return super(MultiKeyDict, self).__getitem__(key)
        else:
            return dict.__getitem__(self, key)


    def __setitem__(self, key, val):
        self._keyorder_add(key)
        super(MultiKeyDict, self).__setitem__(key, val)


    def _keyorder_apply(self, method, *args, **kwargs):
        ko = self._keyorder
        if not hasattr(ko, method):
            newtype = tuple if type(ko) == OrderedSet else OrderedSet
            ko = self._keyorder = newtype(ko)
        return getattr(ko, method)(*args, **kwargs)


    def _keyorder_add(self, v):
        self._keyorder_apply('add', v)


    def _keyorder_pop(self, v):
        return self._keyorder_apply('pop', v)


    def _keyorder_index(self, v):
        return self._keyorder_apply('index', v)


    def has_key(self, *keys):
        l = self._chkkeys(keys)
        yn = super(MultiKeyDict, self).has_key(keys[0])
        if l == 1 or not yn:
            return yn
        else:
            v = self.__getitem__(keys[0])
            return v.has_key(*keys[1:])


    def _chkkeys(self, keys):
        if not hasattr(keys, '__iter__'):
            raise TypeError('keys argument must be a sequence')
        l = len(keys)
        if not l:
            raise ValueError('keys argument may not be empty')
        # md = self.maxdepth
        md = self.height
        if md is not None and l > md:
            raise KeyError('%s (exceeds maxdepth=%d)' % (str(keys), md))
        return l


    def get(self, keys):
        l = self._chkkeys(keys)
        return self._get(keys, l)


    def _get(self, keys, l):
        key = keys[0]
        hk = self.has_key(key)
        v = self.__getitem__(key)

        assert (l == 1 or isinstance(v, MultiKeyDict) or
                (isinstance(v, dict) and l == 2))

        if not hk:
            self._keyorder_add(key)

        return (v if l == 1
                else v._get(keys[1:], l - 1) if isinstance(v, MultiKeyDict)
                else v[keys[1]])


    _OK = object()
    def set(self, keys, val):
        l = self._chkkeys(keys)
        retval = error = None
        try:
            retval = self._set(l - 1, val, keys[0], *keys[1:])
        except Exception, e:
            import traceback as tb
            tb.print_exc()
            error = str(e)

        if retval is not MultiKeyDict._OK:
            msg = 'invalid multikey'
            if error is not None:
                msg += ' (%s)' % error
            if hasattr(retval, '__iter__'):
                msg += (' (must first delete item for multikey [%s])'
                        % ', '.join(map(repr, retval)))
            raise TypeError(msg)

    def _set(self, l, val, key, *subkeys):
        hk = self.has_key(key)
        if hk or l > 0:
            v = self.__getitem__(key)
            if hk and ((l == 0) == isinstance(v, MultiKeyDict)):
                return (key,)

        if l == 0:
            if not (hk and v == val):
                if self.noclobber and hk:
                    return (key,)
                self.__setitem__(key, val)
        else:
            stat = v._set(l - 1, val, subkeys[0], *subkeys[1:])
            if stat is not MultiKeyDict._OK:
                return (key,) + stat

        if not hk:
            self._keyorder_add(key)

        return MultiKeyDict._OK

    @staticmethod
    def __todict(value):
        return (value.todict() if hasattr(value, 'todict') else value)

    def todict(self):
        return dict(zip(self.keys(),
                        map(MultiKeyDict.__todict, self.values())))

    @classmethod
    def fromdict(cls, dict_, deep=False):
        ret = cls()
        for k, v in dict_.items():
            ret[k] = (cls.fromdict(v, deep) if isinstance(v, dict)
                      else _deepcopy(v) if deep else v)
        return ret
        

    def __str__(self):
        return self.todict().__str__()


    def update(self, dict_):
        for k, v in dict_.items():
            if self.has_key(k):
                vv = self[k]
                if isinstance(vv, MultiKeyDict):
                    if isinstance(v, dict):
                        vv.update(v)
                    else:
                        self.set((k,), vv)
                continue
            self[k] = v


    def delete(self, keys):
        self.popmk(keys)


    def popmk(self, keys):
        l = self._chkkeys(keys)
        try:
            return self._pop(keys, l)
        except KeyError:
            raise KeyError(str(keys))


    def __pop(self, key):
        self._keyorder_pop(key)
        return self.pop(key)


    def _pop(self, keys, l):
        k = keys[0]
        if not super(MultiKeyDict, self).has_key(k):
            raise KeyError
        if l > 1:
            d = self.__getitem__(k)
            if not isinstance(d, MultiKeyDict):
                raise KeyError

            ret = d._pop(keys[1:], l - 1)
            if not d.keys():
                self.__pop(k)
            return ret

        return self.__pop(k)


    def iterkeysmk(self, height=float('inf')):
        for k, v in self.iteritems():
            ck = () if k == self.NIL else (k,)
            if isinstance(v, MultiKeyDict) and height > 1:
                for kk in v.iterkeysmk(height=height-1):
                    yield ck + kk
            else:
                yield ck


    def itervaluesmk(self, height=float('inf')):
        for v in self.itervalues():
            if isinstance(v, MultiKeyDict) and height > 1:
                for vv in v.itervaluesmk(height=height-1):
                    yield vv
            else:
                yield v


    def iteritemsmk(self, height=float('inf')):
        for k, v in self.iteritems():
            ck = () if k == self.NIL else (k,)
            if isinstance(v, MultiKeyDict) and height > 1:
                for kk, vv in v.iteritemsmk(height=height-1):
                    yield ck + kk, vv
            else:
                yield ck, v


    def keys(self):
        return [k for k in self._keyorder]


    def values(self):
        return [self[k] for k in self.keys()]


    def items(self):
        return [(k, self[k]) for k in self.keys()]


    def sortedkeysmk(self, key=(), height=float('inf')):
        for k in self._sortedkeysmk(key, height):
            yield k[0] if height == 1 else k


    def _sortedkeysmk(self, key, height):
        ks = sorted(self.keys(), key=key[0]) if key and key[0] \
             else self._keyorder
        for k in ks:
            ck = () if k == self.NIL else (k,)
            v = self[k]
            if isinstance(v, MultiKeyDict) and height > 1:
                for kk in v._sortedkeysmk(key=key[1:], height=height-1):
                    yield ck + kk
            else:
                yield ck


    def sortedvaluesmk(self, key=(), height=float('inf')):
        vs = sorted(self.values(), key=key[0]) if key and key[0] \
             else [self[k] for k in self._keyorder]
        for v in vs:
            if isinstance(v, MultiKeyDict) and height > 1:
                for vv in v.sortedvaluesmk(key=key[1:], height=height-1):
                    yield vv
            else:
                yield v


    def sorteditemsmk(self, key=(), height=float('inf')):
        for k, v in self._sorteditemsmk(key, height):
            yield ((k[0] if height == 1 else k), v)


    def _sorteditemsmk(self, key, height):
        kvs = sorted(self.items(), key=key[0]) if key \
              else [(k, self[k]) for k in self._keyorder]
        for k, v in kvs:
            ck = () if k == self.NIL else (k,)
            if isinstance(v, MultiKeyDict) and height > 1:
                for kk, vv in v._sorteditemsmk(key=key[1:], height=height-1):
                    yield ck + kk, vv
            else:
                yield ck, v


    def hasmultikey(self, keytuple):
        try:
            (ignored) = self.index(keytuple)
        except ValueError, e:
            if 'not found' in str(e):
                return False
            raise
        return True


    def index(self, keytuple):
        lk = self._chkkeys(keytuple)
        try:
            return self._index(keytuple, lk)
        except ValueError, e:
            if str(e).endswith('not in tuple'):
                raise ValueError("'%s' not found" % str(keytuple))
            else:
                raise
            

    def _index(self, keytuple, lk):
        k0 = keytuple[0]
        i = (self._keyorder_index(k0),)
        return i if lk == 1 else i + self[k0]._index(keytuple[1:], lk - 1)


    def permutekeys(self, perm, deepcopy=False):
        mkd = type(self)()
        mkd.__dict__ = _deepcopy(self.__dict__)

        if hasattr(perm, '__call__'):
            _permute = perm
        else:
            _permute = lambda k: tuple(k[i] for i in perm)

        mkd._keyorder = OrderedSet()
        if deepcopy:
            for k, v in self.sorteditemsmk():
                v = _deepcopy(v)
                mkd.set(_permute(k), v)
        else:
            for k, v in self.sorteditemsmk():
                mkd.set(_permute(k), v)
            
        return mkd


    def collapsekeys(self, height, deepcopy=False):
        assert height > 1
        mkd = type(self)()
        if deepcopy:
            for k, v in self.sorteditemsmk(height=height):
                mkd[k] = _deepcopy(v)
        else:
            for k, v in self.sorteditemsmk(height=height):
                mkd[k] = v

        return mkd


    def __dimvals(self):
        def _dv(x):
            return x.__dimvals() if isinstance(x, MultiKeyDict) else set()
        children = map(_dv, self.values())
        return [map(unicode, self._keyorder)] + [sum(k, OrderedSet())
                                                 for k in zip(*children)]


    def _dimvals(self):
        return map(tuple, self.__dimvals())


    def __reduce_ex__(self, proto):
        assert proto == 0
        return (type(self), (), self.__dict__, None, self.iteritems())
        

def _subclass_factory(cls, default_maxdepth, _memo=dict(), **nspace):
    assert default_maxdepth > 0
    key = tuple([default_maxdepth] + sorted(nspace.items()))
    ret = _memo.get(key, None)
    if ret is None:
        kwargs = _deepcopy(nspace)
        class _submkd(cls):
            def __init__(self, maxdepth=default_maxdepth, **ignored):
                cls.__init__(self, maxdepth=maxdepth, **kwargs)

            def __reduce_ex__(self, proto):
                return (MultiKeyDict, (), self.__dict__, None, self.iteritems())

        _submkd.__name__ = name = '_submkd__%d' % id(_submkd)
        globals()[name] = _memo[key] = ret = _submkd
    return ret
