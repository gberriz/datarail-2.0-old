from collections import defaultdict
from copy import deepcopy

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

    def __init__(self):
        # self.maxdepth = maxdepth
        super(MultiKeyDict, self).__init__(type(self))

    def has_key(self, *keys):
        l = len(keys)
        assert l
        yn = super(MultiKeyDict, self).has_key(keys[0])
        if l == 1 or not yn:
            return yn
        else:
            v = super(MultiKeyDict, self).__getitem__(keys[0])
            return v.has_key(*keys[1:])

    def get(self, keys):
        l = len(keys)
        assert l
        v = super(MultiKeyDict, self).__getitem__(keys[0])
        if l == 1:
            return v
        else:
            return v.get(keys[1:])

    _OK = object()
    def set(self, keys, val):
        retval = None
        error = None
        if hasattr(keys, '__iter__') and len(keys):
            try:
                retval = self._set(val, keys[0], *keys[1:])
            except Exception, e:
                raise
                # from traceback import traceback as tb
                # error = str(e)

        if retval is not MultiKeyDict._OK:
            msg = 'invalid multikey'
            if error is not None:
                msg += ' (%s)' % error
            if hasattr(retval, '__iter__'):
                msg += (' (must first delete item for multikey [%s])'
                        % ', '.join(map(repr, retval)))
            raise TypeError(msg)

    def _set(self, val, key, *subkeys):
        l = len(subkeys)
        hk = self.has_key(key)
        sc = super(MultiKeyDict, self)
        if hk or l > 0:
            v = sc.__getitem__(key)
            if hk and ((l == 0) == isinstance(v, MultiKeyDict)):
                return (key,)

        if l == 0:
            sc.__setitem__(key, val)
        else:
            stat = v._set(val, subkeys[0], *subkeys[1:])
            if stat is not MultiKeyDict._OK:
                return (key,) + stat

        return MultiKeyDict._OK

    def __setitem__(self, key, val):
        try:
            self.set((key,), val)
        except TypeError, e:
            raise TypeError(str(e))

    @staticmethod
    def __todict(value):
        # return (value.todict() if isinstance(value, MultiKeyDict) else value)
        return (value.todict() if hasattr(value, 'todict') else value)

    def todict(self):
        return dict(zip(self.keys(),
                        map(MultiKeyDict.__todict, self.values())))

    @classmethod
    def fromdict(cls, dict_, deep=False):
        ret = cls()
        for k, v in dict_.items():
            ret[k] = (cls.fromdict(v, deep) if isinstance(v, dict)
                      else deepcopy(v) if deep else v)
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
    #     l = len(keys)
    #     assert l and hasattr(keys, '__iter__')
    #     try:
    #         self._del(keys, l)
    #     except KeyError:
    #         raise KeyError(str(keys))

    # def _del(self, keys, l):
    #     k = keys[0]
    #     if not super(MultiKeyDict, self).has_key(k):
    #         raise KeyError
    #     if l > 1:
    #         d = super(MultiKeyDict, self).__getitem__(k)
    #         if not isinstance(d, MultiKeyDict):
    #             raise KeyError
    #         d._del(keys[1:], l - 1)
    #         if d.keys():
    #             return
    #     self.pop(k)


    def popmk(self, keys):
        l = len(keys)
        assert l and hasattr(keys, '__iter__')
        try:
            return self._pop(keys, l)
        except KeyError:
            raise KeyError(str(keys))


    def _pop(self, keys, l):
        k = keys[0]
        if not super(MultiKeyDict, self).has_key(k):
            raise KeyError
        if l > 1:
            d = super(MultiKeyDict, self).__getitem__(k)
            if not isinstance(d, MultiKeyDict):
                raise KeyError
            ret = d._pop(keys[1:], l - 1)
            if not d.keys():
                self.pop(k)
            return ret
        return self.pop(k)
