import issequence as iss

import traceback as tb
from pdb import set_trace as ST

def _checkargs(func, *args, **kw):
    return func

class NullLevel(object):
    def __repr__(self):
        return 'NULL_LEVEL'

NULL_LEVEL = NullLevel()

class Dimension(tuple):
    @classmethod
    def _NullDimension(cls, name, _null=NULL_LEVEL):
        return cls(name, (_null,))

    @_checkargs
    def __new__(cls, name, levels):
        l = len(levels)
        assert l == len(set(levels)) # avoid repeats
        return super(Dimension, cls).__new__(cls, levels)


    @_checkargs
    def __init__(self, name, levels):
        self.name = name
        self.__len = l = len(levels)
        self.__index = dict(zip(levels, range(l)))


    def index(self, spec):
        if iss.issequence(spec):
            return tuple(map(self.__toindex, spec))
        if spec is None:
            return slice(0, self.__len, None)
        if isinstance(spec, slice):
            b, e = spec.start, spec.stop
            try:
                start = 0 if b is None else self.__toindex(b)
            except IndexError:
                start = None
            try:
                stop = self.__len if e is None else self.__toindex(e)
            except IndexError:
                stop = None
            return slice(start, stop, spec.step)
        if callable(spec):
            return tuple(i for i, v in enumerate(self)
                         if spec(i, v))

        # we must convert simple integer indices to slices, otherwise
        # numpy will get rid of the corresponding (trivial) dimension
        ret = self.__toindex(spec)
        return slice(ret, ret + 1) 
        

    def __toindex(self, spec):
        if isinstance(spec, int):
            l = self.__len
            if -l <= spec < l:
                return spec
            raise IndexError, 'index out of range'
        try:
            return self.__index[spec]
        except KeyError:
            raise ValueError, 'invalid level for dimension "%s" (%r)' \
                              % (self.name, spec)


    def __getitem__(self, spec):
        _get = super(Dimension, self).__getitem__
        if spec is None:
            return _get(slice(None))
        try:
            if iss.issequence(spec):
                return tuple(map(_get, spec))

            if not callable(spec):
                return _get(spec)
        except:
            raise IndexError, 'invalid index for dimension "%s" (%r)' \
                              % (self.name, spec)

        try:
            return tuple(v for i, v in enumerate(self)
                    if spec(i, v))
        except Exception, e:
            raise type(e), 'predicate error for dimension "%s" (%s)' \
                           % (self.name, e)



Dimension.__call__ = Dimension.index
Dimension.level = Dimension.__getitem__
