import sys
from copy import deepcopy


class Dimension(object):
    def __init__(self, name, category=None, type_=None, units=None,
                 values=None, missing=None, position=None):
        kw = deepcopy(locals())
        kw.pop('self'); kw.pop('name')
        self._kwargs = [('name', name),] + kw.items()
        del kw

        object.__setattr__(self, 'name', name)
        self.category = category
        self.type = type_
        self.units = units
        self.missing = missing
        if values is None:
            self.values = self.__allowable = None
        else:
            self.values = o = [Dimension.convert(x, type_)
                               for x in values.split('|')]
            self.__allowable = set(o)
        self.position = sys.maxint if position is None else position
        from pdb import set_trace as ST


    def __repr__(self):
        kw = ', '.join(['%s=%r' % p for p in self._kwargs])
        return '%s(%s)' % (self.__class__.__name__, kw)


    def __str__(self):
        return self.name


    def __eq__(self, dim):
        return self.name.__eq__(str(dim))


    def __hash__(self):
        return self.name.__hash__()


    def __setattr__(self, attrname, value):
        if attrname == 'name':
            raise AttributeError("'%s' object attribute '%s' is read-only" %
                                 (self.__class__.__name__, attrname))
        object.__setattr__(self, attrname, value)


    def read(self, value):
        ret = self.convert(value, self.type)
        allowable = self.__allowable
        if not (allowable is None or ret in allowable):
            raise ValueError("'%s' is not a valid value for '%s'" %
                             (ret, self.name))
        return ret


    def konvert(self, value):
        return self.convert(value, self.type)

    @classmethod
    def convert(cls, value, type_):
        t = type_
        if t == 'integer':
            return int(value)
        if t == 'real':
            return float(value)
        if t == 'string':
            return value
        raise ValueError('Unsupported type: %s' % t)


    def collapse(self, vals):
        v = filter(lambda x: x != self.missing, vals)
        if not v:
            v = [self.missing]
        return v
