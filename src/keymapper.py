from multikeydict import MultiKeyDict as mkd


class SimpleKeyMapper(dict):
    __max_ids = 2**32
    __fill = [object()]

    def __init__(self, seq=None, offset=0,
                 _id_min=-__max_ids/2, _id_max=__max_ids/2, **kwargs):

        self._offset = offset
        self._inverse = inv = {}
        self._len = len(self)
        self.seq = lambda: next(seq)
        super(SimpleKeyMapper, self).__init__()


    def __call__(self, key):
        return super(SimpleKeyMapper, self).__getitem__(unicode(key))


    def __getitem__(self, i):
        try:
            ret = self._inverse[i]
            assert super(SimpleKeyMapper, self).__getitem__(ret) == i
        except (KeyError, IndexError, AssertionError):
            raise KeyError('unknown index (%d)' % i)

        return ret


    def __setitem__(self, i, v):
        raise TypeError('read-only access')
        

    def getid(self, key):
        try:
            i = self(key)
            # i.e., i = super(SimpleKeyMapper, self).__getitem__(unicode(key))
        except KeyError:
            ukey, i = unicode(key), self.seq()
            super(SimpleKeyMapper, self).__setitem__(ukey, i)
            self._update_inverse(ukey, i)
        return i


    def todict(self):
        return dict(self)


    key2idmap = todict


    def id2keymap(self):
        return dict((v, k) for k, v in self.items())


    def _update_inverse(self, ukey, i):
        inverse = self._inverse
        if hasattr(inverse, 'extend'):
            s = 1 + i - len(inverse)
            assert s > 0
            inverse.extend(self.__fill * s)
        else:
            assert hasattr(inverse, 'items') and i not in inverse
        inverse[i] = ukey


    del __max_ids


class KeyMapper(object):
    def __init__(self, arg0, *rest):
        if type(arg0) == int:
            if len(rest) > 0:
                raise TypeError('(only one int argument allowed)')
            args = (None for _ in xrange(arg0))
        else:
            args = (arg0,) + rest

        def _getmapper(m):
            return (SimpleKeyMapper(m) if m is None
                      or hasattr(m, '__iter__')
                      or hasattr(m, '__next__')
                    else m)

        self.mappers = tuple(map(_getmapper, args))


    def __call__(self, key):
        # NOTE: key must be a tuple (or at least a sequence)
        return tuple([mpr(v) for mpr, v in zip(self.mappers, key)])


    def __getitem__(self, i):
        # NOTE: i must be a tuple (or at least a sequence)
        return tuple([mpr[v] for mpr, v in zip(self.mappers, i)])


    def __setitem__(self, i, v):
        raise TypeError('read-only access')


    def getid(self, key):
        # NOTE: key must be a tuple (or at least a sequence)
        return tuple([mpr.getid(v) for mpr, v in zip(self.mappers, key)])


    def todict(self, _seqtype=tuple):
        return _seqtype(m.todict() for m in self.mappers)


    def key2idmap(self, _seqtype=tuple):
        return _seqtype(m.key2idmap() for m in self.mappers)


    def id2keymap(self, _seqtype=tuple):
        return _seqtype(m.id2keymap() for m in self.mappers)


    def getkey(self, ids):
        return tuple([mpr[i] for mpr, i in zip(self.mappers, ids)])
