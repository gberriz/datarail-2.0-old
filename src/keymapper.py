from multikeydict import MultiKeyDict as mkd


class SimpleKeyMapper(dict):
    __max_ids = 2**32

    def __init__(self, seq=None, offset=0,
                 _id_min=-__max_ids/2, _id_max=__max_ids/2, **kwargs):

        self._offset = offset
        self._inverse = inv = []
        self._len = len(inv)

        if seq is None:
            id_min = kwargs.pop('id_min', _id_min)
            id_max = kwargs.pop('id_max', _id_max)
            if kwargs:
                raise TypeError('unrecognized keyword(s): %s' %
                                ', '.join(kwargs.keys()))
            max_ids = id_max - id_min
            if max_ids < 0:
                raise ValueError('id_max - id_min must be nonnegative')
            if not id_min <= offset <= id_max:
                raise ValueError('offset out of range')

            self._state = self._offset
            def _newid(id_min=id_min, id_max=id_max, max_ids=max_ids):
                if self._len >= max_ids:
                    raise ValueError('no more ids available')
                ret = self._state
                if ret >= id_max:
                    assert ret == id_max
                    self._state = ret = id_min
                self._state += 1

                s = ret if ret >= self._offset else ret + self._max_ids
                assert self._len == s - self._offset

                return ret

            self.seq = _newid
        else:
            if 'id_min' in kwargs or 'id_max' in kwargs:
                raise TypeError('specifying id_min or id_max is '
                                'incompatible with specifying seq')
            self.seq = lambda: next(seq)

        super(SimpleKeyMapper, self).__init__()


    def __call__(self, key):
        return super(SimpleKeyMapper, self).__getitem__(unicode(key))


    def __getitem__(self, i):
        ii = self._inverse_index(i)
        try:
            return self._inverse[ii]
        except IndexError:
            raise KeyError('mapper index out of range')


    def __setitem__(self, i, v):
        raise TypeError('read-only access')
        

    def getid(self, key):
        try:
            ret = self(key)
        except KeyError:
            key, ret = unicode(key), self.seq()
            super(SimpleKeyMapper, self).__setitem__(key, ret)
            self._update_inverse(key)
        return ret


    def todict(self):
        return dict(self)


    key2idmap = todict


    def id2keymap(self):
        return dict((v, k) for k, v in self.items())


    def _inverse_index(self, i):
        if not isinstance(i, int):
            raise TypeError('argument must be an integer')
        return i - self._offset


    def _update_inverse(self, ukey):
        inverse = self._inverse
        inverse.append(ukey)
        self._len = len(inverse)
        

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
