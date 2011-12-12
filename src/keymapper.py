class SimpleKeyMapper(dict):
    __MAX_IDS = 2**32
    __ID_MAX = __MAX_IDS/2
    __ID_MIN = -__MAX_IDS/2

    def __init__(self, offset=0,
                 _ID_MIN=__ID_MIN, _ID_MAX=__ID_MAX):
        self._id_min = _ID_MIN
        self._id_max = _ID_MAX
        self._max_ids = _max_ids = _ID_MAX - _ID_MIN
        if _max_ids < 0:
            raise ValueError('_ID_MAX - _ID_MIN must be nonnegative')
        if not _ID_MIN <= offset <= _ID_MAX:
            raise ValueError('offset out of range')
        self._state = self._offset = offset
        self._inverse = inv = []
        self._len = len(inv)
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
            _, ret = kv = unicode(key), self._newid()
            super(SimpleKeyMapper, self).__setitem__(*kv)
            self._update_inverse(*kv)
        return ret


    def _newid(self):
        if self._len >= self._max_ids:
            raise ValueError('no more ids available')
        ret = self._state
        if ret >= self._id_max:
            assert ret == self._id_max
            self._state = ret = self._id_min
        self._state += 1
        return ret

    def _update_inverse(self, ukey, id_):
        inverse = self._inverse
        inverse.append(ukey)
        self._len = l = len(inverse)
        s = id_ if id_ >= self._offset else id_ + self._max_ids
        assert l == s + 1 - self._offset
        

    def _inverse_index(self, i):
        if not isinstance(i, int):
            raise TypeError('argument must be an integer')
        return i - self._offset


class KeyMapper(object):
    def __init__(self, arg0, *rest):
        if type(arg0) == int:
            if len(rest) > 0:
                raise TypeError('(only one int argument allowed)')
            args = (None for _ in xrange(arg0))
        else:
            args = (arg0,) + rest

        self.mappers = tuple([SimpleKeyMapper() if arg is None else arg
                              for arg in args])

    def __call__(self, tup):
        return tuple([mpr(v) for mpr, v in zip(self.mappers, tup)])


    def __getitem__(self, tup):
        return tuple([mpr[v] for mpr, v in zip(self.mappers, tup)])


    def __setitem__(self, i, v):
        raise TypeError('read-only access')


    def getid(self, tup):
        return tuple([mpr.getid(v) for mpr, v in zip(self.mappers, tup)])
