try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

from collections import Sequence

class frozendict(OrderedDict):
    """An immutable ordered dictionary class.

    Initialize using either an ordered sequence (e.g. list or tuple)
    of valid key-value pairs or an instance of OrderedDict.

    Since it is immutable, this class is hashable.

    Any attempt to call one of dict's destructive methods results in a
    TypeError.  E.g.:

    >>> from frozendict import frozendict
    >>> f = frozendict((i, i + 1) for i in range(0, 10, 2))
    >>> f
    {0: 1, 8: 9, 2: 3, 4: 5, 6: 7}
    >>> f[0] = 5
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "src/frozendict.py", line 15, in _immutable_error
        raise TypeError("'%s' object is not mutable" % self.__class__.__name__)
    TypeError: 'frozendict' object is not mutable
    """

    def __new__(cls, sequence_or_ordered_dict):
        if not (isinstance(sequence_or_ordered_dict, Sequence) or
                isinstance(sequence_or_ordered_dict, OrderedDict)):
            raise TypeError('initializer must be ordered')

        return super(frozendict, cls).__new__(cls, sequence_or_ordered_dict)


    def __init__(self, sequence_or_ordered_dict):
        self.__initializing = True
        OrderedDict.__init__(self, sequence_or_ordered_dict)
        self.__initializing = False


    def __hash__(self):
        return hash(frozenset(self.items()))


def _make_method(name):
    def proxy(self, *args, **kwargs):
        if not self.__initializing:
            raise TypeError("'%s' object is not mutable" % type(self).__name__)

        return super(frozendict, self).__getattribute__(name)(*args, **kwargs)

    proxy.__name__ = name
    return proxy


for m in ('__delitem__ __setitem__ update clear pop '
          'popitem setdefault'.split()):
    setattr(frozendict, m, _make_method(m))

del _make_method
