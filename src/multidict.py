from collections import defaultdict
from pdb import set_trace as ST

class MultiDict(defaultdict):
    def __init__(self, *args, **kwargs):
        super(MultiDict, self).__init__(list, *args, **kwargs)
        self.__checkvalues__(self, '__init__', 'append', 'extend')

    def update(self, multidict):
        """
        Update this dictionary with the values in multidict.
        """

        if not isinstance(d, MultiDict):
            raise ValueError('only instances of MultiDict may be '
                             'passed to %s.update method' %
                             type(self).__name__)

        self.__checkvalues__(d, 'update', '__iter__')
        for k, v in d.items():
            self[k].extend(v)


    @staticmethod
    def __has_method__(obj, methodname):
        try:
            m = obj.getattr(methodname)
        except AttributeError:
            return False
        else:
            return hasattr(m, '__call__')


#     @classmethod
#     def __bad_value_err__(cls, caller, *methodnames):
#         return TypeError('values in argument to %s.%s must have '
#                          'the following method(s): %s' %
#                          (type(self).__name__, caller,
#                           ', '.join(['"%s"' for m in methodnames])))


    def __checkvalues__(self, dict_, caller, *methodnames):
        if any([True for v in dict_.itervalues()
                if not all([MultiDict.__has_method__(obj, mn)
                            for mn in methodnames])]):
            raise TypeError('values in argument to %s.%s must have '
                            'the following method(s): %s' %
                            (type(self).__name__, caller,
                             ', '.join(['"%s"' for m in methodnames])))


    def __missing__(self, key):
        value = self.default_factory()
        dict.__setitem__(self, key, value)
        return value

    def __setitem__(self, key, value):
        self[key].append(value)

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, dict.__repr__(self))
