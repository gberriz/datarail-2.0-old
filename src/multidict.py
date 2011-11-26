from collections import defaultdict
from itertools import chain

from pdb import set_trace as ST

class MultiDict(defaultdict):
    def __init__(self, *args, **kwargs):
        if len(args) > 1:
            raise TypeError('%s expected at most 1 argument, got %d' %
                            (type(self).__name__, len(args)))
        super(MultiDict, self).__init__(list)
        self.update(*args, **kwargs)
        self.__checkvalues__(self, '__init__', 'append', 'extend')


    @staticmethod
    def _isiterable(item):
        return hasattr(item, '__iter__')


    class __bad_value_err__(Exception):
        _LISTATTRIBS = 'append extend'.split()
        def __init__(self, cname, caller, methodnames=_LISTATTRIBS):
            self._msg = ('values in argument to %s.%s must have '
                         'the following method(s): %s' %
                         (cname, caller,
                          ', '.join(['%s' % m for m in methodnames])))
        def __str__(self):
            return self._msg


    def _update(self, iterable, exc=__bad_value_err__,
                attribs=__bad_value_err__._LISTATTRIBS):
        for k, v in iterable:
            if not all(hasattr(v, a) for a in attribs):
                raise exc(type(self).__name__, 'update')
            self[k].extend(v)


    def update(self, *args, **kwargs):
        """
        Update this dictionary with the values in multidict.
        """

        nargs = len(args)
        cname = type(self).__name__
        if nargs > 1:
            raise TypeError('%s.update expected at most 1 argument, got %d' %
                            (cname, nargs))
        del nargs

        try:
            if args:
                arg = args[0]
                if hasattr(arg, 'iteritems'):
                    iter_ = arg.iteritems()
                else:
                    _ii = MultiDict._isiterable
                    if not _ii(arg):
                        raise TypeError("'%s' object is not iterable" % type(arg))
                    iter_ = arg.__iter__()

                try:
                    self._update(iter_)
                except MultiDict.__bad_value_err__:
                    raise
                except:
                    # import sys, pdb, traceback
                    # _, value, tb = sys.exc_info()
                    # traceback.print_exc()
                    # pdb.post_mortem(tb)
                    raise TypeError('cannot convert %s update sequence '
                                'element #0 to a sequence' % cname)

            if kwargs:
                self._update(kwargs.iteritems())

        except MultiDict.__bad_value_err__, e:
            raise ValueError(str(e))


    @staticmethod
    def __has_method__(obj, methodname):
        try:
            m = getattr(obj, methodname)
        except AttributeError:
            return False
        else:
            return hasattr(m, '__call__')


    def __checkvalues__(self, dict_, caller, *methodnames):
        if any([True for v in dict_.itervalues()
                if not all([MultiDict.__has_method__(v, mn)
                            for mn in methodnames])]):
            raise TypeError('values in argument to %s.%s must have '
                            'the following method(s): %s' %
                            (type(self).__name__, caller,
                             ', '.join(['"%s"' % m for m in methodnames])))


    def __missing__(self, key):
        value = self.default_factory()
        dict.__setitem__(self, key, value)
        return value

    def __setitem__(self, key, value):
        self[key].append(value)

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, dict.__repr__(self))
