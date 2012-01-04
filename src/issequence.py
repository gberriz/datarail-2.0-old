import types as ty


try:
    from collections import Iterable
except ImportError:
    def isiterable(x):
        try: iter(x)
        except TypeError: return False
        else: return True
else:
    def isiterable(x):
        return isinstance(x, Iterable)

try:
    from collections import Iterator
    # collections.Iterator is useless for this.  E.g.:
    # >>> li = [1, 2, 3].__iter__()
    # >>> type(li)
    # <type 'listiterator'>
    # >>> hasattr(li, '__iter__') and hasattr(li, 'next')
    # True
    # >>> isinstance(li, Iterator)
    # False
    raise ImportError, 'until collections.Iterator works'
except ImportError:
    def isiterator(x):
        return hasattr(x, '__iter__') and hasattr(x, 'next')
else:
    def isiterator(x):
        return isinstance(x, Iterator)


def isstring(x):
    return isinstance(x, ty.StringTypes)


def issequence(x):
    return isiterable(x) and not isstring(x)
