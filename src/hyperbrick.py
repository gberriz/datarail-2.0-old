import sys as sy
import types as ty
import numpy as np

import h5helper as h5h
from pdb import set_trace as ST

class HyperBrick(object):
    _dimnames = ()
    _dimvals = ()
    _shape = ()


    def __init__(self, block):
        self._block = block


    def _toindex(self, *args, **kwargs):
        assert args or kwargs

        nargs = len(args)
        dimnames = self._dimnames
        ndims = len(dimnames)
        if nargs > ndims:
            raise ValueError('too many arguments')
        dimvals = self._dimvals
        argslist = list(args) + [None] * (ndims - nargs)
        #ST()
        for k, v in kwargs.items():
            try:
                i = dimnames.index(k)
                j = dimvals[i].index(v)
            except ValueError:
                raise ValueError, 'invalid index: "%s='"'%s'"'"' % (k, v)

            if i < nargs:
                raise ValueError, 'repeated dimension: "%s"' % k

            argslist[i] = j

        idx = [slice(None)] * ndims

        for i, pa in enumerate(argslist):
            if pa is None:
                continue
            elif isinstance(pa, int):
                idx[i] = pa
            else:
                try:
                    if isinstance(pa, tuple):
                        idx[i] = dimvals[i].index(pa)
                    else:
                        idx[i] = dimvals[i].index(str(pa))
                except ValueError:
                    raise ValueError, 'invalid index: "%s"' % pa

        return tuple(idx)


    def _tolabels(self, idx):
        dimvals = self._dimvals
        dimnames = self._dimnames

            
    def __call__(self, *args, **kwargs):
        idx = self._toindex(*args, **kwargs)
        return self.__getitem__(self._toindex(*args, **kwargs))
        

    def __getitem__(self, args):
        print args
        return self._block.__getitem__(args)


    @staticmethod
    def _canonicalize_labels(labels):
        # this = HyperBrick
        if not hasattr(labels, '__iter__'):
            raise TypeError('argument is not an iterable')

        return tuple(labels.items()) \
               if hasattr(labels, 'items') else \
               tuple([(i[0], tuple(i[1])) for i in labels])


    @staticmethod
    def makesubclass(labels, _memo={}):
        this = HyperBrick
        labels = this._canonicalize_labels(labels)

        cls = _memo.get(labels)
        if cls is None:
            h = labels.__hash__()
            superclass = this.__name__
            classname = '_%s_%s%s' % (superclass, ('', '_')[h < 0], abs(h))
            dimnames, dimvals = map(tuple, zip(*labels))
            dimvals = tuple(map(tuple, dimvals))
            shape = map(len, dimvals)

            code = ('''
class %(classname)s(%(superclass)s):\n
    _dimnames = %(dimnames)s
    _dimvals = %(dimvals)s
    _shape = %(shape)s\n
    ''' % locals()).lstrip('\n').rstrip(' ')


            # CODE BELOW LIFTED FROM collections.namedtuple

            # Execute the template string in a temporary namespace and
            # support tracing utilities by setting a value for
            # frame.f_globals['__name__']
            namespace = dict(classname=classname, dimnames=dimnames,
                             dimvals=dimvals)
            namespace[superclass] = this

            exec code in namespace
            # try:
            #     exec code in namespace
            # except SyntaxError, e:
            #     raise SyntaxError(e.message + ':\n' + code)
            _memo[labels] = cls = namespace[classname]

            # For pickling to work, the __module__ variable needs to
            # be set to the frame where the named tuple is created.
            # Bypass this step in enviroments where sys._getframe is
            # not defined (Jython for example) or sys._getframe is not
            # defined for arguments greater than 0 (IronPython).
            try:
                cls.__module__ = sy._getframe(1).f_globals.get('__name__',
                                                               '__main__')
            except (AttributeError, ValueError):
                pass

        return cls
            
# import numpy as np
# shape = (3, 4, 5)
# nda = np.ndarray(shape)
# db = HyperBrick(nda)
# eggs = db[0]
# spam = db[1, 2, 3]
# ham = db[::, ::]

path = 'icbp45.h5'
with h5h.Hdf5File(path, 'r+') as h5:
    block = h5['from_IR/CK']
    data = block['data']
    labels = h5h.load(block['labels'].value)
    sc = HyperBrick.makesubclass(labels)
    hyperbrick = sc(data)
    print hyperbrick._toindex(ligand_name='CTRL')
    print hyperbrick(ligand_name='CTRL')
