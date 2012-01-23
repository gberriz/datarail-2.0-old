import numpy as np
import dimension as di
import align as al

from pdb import set_trace as ST

class HyperBrickDimensions(tuple):
    def __init__(self, dims):
        self.__dict__ = dict((d.name, d) for d in dims)
        self.__indexlookup = dict((d.name, i) for i, d in enumerate(dims))

    def __call__(self, key):
        return self.__dict__[key]

    def _get(self, key, default=None):
        return self.__dict__.get(key, default)

    def _index(self, key):
        return self.__indexlookup[key]

    def _replace(self, key, value=None):
        if value is None:
            if not isinstance(key, di.Dimension):
                raise TypeError, 'single argument must be a Dimension object'
            value = key
            key = value.name
        else:
            if hasattr(value, 'name') and value.name != key:
                raise TypeError, 'inconsistent arguments'

            if not isinstance(value, di.Dimension):
                value = di.Dimension(key, value)

        if key not in self.__dict__:
            raise TypeError, 'unknown dimension: %r' % key
            
        as_list = list(self)
        as_list[self._index(key)] = value
        return type(self)(as_list)

    @property
    def _names(self):
        return tuple(d.name for d in self)

    def __contains__(self, key):
        return key in self.__dict__


class HyperBrick(object):

    def __init__(self, data, labels):
        dlist = [nv if isinstance(nv, di.Dimension)
                 else di.Dimension(*nv) for nv in labels]
        self._dims = dims = HyperBrickDimensions(dlist)

        self.ndim = ndim = len(dims)
        if len(set([d.name for d in dims])) < ndim:
            raise TypeError('repeated dimension names')            

        self._axes = dict([(d.name, i) for i, d in enumerate(dims)])
        self.shape = shape = tuple(map(len, dims))
        try:
            self._data = data.reshape(shape)
        except ValueError, e:
            if str(e) != 'total size of new array must be unchanged':
                raise
            raise TypeError('data has size %d but labels imply a size of '
                            '%d' % (data.size, np.product(shape)))
        self.data = self._data


    def _toaxis(self, dimname):
        return dimname if isinstance(dimname, int) else self._axes[dimname]

    def _fromaxis(self, axis):
        return self._dims[axis] if isinstance(axis, int) else axis


    def complement(self, dims):
        nonaxes = tuple([i for i in range(len(inshape))
                         if i not in set(axes)])


    def _toslice(self, pa, dimvals, dimname):
        if pa is None:
            return slice(None)
        if isinstance(pa, slice):
            return pa
        if isinstance(pa, tuple):
            start = pa[0]
            stop = pa[-1] + 1
            step = (stop - start)//len(pa)
            if pa == tuple(range(start, stop, step)):
                return slice(start, stop, step)
            return pa
        try:
            ii = (pa if isinstance(pa, int) else dimvals.index(pa))
        except IndexError:
            raise IndexError, 'invalid level for "%s": %s' % (dimname, pa)

        jj = None if ii >= len(dimvals) else ii + 1
        return slice(ii, jj, None)


    def _toindex(self, *args, **kwargs):
        """Convert callargs to an ndarray index.
        """
        assert args or kwargs

        nargs = len(args)
        ndim = self.ndim
        if nargs + len(kwargs) > ndim:
            raise TypeError('too many arguments')

        try:
            firstellipsis = args.index(Ellipsis)
        except ValueError:
            pass
        else:
            if kwargs:
                raise TypeError('Ellipsis is not compatible with '
                                'keyword parameters')

            extra = ndim - nargs + 1

            # Note: there's only one slice(None) object, so the way
            # the expression [slice(None)] * extra is OK
            args = args[:firstellipsis] + \
                   tuple([slice(None)] * extra) + \
                   tuple(None if arg is Ellipsis else arg
                         for arg in args[(firstellipsis + 1):])
            nargs = ndim

        argslist = list(args) + [None] * (ndim - nargs)

        dims = self._dims
        dimnames = tuple(d.name for d in dims)

        for k, pa in kwargs.items():
            try:
                i = dimnames.index(k)
            except ValueError:
                raise ValueError, 'invalid dimension: %s' % k
            if i < nargs:
                raise ValueError, 'repeated dimension: %s' % k

            argslist[i] = pa

        idx = [slice(None)] * ndim
        for i, pa in enumerate(argslist):
            idx[i] = dims[i].index(pa)

        return tuple(idx)


    def _tolabels(self, idx):
        nargs = len(idx)
        ndim = self.ndim
        if nargs > ndim:
            raise ValueError('too many arguments')

        dimvals = self._dims[:nargs]
        dimnames = tuple(d.name for d in self._dims[:nargs])
        slices = map(self._toslice, idx, dimnames, dimvals) + \
                 [slice(None) for _ in range(ndim - nargs)]
        return zip(dimnames, tuple(dv.__getitem__(sl) for
                                   dv, sl in zip(dimvals, slices)))

    def _tolevels(self, idx):
        return tuple(l[1][0] for l in self._tolabels(idx))
            
    def __call__(self, *args, **kwargs):
        """
        """
        if args or kwargs:
            return self.__getitem__(self._toindex(*args, **kwargs))
        else:
            raise NotImplementedError


    def __getitem__(self, args):
        subbrick = self._data.__getitem__(args)
        if subbrick.size <= 2:
            return subbrick
        else:
            return type(self)(subbrick, self._tolabels(args))


    @staticmethod
    def _canonicalize_labels(labels):
        if not hasattr(labels, '__iter__'):
            raise TypeError('argument is not an iterable')

        return tuple(labels.items()) \
               if hasattr(labels, 'items') else \
               tuple([(i[0], tuple(i[1])) for i in labels])


    @property
    def labels(self):
        return tuple([(d.name, d) for d in self._dims])


    def squeeze(self):
        return self.reshape(tuple(filter(lambda d: len(d) > 1, self._dims)))


    def reshape(self, dims):
        return HyperBrick(self._data, dims)


    def extrude(self, newlabels):
        newshape = tuple(len(l if isinstance(l, di.Dimension) else l[1])
                         for l in newlabels)
        base = self._data
        extrusion = np.empty(newshape, dtype=base.dtype)
        extrusion[...] = base
        return HyperBrick(extrusion, newlabels)

    def concatenate(self, bricks, dim=None):
        bb = (self, bricks) \
             if isinstance(bricks, HyperBrick) \
             else (self,) + tuple(bricks)

        if len(set(b.ndim for b in bb)) > 1:
            raise TypeError, ('concatenation requires equal numbers '
                              'of dimensions')

        axis = 0 if dim is None else self._toaxis(dim)

        for i, ls in enumerate(zip(*(b.labels for b in bb))):
            if i == axis:
                if len(set((l[0] for l in ls))) > 1:
                    raise TypeError, ('concatenation requires equally '
                                      'named dimensions')
                newdims = self._dims._replace(ls[0][0],
                                              sum((l[1] for l in ls), ()))
            else:
                if len(set(ls)) > 1:
                    raise TypeError, ('concatenation requires equal '
                                      'levels on the non-hinge dimensions')

        newdata = np.concatenate([b._data for b in bb], axis)
        return HyperBrick(newdata, newdims)


    def align(self, other,
              _pad=lambda d: di.Dimension._NullDimension(d[0]),
              _key=lambda t: t[0]):

        di00 = other.labels
        di10 = self.labels

        di01, di11 = al.align(di00, di10, key=_key, pad=_pad)
        def maybe_update(brick, olddims, newdims):
            return brick if olddims == newdims \
                   else HyperBrick(brick._data, newdims)

        return tuple(maybe_update(b, d0, d1)
                     for b, d0, d1 in zip((self, other),
                                          (di10, di00),
                                          (di11, di01)))
