import sys as sy
import types as ty
import numpy as np
import memoized as mm
import h5py

import traceback as tb

import dimension as di
import issequence as iss
import h5helper as h5h
import align as al

from pdb import set_trace as ST

class HyperBrickDimensions(tuple):
    def __init__(self, dims):
        self.__dict__ = dict((d.name, d) for d in dims)
        self.__indexlookup = dict((d.name, i) for i, d in enumerate(dims))

    def __call__(self, key):
        return self.__dict__[key]

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def index(self, key):
        return self.__indexlookup[key]

    def replace(self, key, value=None):
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
        as_list[self.index(key)] = value
        return type(self)(as_list)

    @property
    def names(self):
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
            ii = (pa if isinstance(pa, int)
                  else dimvals.index(pa))
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
                newdims = self._dims.replace(ls[0][0],
                                             sum((l[1] for l in ls), ()))
            else:
                if len(set(ls)) > 1:
                    raise TypeError, ('concatenation requires equal '
                                      'levels on the non-hinge dimensions')

        newdata = np.concatenate([b._data for b in bb], axis)
        return HyperBrick(newdata, newdims)


    def align(self, other,
              _pad=lambda d: (d[0], (di.Dimension.NULL_LEVEL,)),
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


def mu_sigma(box, axes, newdim=(u'stat', (u'mean', u'stddev'))):
    if not iss.issequence(axes):
        axes = axes.split()

    axes = tuple(map(box._toaxis, axes))
    inshape = box.shape
    nonaxes = tuple([i for i in range(len(inshape))
                     if i not in set(axes)])
    permuted = box._data.transpose(nonaxes + axes)

    newshape = tuple(inshape[i] for i in nonaxes) + (-1,)
    reshaped = permuted.reshape(newshape)

    outshape = newshape[:-1] + (1,)
    mean = reshaped.mean(axis=-1).reshape(outshape)
    std = reshaped.std(axis=-1).reshape(outshape)

    newdims = map(box._fromaxis, nonaxes) + [newdim]

    return HyperBrick(np.concatenate((mean, std), axis=-1), newdims)
                      
ICBP45H5 = '/home/gfb2/IR/icbp45.h5'
ICBP45PKL = '/home/gfb2/IR/icbp45.pkl'

def readpkl(_path=ICBP45PKL):
    import collections as co
    import cPickle as pkl

    with open(_path) as fh:
        bricks = pkl.load(fh)

    ret = co.defaultdict(dict)
    for subassay in 'GF', 'CK':
        for grpnm in 'from_IR', 'confounders':
            ret[subassay][grpnm] = HyperBrick(*bricks[subassay][grpnm])

    return ret


def extrude(base, newshape):
    extrusion = np.empty(newshape, dtype=base.dtype)
    extrusion[...] = base
    return extrusion


def propagate_controls(brick):
    sel_labels = dict(ligand_name=u'CTRL', stat=u'mean')
    proj_labels = dict(ligand_concentration=u'0', time=u'0')

    ctrl = brick(**sel_labels).squeeze()
    ms_projn, toss = mu_sigma(ctrl, proj_labels.keys()).align(brick)
    assert id(toss) == id(brick)
    del toss

    # create a new slab (mu_sigma_all) by replicating the mu_sigma slab
    #   along the selection dimensions
    origdims = brick._dims
    newdims = ms_projn._dims
    for dimname in sel_labels.keys():
        if dimname in origdims:
            newdims = newdims.replace(origdims(dimname))
        
    zero_mu_sigma = ms_projn.extrude(newdims)

    # sequentially replicate mu_sigma_all slab along all the mu_sigma
    #   projection labels (i.e. ligand_concentration and time), and merge the
    #   resulting slab with the existing brick
    ret = brick
    for dim, level in proj_labels.items():
        newdims = ret._dims.replace(dim, (level,))
        newslab = zero_mu_sigma.extrude(newdims)
        ret = newslab.concatenate(ret, dim=dim)

    return ret


def doit():
    brick = readpkl()['GF']['from_IR']
    fullbrick = propagate_controls(brick)
    return locals()


def __doit(_h5path=ICBP45H5):
    bricks = dict()
    openh5 = h5h.Hdf5File
    # openh5 = h5py.File
    with openh5(_h5path, 'r+') as h5:
        source = 'from_IR'
        target = h5.require_group('from_IR_w_zeros')
        for subassay in 'GF', 'CK':
            brick = h5['/'.join((source, subassay))]

            # must copy data into an in-memory array, otherwise the h5 file
            # can't be closed (and therefore it can't be re-opened either), nor
            # can the data be pickled.
            data = np.array(brick['data'])
            labels = h5h.load(brick['labels'].value)
            fullbrick = propagate_controls(HyperBrick(data, labels))
            h5h.write_hyperbrick(target.require_group(subassay), fullbrick)

    return locals()


if __name__ == '__main__':
    reload(di)
    reload(al)
    brick = readpkl()['GF']['from_IR']
    fullbrick = propagate_controls(brick)
    ST()
    pass




if False:
    """CRUFT"""
# import easydict as ez
# def setup(_h5path=ICBP45H5, _pklpath=ICBP45PKL):
#     import collections as co
#     global HBS0
#     HBS0 = co.defaultdict(dict)
#     h5 = h5py.File(_h5path, 'r+')
#     def _gethb(grpnm, subassay):
#         brick = h5['/'.join((grpnm, subassay))]

#         # must copy data into an in-memory array, otherwise the h5
#         # file can't be closed (and therefore it can't be re-opened
#         # either), nor can the data be pickled.
#         data = np.array(brick['data'])

#         labels = h5h.load(brick['labels'].value)
#         return data, labels

#     for subassay in 'GF', 'CK':
#         for grpnm in 'from_IR', 'confounders':
#             HBS0[subassay][grpnm] = _gethb(grpnm, subassay)

#     h5.close()
#     import cPickle as pkl
#     fh = open(_pklpath, 'w')
#     pkl.dump(HBS0, fh)
#     fh.close()
#     update()

# def update():
#     import collections as co
#     global HBS
#     HBS = co.defaultdict(dict)
#     global HBS0
#     for subassay in 'GF', 'CK':
#         for grpnm in 'from_IR', 'confounders':
#             HBS[subassay][grpnm] = HyperBrick(*HBS0[subassay][grpnm])
    
# import isinteractive as ii
# if ii.II:
#     setup()

# if __name__ == '__main__':
#     setup()
#     exit(0)
    
    #__name__ == '__main__':
    # databrick = _gethb('from_IR', subassay)
    # ctrlconf = confounders(ligand_name='CTRL', confounder='well')
    # ctrlbrick = databrick(ligand_name='CTRL')
    # ST()
    # pass

    # exit(0)

    # import random as rn
    # rn.seed(0)
    # data = np.ndarray(shape,
    #                   buffer=np.array([rn.random()
    #                                    for _ in np.ndindex(shape)]))

#     import pseudorepl as su

#     shape = 5, 4, 3
#     data = np.ndarray(shape,
#                       buffer=np.array(range(np.product(shape)), dtype='int32'),
#                       dtype='int32')
#     dimnames = tuple('month state color'.split())
#     months = tuple('Mar Apr May Jun Jul'.split())
#     states = tuple('AK HI MO NV'.split())
#     colors = tuple('black white orange'.split())
#     dimvals = months, states, colors
#     labels = zip(dimnames, dimvals)
#     hb = HyperBrick(data, labels)

#     ps = su.PseudoRepl(globals(), locals(), _indentlevel=2)

#     cases = ("('Apr', 'NV', 'orange')",
#              "('Apr', 3, 'orange')",
#              "('Apr', color='orange', state='NV')",
#              "(color='orange', state='NV', month='Apr')",
#              "('Apr', -3, 'orange')",
#              "('Apr', 10, 'orange')",
#              "('Apr', -10, 'orange')",
#              "('Apr', 'NV')",
#              "(None, 'NV')",
#              "('Sunday', color='white')",
#              "('Jun', color='white')",
#              "(slice('Apr', 'Jun'), color='white')",
#              "(2, month='May')",
#              "(2)",
#              "(2.0)",
#              "(month='Apr')",
#              "(state='MO')",
#              "(color='black')",
#              "(frobozz=0)",
#              "(color=0)",
#              "(color='banana')",
#              "('Apr', ('HI', 'NV'), 'orange')",
#              "('Apr', (slice('HI'), 'NV'), 'orange')",
#              "('Apr', color=('orange',), state=('HI', 'NV'))",
#             )

#     for m in ('._toindex', ''):
#         if not m:
#             continue
#         for case in cases:
#             src = 'hb%s%s' % (m, case)
#             ps.simulate(src)

#             if m != '._toindex':
#                 continue
#             try:
#                 idx = eval(src)
#             except:
#                 continue

#             src = 'hb._tolabels(%s)' % (idx,)
#             ps.simulate(src)

#             print

# # JUNK
#     # def _toslices(self, *args):
#     #     nargs = len(args)
#     #     ndim = len(self._dimnames)
#     #     if nargs > ndim:
#     #         raise ValueError('too many arguments')

#     #     def _ts(pa, nested=True):
#     #         if isinstance(pa, tuple):
#     #             return tuple(map(_ts, pa))

#     #         if isinstance(pa, slice):
#     #             if nested:
#     #                 raise ValueError, 'index expression contains '\
#     #                                   'a nested slice'
#     #             return pa
#     #         else:
#     #             li = len(dvi)
#     #             if isinstance(pa, int):
#     #                 if not -li <= pa < li:
#     #                     raise IndexError, 'dimension "%s": ' \
#     #                                       'index out of range (%d)' \
#     #                                       % (dni, pa)
#     #                 ii = pa + li if pa < 0 else pa
#     #                 ii1 = ii + 1
#     #                 if ii1 == li:
#     #                     ii1 = None
#     #             else:
#     #                 ii = pa
#     #                 jj = dvi.index(str(pa)) + 1
#     #                 if jj == li:
#     #                     ii1 = None
#     #                 else:
#     #                     ii1 = dvi[jj]

#     #             if nested:
#     #                 return ii

#     #             sss = (ii, ii + 1)

#     #         return slice(*sss)


    # import collections as co
    # path = '/home/gfb2/IR/icbp45.h5'
    # global HBS
    # HBS = co.defaultdict(dict)
    # with h5h.Hdf5File(path, 'r+') as h5:
    #     def _gethb(grpnm, subassay):
    #         brick = h5['/'.join((grpnm, subassay))]
    #         data = brick['data']
    #         labels = h5h.load(brick['labels'].value)
    #         return HyperBrick(data, labels)

    #     for subassay in 'GF', 'CK':
    #         for grpnm in 'from_IR', 'confounders':
    #             HBS[subassay][grpnm] = _gethb(grpnm, subassay)


