import numpy as np

import issequence as iss
import h5helper as h5h
import hyperbrick as hb

from pdb import set_trace as ST

def _parseargs(argv):
    h5path, = argv[1:]
    d = dict()
    l = locals()
    params = 'h5path'
    for p in params.split():
        d[p] = l[p]
    _setparams(d)


def _setparams(d):
    global PARAM
    try:
        pd = PARAM.__dict__
        pd.clear()
        pd.update(d)
    except NameError:
        class _param(object): pass
        PARAM = _param()
        _setparams(d)


def _mu_sigma(nda, axes):
    """Compute mean and stddev over all the specified axes.

    nda:     an (n-dimensional) numpy ndarray;
    axes:    a k-tuple of integers, representing the dimensions over
             which to carry out the averaging;
    returns: an (n - k + 1)-dimensional numpy ndarray, whose first n-k
             dimensions are the same as the dimensions of nda (in
             their original ordering) that were not specified in the
             axes parameter, and whose last dimension has length 2;
             the hyperslab corresponding to the first and second
             elements of this last dimension are, respectively, the
             computed means and stddevs.

    >>> inshape = 4, 2, 4, 3, 4
    >>> inbuf = np.array(map(float, range(np.product(inshape))))
    >>> inbox = np.ndarray(inshape, buffer=inbuf)
    >>> outbox = _mu_sigma(inbox, tuple(range(len(inshape))[1::2]))

    >>> assert all(outbox[..., 1].ravel() ==
                   [inbox[0, :, 0, :, 0].std()] * outbox[..., 1].size)
    >>> assert all(outbox[..., 0].ravel() == [float(4*(v + 3*w) + x)
                                              for v in [8*y - 1
                                                        for y in [3*z + 1
                                                                  for z in range(4)]]
                                              for w in range(4)
                                              for x in range(4)])
    """

    # get a tuple of integers describing the shape of hbrick;
    inshape = nda.shape

    # from this tuple extract those integers that are not present in
    # the (current) axes variable;
    nonaxes = tuple([i for i in range(len(inshape))
                     if i not in set(axes)])

    # at this point the tuples axes and nonaxes should be disjoint,
    # and their union should contain all the integers from 0 to
    # (d - 1) where d is the number of nda's dimensions

    # the next statement permutes the dimensions of nda, so that
    # those whose indices are in the nonaxes tuple come first (aside
    # from this, the relative ordering of dimensions within each of
    # the nonaxes and axes tuples remains unchanged);
    permuted = nda.transpose(nonaxes + axes)

    # the tuple in newshape consists of the lengths of the nonaxes
    # dimensions followed by -1 (a special value that numpy interprets
    # as "the rest of the dimension indices"); therefore, the
    # subsequent call to reshape, effectively, *collapses* all of the
    # original hyperbrick's axes dimensions in into a single
    # dimension;
    newshape = tuple(inshape[i] for i in nonaxes) + (-1,)
    reshaped = permuted.reshape(newshape)

    # the outshape tuple contains the lengths of the dimensions in
    # nonaxes followed by 1; this is the shape of the "hyperslab"
    # produced by projecting the original hyperbrick down along the
    # (collapsed) axes dimensions; the purpose of the outshape tuple
    # is described below;
    outshape = newshape[:-1] + (1,)

    # the next two statements compute hyperslabs corresponding to the
    # mean and standard deviation obtained by projecting along the
    # axes dimensions (now collapsed into the array's last dimension,
    # and referred to below by the "axis=-1" argument); the calls to
    # reshape(outshape) are needed to recover the hyperslab's
    # "trivial" (i.e. length-one) dimension, since it is discarded by
    # both numpy.mean and numpy.stddev (in numpy 2.x these functions
    # have an optional keepdims keyword to suppress this behavior);
    mean = reshaped.mean(axis=-1).reshape(outshape)
    std = reshaped.std(axis=-1).reshape(outshape)

    # the next statement concatenates the two hyperslabs along their
    # last dimension; the reason for the two calls to reshape above is
    # have the dimension along which to carry out this concatenation;
    return np.concatenate((mean, std), axis=-1)



def mu_sigma(hbrick, axes, newdim=(u'stat', (u'mean', u'stddev'))):
    """Compute mean and stddev over all the specified axes.

    hbrick:  an (n-dimensional) HyperBrick object;
    axes:    either a k-tuple of integers or strings, or a string
             describing a space-separated list of k dimension names or
             dimension indices (integers);
    newdim:  a pair whose first element should be a string and second
             element a 2-tuple of strings; this pair serves as the
             specification (name and levels) for the last dimension of
             the returned HyperBrick;
             default: (u'stat', (u'mean', u'stddev'))
    returns: a HyperBrick object, whose _data attribute contains the
             ndarray computed by _mu_sigma(hbrick._data, indices)
             (where indices holds the integer equivalents of the axes
             specified in the axes parameter); the first n-k
             dimensions of the returned HyperBrick are the same as the
             dimensions of hbrick that were not specified in the axes
             parameter, and whose last dimension is a 2-level
             dimension; the hyperslab corresponding to the first and
             second levels of this last dimension are, respectively,
             the computed means and stddevs; the ordering of the first
             n-k (i.e. the kept) dimensions is the same as it was in
             hbrick.
    """

    # the next line allows us to pass several named dimensions as a
    # single space-separated string.
    if not iss.issequence(axes): axes = axes.split()

    # we convert (as necessary) the dimensions specified in axes to
    # numerical indices;
    axes = tuple(map(hbrick._toaxis, axes))

    data = _mu_sigma(hbrick._data, axes)
    newdims = [hbrick._fromaxis(i)
               for i in range(hbrick.ndim)
               if i not in set(axes)] + [newdim]

    return hb.HyperBrick(data, newdims)

                      
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
            newdims = newdims._replace(origdims(dimname))
        
    zero_mu_sigma = ms_projn.extrude(newdims)

    # sequentially replicate mu_sigma_all slab along all the mu_sigma
    #   projection labels (i.e. ligand_concentration and time), and merge the
    #   resulting slab with the existing brick
    ret = brick
    for dim, level in proj_labels.items():
        newdims = ret._dims._replace(dim, (level,))
        newslab = zero_mu_sigma.extrude(newdims)
        ret = newslab.concatenate(ret, dim=dim)

    return ret


def main(argv):
    _parseargs(argv)

    bricks = dict()
    openh5 = h5h.Hdf5File
    with openh5(PARAM.h5path, 'r+') as h5:
        source = 'from_IR'
        target = h5.require_group('from_IR_w_zeros')
        for subassay in 'GF', 'CK':
            brick = h5h.read_hyperbrick(h5['/'.join((source, subassay))])
            fullbrick = propagate_controls(brick)
            h5h.write_hyperbrick(target.require_group(subassay), fullbrick)

    return 0


def _test():
    import numpy as np
    inshape = 4, 2, 4, 3, 4
    inbuf = np.array(map(float, range(np.product(inshape))))
    inbox = np.ndarray(inshape, buffer=inbuf)
    outbox = _mu_sigma(inbox, tuple(range(len(inshape))[1::2]))
    assert all(outbox[..., 1].ravel() ==
               [inbox[0, :, 0, :, 0].std()] * outbox[..., 1].size)
    assert all(outbox[..., 0].ravel() == [float(4*(v + 3*w) + x)
                                          for v in [8*y - 1 for y in
                                                    [3*z + 1
                                                     for z in range(4)]]
                                          for w in range(4)
                                          for x in range(4)])
    return 0


if __name__ == '__main__':
    import sys
    exit(main(sys.argv))
    # exit(_test())

