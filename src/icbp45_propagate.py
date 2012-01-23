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

    return hb.HyperBrick(np.concatenate((mean, std), axis=-1), newdims)

                      
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
            brick = h5['/'.join((source, subassay))]

            # must copy data into an in-memory array, otherwise the h5 file
            # can't be closed (and therefore it can't be re-opened either), nor
            # can the data be pickled.
            data = np.array(brick['data'])
            labels = h5h.load(brick['labels'].value)
            fullbrick = propagate_controls(hb.HyperBrick(data, labels))
            h5h.write_hyperbrick(target.require_group(subassay), fullbrick)

    return 0

if __name__ == '__main__':
    import sys
    exit(main(sys.argv))
