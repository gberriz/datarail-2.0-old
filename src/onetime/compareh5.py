import sys
import numpy as np
from pdb import set_trace as ST

import bag as ba
import h5helper as h5h

OLD = 'icbp45.h5'
NEW = sys.argv[1]

def partial_enumerate(nda, k, predicate=None):
    if predicate is None:
        for ii in np.ndindex(nda.shape[:k]):
            yield ii, nda[ii]
    else:
        for ii in np.ndindex(nda.shape[:k]):
            if predicate(ii, nda[ii]):
                yield ii, nda[ii]

if True:
    cases = old, new = ba.Bag(path=OLD), ba.Bag(path=NEW)
    for c in old, new:
        with h5h.Hdf5File(c.path, 'r') as c.h5:
            grp = c.h5['confounders']
            c.keymap = h5h.read_keymap(grp)
            c.conf_hb = h5h.read_hyperbrick(grp['GF'])
            c.data_hb = h5h.read_hyperbrick(c.h5['from_IR/GF'])

            assert c.conf_hb.labels[:-1] == c.data_hb.labels[:-1]

    assert old.keymap == new.keymap
    keymap = old.keymap
    inverse_keymap = h5h.invert_keymap(old.keymap)

    assert old.conf_hb != new.conf_hb
    conf_hb = new.conf_hb

    assert old.data_hb.labels == new.data_hb.labels
    labels = old.data_hb.labels

    def ok(ii, old_ms, new_data=new.data_hb.data):
        return not (old_ms == new_data[ii]).all()

    def keycoords(indices, hb=old.data_hb):
        ks = [c[1][0] for c in hb._tolabels(ii)]
        ks[0] = ks[0][0]
        return tuple(ks)

    def valcoords(indices, hb=conf_hb, km=keymap):
        jj = conf_hb.data[indices]
        return tuple(d[c] for d, c in zip(km, jj))

    data = old.data_hb.data.copy()
    ct = 0
    for ii, ms in partial_enumerate(data, -1, ok):
        print ','.join(map(str, keycoords(ii)) + ['\t'] + map(str, valcoords(ii)))
        ct += 1

    if ct == 0:
        print 'all OK'
