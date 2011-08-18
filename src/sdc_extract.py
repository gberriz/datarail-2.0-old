import os
import re
import numpy as np
import h5py
from operator import indexOf, itemgetter
from math import sqrt
from icbp45_utils import idx2plate, idx2rc
from functools import partial


def iterwells(h5h):
    for k, v in h5h['Children'].items():
        yield k, v


def iterfields(well):
    _err = ValueError('argument is not a valid well: %r' % (well,))
    try:
        children = well['Children']
    except:
        raise _err

    for key, field in children.items():
        try:
            field_id = field['Meta']['Field_ID'].value[0]
            field_idx = field_id[-1]
        except:
            raise _err
        assert key == field_idx, str((key, field_id, field_idx))
        yield field


def plate_well(well):
    return well['Meta']['Plate_Well']


def well_coords(well):
    plate_idx, well_idx = plate_well(well)
    return (idx2plate(plate_idx),) + idx2rc(well_idx)


def field_feature_values(field, wanted_feature_names=None):
    v = field['Data']['feature_values'].value
    if wanted_feature_names is None:
        return v
    else:
        n_x_1 = (len(v), 1)
        names = field_feature_names(field)
        return np.hstack([v[:, i].reshape(n_x_1) for i in
                          map(partial(indexOf, names),
                              wanted_feature_names)])


def field_feature_names(field):
    return field['Meta']['feature_names'].value


def well_stats(well, wanted_feature_names=None):
    matrix = np.vstack([field_feature_values(f, wanted_feature_names)
                        for f in iterfields(well)])
    return (matrix.mean(0, np.float64), matrix.std(0, np.float64))



if __name__ == '__main__':
    import doctest
    doctest.testmod()
