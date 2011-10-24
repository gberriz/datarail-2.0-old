import os
import os.path as op
import re
import numpy as np
import h5py
from operator import indexOf, itemgetter
from math import sqrt
from icbp45_utils import idx2plate, idx2rc, scrape_coords
from functools import partial
from find import find


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

class NoDataError(Exception): pass

def _extract_field_data(path, wanted_features=None):
    # NOTE: this function is not at all general; it assumes that the
    # HDF5 file at path contains data for exactly one field of exactly
    # one well.
    with h5py.File(path, 'r') as h5:
        wells = list(iterwells(h5))
        nwells = len(wells)
        if nwells == 0:
            raise NoDataError()
        assert nwells == 1, 'found %d wells; expected 1' % nwells # see NOTE above
        # assert len(wells)
#         if not len(wells):
#             # TODO: log warning
#             continue
        flds = list(iterfields(wells[0][1]))
        assert len(flds) == 1 # see NOTE above
        # name = flds[0].name
        return field_feature_values(flds[0], wanted_features)

def _extract_well_data(path, wanted_features=None, _basename='Data.h5'):
    paths = find(path,
                 lambda b, d, isdir:
                 not isdir and b == _basename)
    return _extract_wells_data(paths, wanted_features)


def _extract_wells_data(paths, wanted_features=None):
    nds = dict()
    for path in sorted(set(paths)):
        coords = scrape_coords(path)
        try:
            nds[coords] = _extract_field_data(path, wanted_features)
        except NoDataError:
            pass

    if not nds:
        raise NoDataError()

    return nds


def inspect_nd(d):
    for a in [x for x in dir(d)
              if not x.startswith('__') and
                 not x in set(['data', 'dumps', 'tolist', 'tostring'])]:
        v0 = getattr(d, a)
        try:
            v = v0() if hasattr(v0, '__call__') else v0
        except Exception, e:
            #help(v0)
            v = 'ERROR: %s' % str(e)
        print '%s: %s' % (a, v)


def inspect_thing(d,
                  keep=lambda x: True,
                  skip=lambda x: x.startswith('__') and x.endswith('__'),
                  comb=lambda x, y: x and y):
    for a in [x for x in dir(d) if comb(keep(x), not skip(x))]:
        v0 = getattr(d, a)
        try:
            v = v0() if hasattr(v0, '__call__') else v0
        except Exception, e:
            #help(v0)
            v = 'ERROR: %s' % str(e)
        print '%s: %s' % (a, v)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
