# -*- coding: utf-8 -*-
from __future__ import division
import os
from os.path import join as opjoin, exists
import h5py
import sdc_extract
import traceback as tb
import re
import numpy as np
from functools import partial

# from findbin import findbin
# from find import find
from walk import find
from icbp45_utils import read_exp_design, parse_antibody_string
from noclobberdict import NoClobberDict
from collections import namedtuple, defaultdict

def _get_exp_design(path,
                    _val=namedtuple('_val',
                                    'ligand_name ligand_concentration time '
                                    'antibody_0 antibody_1 coords'.split())):
                    # _key=namedtuple('_key', ('plate', 'well'))):
                                    
    ret = NoClobberDict()
    ks = []
    ret = defaultdict(dict)
    for record in read_exp_design(path):
        plate = record.plate

        row = record.row
        column = record.column
        well = '%s%02d' % (row.upper(), int(column))
        # key = _key(plate, well)
        ks.append((plate, well))

        time = record.time__min_
        ligand_name = record.ligand
        ligand_concentration = record.concentration__ng_ml_
        ab0 = parse_antibody_string(record.primary_body1)
        ab1 = parse_antibody_string(record.primary_body2)
        ret[plate][well] = _val(ligand_name, ligand_concentration,
                                time, ab0, ab1, plate + well)

    assert len(ks) == len(set(ks))
    return dict(ret)


def get_exp_design(path):
    return _get_exp_design(path)


def gimme(root, wanted):
    try:
        for f in find(root, wanted):
            yield opjoin(f + '.sdc', 'Data.h5')
        return
    except ValueError, e:
        if not str(e).startswith('not a directory'):
            raise
        with open(root) as h:
            for line in h:
                yield line.rstrip('\n')


def main(argv):
    root = argv[1]
    an_mod = __import__(argv[2])
    an = an_mod.Analysis()

    fieldnos = set('1234')
    def wanted(bn, path, isdir):
        return isdir and bn in fieldnos

    count = 0
    skipset = set()
    done = set()

    def error_in(path, tbck, skip=False):
        if (path, tbck) in done:
            return
        done.add((path, tbck))
        if skip:
            assert not path in skipset, path
            skipset.add(path)
        import sys
        print >> sys.stderr, 'ERROR in %s:\n%s' % (path, tbck)

    for pass_ in an.passes:
        for hdf in gimme(root, wanted):
            cell_line, plate, rc, sdcdir = (hdf.split('/'))[-5:-1]
            fieldno = sdcdir[0]
            index = '%s_%s_%s_%s' % (cell_line, plate, rc, fieldno)
#             plate, rc = (hdf.split('/'))[-4:-2]
            r, c = rc[0], rc[1:]
            try:
                with h5py.File(hdf, 'r') as h5:
                    wells = list(sdc_extract.iterwells(h5))
                    w = None

                    try:
                        w = wells[0][1]
                        assert len(wells) == 1
                    except:
                        wisnone = w is None
                        error_in(hdf, tb.format_exc(), skip=wisnone)
                        if wisnone:
                            continue

                    try:
                        assert (plate, r, c) == sdc_extract.well_coords(w)
                    except:
                        error_in(hdf, tb.format_exc())

                    try:
                        means, stddevs = sdc_extract.well_stats(w)
                        assert (len(means), len(stddevs)) == (26, 26), \
                               repr((len(means), len(stddevs)))
                    except:
                        error_in(hdf, tb.format_exc())

                    flds = list(sdc_extract.iterfields(w))
                    f = None
                    try:
                        f = flds[0]
                        assert len(flds) == 1, repr((hdf, len(flds)))
                    except:
                        fisnone = f is None
                        error_in(hdf, tb.format_exc(), skip=fisnone)
                        if fisnone:
                            continue

                    an.process_rows(sdc_extract.field_feature_values(f), index, pass_)
            except SystemExit:
                raise
            except:
                error_in(hdf, tb.format_exc())
            finally:
                try:
                    h5.close()
                except:
                    pass
                    #print >> sys.stderr, 'oops: %s' % tb.format_exc()

#             count += 1
#             if count >= 5:
#                 break

        an.finish(pass_)

    an.finish()
    an.report()




if __name__ == '__main__':
    import sys
    exit(main(sys.argv))
