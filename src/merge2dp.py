# -*- coding: utf-8 -*-
from __future__ import division
# from pdb import set_trace as ST
# from pprint import pprint as pp

import sys
import os
import os.path as op
import h5py
import sdc_extract
import traceback as tb
import numpy as np
import csv
import platform
from copy import deepcopy
from math import sqrt
from walk import find
from icbp45_utils import read_exp_design, parse_antibody_string, is_valid_rc
from collections import namedtuple, defaultdict
from quantiles import median_iqr

def _get_exp_design(path,
                    _val=namedtuple(u'_val', u'treatment info'),
                    _treatment=namedtuple(u'_treatment',
                                          u'ligand_name ligand_conc'
                                          u'entration time antibodies'),
                    _antibody=namedtuple(u'_antibody',
                                         u'target species wavelength'),
                    _info=namedtuple(u'_info',
                                     u'replicate coords zone')):

                    # _key=namedtuple('_key', ('plate', 'well'))):
                                    
    ks = []
    replicate = defaultdict(int)
    platemap = defaultdict(lambda: defaultdict(dict))
    for record in read_exp_design(path):
        plate = record.plate
        try:
            well = record.well.upper()
            column = well[1:]
        except AttributeError:
            row = record.row
            column = record.column
            well = '%s%02d' % (row.upper(), int(column))

        # key = _key(plate, well)

        coords = (plate, well)
        zone = (plate if plate.startswith('GF')
                else plate + ('-L' if (int(column) - 1) < 6
                              else '-R'))
        ks.append(coords)

        time_ = record.time__min_
        ligand_name = record.ligand
        ligand_conc = record.concentration__ng_ml_
        primary_abs = (parse_antibody_string(record.primary_body1),
                       parse_antibody_string(record.primary_body2))
        secondary_abs = (parse_antibody_string(record.secondary_body1),
                         parse_antibody_string(record.secondary_body2))


        abs_ = []
        ws = set()
        for primary, secondary in zip(primary_abs, secondary_abs):
            ab_target, ab_species = primary
            assert ab_species == secondary[0]
            wavelength = secondary[2]
            assert wavelength not in ws
            ws.add(wavelength)
            abs_.append((wavelength,
                         _antibody(*(primary + (wavelength,)))))

        treatment = _treatment(ligand_name, ligand_conc, time_,
                               tuple(abs_))

        repl = replicate[treatment]
        replicate[treatment] += 1
        info = _info(repl, coords, zone)

        try:
            assert repl == int(record.replicate)
        except AttributeError:
            pass

        assert well not in platemap[plate]
        platemap[plate][well] = _val(treatment, info)

    assert len(ks) == len(set(ks))
    return dict(platemap)


def gimme(root, wanted):
    try:
        for f in find(root, wanted):
            yield f
        return
    except ValueError, e:
        if not str(e).startswith('not a directory'):
            raise
        with open(root) as h:
            for line in h:
                yield line.rstrip('\n')


def _encode_entries(rec):
    assert hasattr(rec, '__iter__')
    return type(rec)(s.encode('utf-8') if isinstance(s, unicode) else s
                     for s in rec)


def _process_payload(payload, params):
    payload.sort(key=lambda mci: int(mci.info.replicate))
    cell_lines, treatments, infos, matrices = zip(*payload)
    assert len(set(cell_lines)) == 1
    cell_line_ = cell_lines[0]

    zone = params['zone']
    ligand = params['ligand']

    assert len(set(treatments)) == 1
    treatment_ = treatments[0]

    nreplicates_ = unicode(len(matrices))
    assert ((treatment_.ligand_name == 'CTRL' and int(nreplicates_) == 6)
            or int(nreplicates_) == 1)

    #coords_ = u'+'.join(u'%s:%s' % i.coords for i in infos)
    coords_ = u'+'.join(u':'.join(c) for i in infos for c in i.coords)

    matrices = sum(matrices, [])
    lens = map(len, matrices)
    data = np.vstack(matrices)
    ncells_ = len(data)
    assert sum(lens) == ncells_

    nfields_ = len(lens)
    if nfields_ > 1:
        mean_cells_per_field_ = mu = ncells_/nfields_
        var = (sum([x**2 for x in lens]) - nfields_ * mu**2)/(nfields_ - 1)
        cells_per_field_stddev_ = sqrt(var)
        ncells_ = u'%d=%s' % (ncells_,
                              u'+'.join(map(unicode, lens)))
    else:
        assert nfields_ == 1
        ncells_ = mean_cells_per_field_ = lens[0]
        cells_per_field_stddev_ = u'(n/a)'

    cells_per_field_median_, cells_per_field_iqr_ = median_iqr(lens)
    mean_and_std = data.mean(0, np.float64), data.std(0, np.float64)

    wanted_features = params['wanted_features']
    wanted_wavelengths = params['wanted_wavelengths']
    wl2wl = params['wl2wl']

    if len(wanted_features) > 1:
        mean_and_std = zip(*mean_and_std)

    pfx = dict((k, u'%s-%s (%s)' %
                (ab.target, ab.species[0], ab.wavelength))
                for k, ab in dict(treatment_.antibodies).items())
    readouts_ = dict()
    for feature, wavelength, vals in zip(wanted_features,
                                         wanted_wavelengths,
                                         mean_and_std):
        assert wavelength in feature
        p = pfx[wl2wl[wavelength]]
        assert p not in readouts_
        readouts_[p] = vals

    assert len(pfx.keys()) == len(readouts_.keys())
    signature_ = ':'.join(pfx[k] for k in sorted(pfx.keys(), key=int))

    return dict((k[:-1], v) for k, v in locals().items() if k.endswith('_'))
                

# def _flatten(tt):
#     ret = []
#     for t in tt:
#         if hasattr(t, '__iter__'):
#             ret += _flatten(t)
#         else:
#             ret.append(t)
#     return tuple(ret)


def _path_to_csv(subpath, basename):
    if not basename.endswith('.csv'):
        basename += '.csv'
    return op.join(subpath, basename)


def _print_as_datapflex(cell_line_data, print_headers=False,
                        wl2wl={'530': '488', '685': '647'},
                        wanted_wavelengths=None,
                        wanted_features=None,
                        outfh=sys.stdout,
                        outpath=os.getcwd(),
                        terminator=None):


    if not (outpath is None or outpath == '-'):
        outfh = None
                        
    if cell_line_data is None:
        return

#     for tr in filter(lambda t: t.ligand_name == 'CTRL', cell_line_data.keys()):
#         # k = ':'.join(_flatten(tr[3:])).encode('utf8')
#         k = tr[3:]
#         assert k not in collect
#         collect[k] = []
#         print k, len(cell_line_data[tr])

#     for tr in filter(lambda t: t.ligand_name != 'CTRL', cell_line_data.keys()):
#         # k = ':'.join(_flatten(tr[3:])).encode('utf8')
#         collect[tr[3:]].append(tr[:3])

    collect = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for zone, zonedict in cell_line_data.items():
        for ligand, liganddict in zonedict.items():
            for conc, concdict in liganddict.items():
                for time_, payload in concdict.items():
                    collect[zone][ligand][conc][time_] = \
                        _process_payload(payload, locals())


    bytreatment = defaultdict(lambda: defaultdict(list))
    for zone, zonedict in collect.items():
        ctrl = zonedict['CTRL']
        assert len(ctrl.keys()) == 1 and len(ctrl.values()[0].keys()) == 1
        ctrl_v = ctrl.values()[0].values()[0]
        ctrl_treatment = ctrl_v['treatment']
        times = None
        for ligand, liganddict in ([kv for kv in zonedict.items()
                                    if kv[0] != 'CTRL'] +
                                   [(u'CTRL', zonedict[u'CTRL'])]):
            if ligand == 'CTRL':
                assert times is not None
                assert len(times) > 0
                assert times.pop(0)
                conc_0_dict = liganddict.values()[0]
            else:
                for conc, concdict in liganddict.items():
                    concdict[u'0'] = v = deepcopy(ctrl_v)
                    v['treatment'] = v['treatment']._replace(ligand_name=ligand,
                                                             ligand_concentration=conc)

                times = sorted(liganddict.values()[0].keys(), key=int)
                liganddict[u'0'] = conc_0_dict = dict()

            assert times is not None
            for t in times:
                assert t not in conc_0_dict
                conc_0_dict[t] = v = deepcopy(ctrl_v)
                v['treatment'] = v['treatment']._replace(ligand_name=ligand, time=t)


            d = bytreatment[zone[:2]]
            for conc, concdict in liganddict.items():
                for time_, prerow in concdict.items():
                    d[(ligand, conc, time_)].append(prerow)

    stat_suffixes = (u'', u'=stdev')
    nstats = len(stat_suffixes)
    blanks = (u'',) * nstats
    global READOUTS
    global HEADERS

    start_h = tuple(['cell_line',
                     'ligand_name',
                     'ligand_concentration',
                     'time'])

    middle_h = tuple([
                      'coords',
                      'nreplicates',
                      'nfields',
                      'ncells',
                      'mean_cells_per_field',
                      'cells_per_field_stddev',
                      'cells_per_field_median',
                      'cells_per_field_iqr',
                      u'', u''])

    count = 0

    batches = dict()
    for zoneclass, zoneclassdict in bytreatment.items():
        row_dict = dict()
        cell_line = None
        first = True
        for prerows in zoneclassdict.values():
            readouts = tuple(sorted(sum([v['readouts'].keys()
                                         for v in prerows], [])))
            try:
                assert readouts == READOUTS[zoneclass]
            except (NameError, KeyError), e:
                if isinstance(e, NameError):
                    READOUTS = dict()
                    HEADERS = dict()
                READOUTS[zoneclass] = readouts

            headers_undefined = zoneclass not in HEADERS

            last_prefix = None
            for prerow in prerows:

                prefix = tuple([prerow[k] if k in prerow
                                else prerow['treatment']._asdict()[k]
                                for k in start_h])

                if cell_line is None:
                    cell_line = prefix[0]
                else:
                    assert cell_line == prefix[0]

                if last_prefix is not None:
                    assert last_prefix == prefix
                    last_prefix = prefix

                middle = tuple([u'' if k == u'' else prerow[k]
                                for k in middle_h])
                              
                coords = prerow['coords']

                ros = []
                rdict = prerow['readouts']
                found = 0
                for ro in readouts:
                    try:
                        stats = rdict[ro]
                        assert len(stats) == nstats
                        found += 1
                    except KeyError:
                        stats = blanks

                    ros += stats

                assert found > 0

                if headers_undefined:
                    finish_h = tuple(sum(([ro + st]
                                          for ro in readouts
                                          for st in stat_suffixes), []))
                    HEADERS[zoneclass] = (start_h +
                                          tuple(u'' for _ in middle_h) +
                                          finish_h)

                key = (-10 * ord(zoneclass[0]),
                       coords, prefix[1], int(prefix[2]), int(prefix[3]))

                assert not key in row_dict
                row_dict[key] = prefix + middle + tuple(ros)
                if first:
                    key_h = (key[0] - 1, u'')
                    assert key_h < key
                    row_dict[key_h] = HEADERS[zoneclass]

                count += 1

                first = False

        batches[(cell_line, zoneclass)] = (row_dict)

    for k, row_dict in batches.items():
        writer = csv.writer((outfh or
                             open(_path_to_csv(outpath, '_'.join(k)), 'w')),
                            lineterminator=(terminator
                                            if terminator is not None
                                            else '\r\n'
                                            if platform.system() == 'Windows'
                                            else '\n'))

        writer.writerows([_encode_entries(row_dict[k])
                          for k in sorted(row_dict.keys())])


def main(argv=sys.argv):
    global READOUTS
    READOUTS = dict()
    global HEADERS
    HEADERS = dict()
    global COUNT
    COUNT = 0
    global OUTH
    OUTH = sys.stdout

    root = argv[1]
    platemap = _get_exp_design(argv[2])
    try:
        outpath = argv[3]
    except IndexError:
        outpath = None

    try:
        terminator = '\r\n' if argv[4].lower() == 'windows' else '\n'
    except IndexError:
        terminator = None

    payload = namedtuple('_payload',
                         'cell_line treatment info data')
    
    wl2wl={'530': '488', '685': '647'}
    wanted_wavelengths = sorted(wl2wl.keys())
    wanted_features = tuple(['Whole_w%s (Mean)' % s
                             for s in wanted_wavelengths])
    
    last_cell_line = None
    print_headers = True
    cell_line_data = headers = None
    done = False

    def wanted(bn, path, isdir):
        return isdir and is_valid_rc(bn)

    well_paths = gimme(root, wanted)

    while True:
        try:
            well_path = well_paths.next()
        except StopIteration:
            well_path = None

        if well_path:
            cell_line, plate, rc = (well_path.split('/'))[-3:]
            treatment, info = platemap[plate][rc]
            coords = info.coords
            r, c = rc[0], rc[1:]
        else:
            done = True

        if done or cell_line != last_cell_line:
            _print_as_datapflex(cell_line_data,
                                print_headers,
                                wl2wl, wanted_wavelengths,
                                wanted_features,
                                outpath=outpath,
                                terminator=terminator)
            if done:
                break
            last_cell_line = cell_line
            print_headers = False
            cell_line_data = defaultdict(lambda:
                               defaultdict(lambda:
                                 defaultdict(lambda:
                                   defaultdict(list))))

        fields_used = []
        data = []
        for sdc in sorted(os.listdir(well_path)):
            if not sdc.endswith('.sdc'):
                continue

            hdf = op.join(well_path, sdc, 'Data.h5')
            try:
                with h5py.File(hdf, 'r') as h5:
                    wells = list(sdc_extract.iterwells(h5))
                    if not len(wells):
                        # TODO: log warning
                        continue
                    flds = list(sdc_extract.iterfields(wells[0][1]))
                    assert len(flds) == 1, str((hdf, len(flds)))
                    data.append(sdc_extract.field_feature_values(flds[0],
                                                                 wanted_features))
                    fieldno = sdc[-5]
                    fields_used.append(unicode(fieldno))
            except IOError:
                tb.print_exc()
            finally:
                try:
                    h5.close()
                except:
                    pass

        info = info._replace(coords=tuple([coords + (fn,)
                                           for fn in fields_used]))

        zone = info.zone
        ligand = treatment.ligand_name
        conc = treatment.ligand_concentration
        time_ = treatment.time
        cell_line_data[zone][ligand][conc][time_].append(payload(unicode(cell_line),
                                                                 treatment, info, data))

    return 0


if __name__ == '__main__':
    exit(main(sys.argv))

