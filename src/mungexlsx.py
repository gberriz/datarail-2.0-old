import os
import os.path as op

prog = op.basename(__file__)
datadir = op.abspath(op.join(op.dirname(__file__), '..', 'data'))
assert op.isdir(datadir)
src = op.join(datadir, 'BreastLinesFirstBatch_MGHData_sent.xlsx')
assert op.isfile(src)
tgt = op.join(datadir, 'bl1.xlsx')
del datadir

import platform as pl
if pl.system() != 'Windows':
    import sys
    print >> sys.stderr, ('%s: unsupported platform: %s... exiting' %
                          (prog, pl.system()))
    sys.exit(1)

import win32com.client as win32
import collections as co

xl = win32.gencache.EnsureDispatch('Excel.Application')
try:
    wb = xl.Workbooks.Open(src)
except Exception, e:
    xl.Application.Quit()
    raise

_record = co.namedtuple('_record',
                        u'barcode cell_id cell_name well_id row column '
                        u'sample_code compound_number compound_concentration '
                        u'signal modified created '
                        u'none_0 none_1 none_2 none_3 none_4 none_5')

wanted_cells = set(['HCC-1954', 'HCC-202', 'BT-20'])
wanted_compounds = set([0, 5])

wanted_concentrations = set([0.3162277660168379, 1.0, 0.0, 10.0,
                             3.162277660168379])

wanted_barcodes = set([u'10/24/12 17:46:33'] +
                      u'B13_1422_00173240 B13_1422_00180590 B13_1422_00183430 '
                      u'B13_1437_00173340 B13_1437_00180690 B13_1437_00183530 '
                      u'B13_1438_00173350 B15_1422_00173460 B15_1422_00177050 '
                      u'B15_1422_00180780 B15_1422_00183630 B15_1437_00177160 '
                      u'B15_1437_00180880 B15_1437_00183730 B15_1438_00173570 '
                      u'B17_1422_00177710 B17_1437_00177820'.split())

def addsheet(wb, rows, name):
    wsnew = wb.Sheets.Add()
    wsnew.Name = name
    h = len(rows)
    w = len(rows[0])
    wsnew.Range(wsnew.Cells(1, 1), wsnew.Cells(h, w)).Value = rows
    wsnew.Columns.AutoFit()

def addrange(sheet, rows):
    h = len(rows)
    w = len(rows[0])
    sheet.Range(sheet.Cells(1, 1), sheet.Cells(h, w)).Value = rows
    sheet.Columns.AutoFit()

if True:
    xl.Visible = True
    keep = set()

    wsname = 'WellDataMapped'
    ws = wb.Sheets(wsname)
    rng = ws.UsedRange
    new = []
    for i, row in enumerate(rng.Value):
        if i > 0:
            rec = _record(*row)
            if not ((rec.cell_name in wanted_cells) and
                    (rec.compound_number in wanted_compounds) and
                    (unicode(rec.barcode) in wanted_barcodes) and
                    (rec.compound_concentration
                     in wanted_concentrations)):
            # if not (rec.cell_name in wanted_cells and
            #         rec.compound_number in wanted_compounds):
            #         rec.compound_concentration
            #         in wanted_concentrations):
            # if not (rec.cell_name in wanted_cells):
                continue
        new.append(row)
        keep.add(i)

    rng.Delete()
    addrange(ws, new)

    wsname = 'WellData'
    ws = wb.Sheets(wsname)
    rng = ws.UsedRange
    new = []
    for i, row in enumerate(rng.Value):
        if i in keep:
            new.append(row)

    #addsheet(wb, new, 'new_%s' % wsname)
    #addsheet(wb, new, '%s' % wsname)

    rng.Delete()
    addrange(ws, new)

    try:
        os.unlink(tgt)
    except WindowsError, e:
        if 'The system cannot find the file specified' not in str(e):
            raise

    wb.SaveAs(tgt, win32.constants.xlOpenXMLWorkbook)

    # xl.Application.Quit()
