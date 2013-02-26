if True:
    import warnings as wrn
    import exceptions as exc
    wrn.filterwarnings(u'ignore',
                       message=r'.*with\s+inplace=True\s+will\s+return\s+None',
                       category=exc.FutureWarning,
                       module=r'pandas')

    import os.path as op
    import pandas as pd
    # from pandas import DataFrame as df, Series as se
    from pandas import Series as se
    # import matplotlib.pyplot as plt
    import numpy as np
    import math as ma
    import sys

    import re

    def readsheet(path, sheet=0):
        wb = pd.ExcelFile(path)
        if type(sheet) is int:
            return wb.parse(wb.sheet_names[sheet])
        else:
            return wb.parse(sheet)

    def _normalize_label(label,
                         _cleanup_re=re.compile(ur'\W+|(?<=[^\WA-Z_])'
                                                ur'(?=[A-Z])')):
        return (u'none_0' if label is None
                else _cleanup_re.sub(u'_', unicode(label).strip()).lower())

    class FloatLabel(unicode):
        _HEX_RE = re.compile(ur'^0x[a-f0-9]\.[a-f0-9]{13}p[-+](?:0|[1-9]\d*)$')
        def __new__(cls, flt):
            if isinstance(flt, FloatLabel):
                h = flt._hex
            else:
                try:
                    if (FloatLabel._HEX_RE.search(flt) and
                        float.fromhex(flt) != None):
                        h = flt
                    else:
                        raise '(goto except clause)'
                except:
                    h = float(flt).hex()
            return super(FloatLabel, cls).__new__(cls, h)

        @property
        def value(self):
            return float(self)

        @property
        def _hex(self):
            # guaranteed to always be displayed as the hex
            # representation, irrespective of context
            return float(self).hex()

        def __unicode__(self):
            # return u"'%f'" % float(self)
            return u"%f" % float(self)

        def __repr__(self):
            return "%r" % float(self)

        def __str__(self):
            return self

        def __float__(self):
            return float.fromhex(self)

    # automate definition of comparison methods for FloatLabel class
    def _setfunc(cls, name):
        ffunc = getattr(float, name)
        def func(s, o):
            try:
                return ffunc(float(s), float(cls(o)))
            except ValueError:
                return NotImplemented
        func.__name__ = name
        func.__doc__ = ffunc.__doc__
        setattr(cls, name, func)

    for _name in ('__%s__' % s for s in 'eq ne ge gt le lt'.split()):
        _setfunc(FloatLabel, _name)

    del _name, _setfunc



    def keep(df, labels, axis=0, level=None):
        drop = list(set(df.columns).difference(set(labels)))
        return df.drop(drop, axis=axis, level=level)

    class Replicates(pd.Series):
        def __new__(cls, data, keycolumns=None, base=1):
            reps = dict()
            ret = []
            idx = data.index
            if keycolumns is not None:
                data = keep(data, keycolumns, axis=1)
            start = base - 1
            for t in [tuple(v) for v in data.values]:
                r = reps[t] = reps.get(t, start) + 1
                ret.append(unicode(r))
            self = super(Replicates, cls).__new__(cls, ret, index=idx)
            self.__class__ = cls
            return self

        def __init__(self, data, keycolumns=None, base=1):
            super(Replicates, self).__init__(self)

    def maybe_to_int(x):
        try:
            f = float(x)
            i = int(round(f))
            if f != float(i):
                i = f
        except ValueError, e:
            i = x
        return unicode(i)

    basedir = op.abspath(op.join(op.dirname(__file__), u'..'))

    datadir = op.join(basedir, u'data')
    outputdir = op.join(basedir, u'dataframes/mgh')
    def tsv_path(name, _outputdir=outputdir):
        return op.join(_outputdir, u'%s.tsv' % name)

    default_basename = 'BreastLinesFirstBatch_MGHData_sent'
    # default_basename = 'bl1'
    filename = '%s.xlsx' % (sys.argv[1] if len(sys.argv) > 1
                            else default_basename)
    del default_basename

    print op.join(datadir, filename)
    print 'reading data...\t',; sys.stdout.flush()
    workbook = pd.ExcelFile(str(op.join(datadir, filename)))

    welldatamapped = workbook.parse(u'WellDataMapped')
    platedata = workbook.parse(u'PlateData')
    calibration = workbook.parse(u'RefSeedSignal', header=1, skiprows=[0],
                                 skip_footer=7)
    seeded = workbook.parse(u'SeededNumbers')

    # del workbook
    print 'done'

    for df in (welldatamapped, platedata, seeded):
        df.rename(columns=_normalize_label, inplace=True)

    def dropcols(df, colnames):
        return df.drop(colnames.split()
                       if hasattr(colnames, 'split')
                       else colnames, axis=1)

    seeded = dropcols(seeded, 'read_date cell_id')
    seeded.rename(columns={'cell_line':'cell_name'}, inplace=True)
    hmssfx_re = re.compile(ur'_HMS$')
    seeded.cell_name = seeded.cell_name.apply(lambda s: hmssfx_re.sub('', s))

    import datetime as dt
    def fix_barcode(b):
        try:
            return unicode(dt.datetime.strptime(b, u'%Y-%m-%d %I:%M:%S %p'))
        except ValueError, e:
            if 'does not match format' not in str(e):
                raise
            return b

    # fix_barcode = dict([(b, b) for b in seeded.barcode])
    # fix_barcode.update(((u'2012-10-24 5:46:33 PM', u'2012-10-24 17:46:33'),
    #                     (u'2012-10-31 3:20:16 PM', u'2012-10-31 15:20:16')))
    # seeded.barcode = [fix_barcode[b] for b in seeded.barcode]
    seeded.barcode = seeded.barcode.apply(fix_barcode)

    # def fix_barcode(b):
    #     if b == u'2012-10-24 5:46:33 PM': return u'2012-10-24 17:46:33'
    #     if b == u'2012-10-31 3:20:16 PM': return u'2012-10-31 15:20:16'
    #     return b

    calibration.rename(columns={calibration.columns[0]:
                                    _normalize_label(calibration.columns[0])},
                    inplace=True)
    fixre=re.compile(ur'^MCFDCIS\.COM$')
    calibration.rename(columns=lambda l: fixre.sub(u'MCF10DCIS.COM', l),
                       inplace=True)
    del fixre

    sd1 = calibration.set_index(calibration.columns[0])

    calibration.set_index(calibration.columns[0], inplace=True)
    calibration = pd.DataFrame(calibration.stack()
                               .swaplevel(0, 1)
                               .sortlevel(), columns=[u'signal'])
    calibration.index.names[0] = u'cell_name'

    def reg(df, index=pd.Index(('coeff', 'intercept'))):
        xcol = 'seed_cell_number_ml'
        ycol = 'signal'
        sdf = df.sort(columns=[xcol], axis=0)
        ls = pd.ols(x=sdf[xcol][2:], y=sdf[ycol][2:])
        ret = ls.beta
        ret.index = index
        return ret

    coeff = (calibration.reset_index().groupby('cell_name').apply(reg)
             .reset_index())

    seeded = pd.merge(seeded, coeff, on='cell_name', how='outer')

    seeded['estimated_seeding_signal'] = np.round(seeded.intercept +
        seeded.seeding_density_cells_ml * seeded.coeff)

    def dropna(df):
        return df.dropna(axis=1, thresh=len(df)//10).dropna(axis=0, how='all')

    welldatamapped.rename(columns={u'compound_no': u'compound_number',
                                   u'compound_conc': u'compound_concentration'},
                          inplace=True)

    platedata.time = platedata.protocol_name.apply(lambda s: s[-4])
    platedata.barcode = platedata.barcode.apply(unicode)

    platedata = keep(platedata, [u'barcode', u'time'], axis=1)

    # data0 = welldatamapped.dropna(axis=1, how='all')
    # data0 = data0.drop(u'none_5', axis=1)
    data0 = dropna(welldatamapped)
    data0 = data0[data0[u'sample_code'] != 'BDR']

    data0 = dropcols(data0, u'cell_id well_id modified created')

    for c in 'compound_number sample_code column'.split():
        data0[c] = data0[c].apply(maybe_to_int)

    for c in data0.columns.drop(['signal']):
        data0[c] = data0[c].apply(unicode)

    data = pd.merge(data0, platedata, on='barcode')

    # data.compound_number = data.compound_number.apply(lambda x: unicode(int(x)))
    # data.compound_concentration = data.compound_concentration.apply(FloatLabel)

    def log10(s):
        f = float(s)
        return (u'-inf' if f == 0.0 else
                unicode(round(ma.log10(f), 1)))

    data.compound_concentration = data.compound_concentration.apply(log10)

    del log10

    data.rename(columns={u'compound_concentration':
                            u'log10_compound_concentration'},
                inplace=True)
                                      
    barcodes = set(data.barcode)

    # it'd be nice to have an "extract" method that combines both of the following:
    compound_0 = data[data.compound_number == '0']
    compound_0.reset_index(inplace=True, drop=True)
    assert len(compound_0) > 0
    assert set(compound_0.barcode) == barcodes
    assert (compound_0.log10_compound_concentration == '-inf').all()

    data = data[data.compound_number != '0']
    data.reset_index(inplace=True, drop=True)
    assert len(data) > 0
    assert len(data[data.sample_code == 'CRL']) == 0
    assert len(data[data.sample_code == 'BL']) == 0
    assert (data.log10_compound_concentration != '-inf').all()


    data = pd.merge(data,
                    keep(seeded, [u'barcode', u'estimated_seeding_signal'],
                         axis=1),
                    on=u'barcode', how='left')


    controls = compound_0[compound_0.sample_code == 'CRL']
    controls.reset_index(inplace=True, drop=True)
    assert set(controls.barcode) == barcodes
    background = compound_0[compound_0.sample_code != 'CRL']
    background.reset_index(inplace=True, drop=True)
    set(background.barcode) == barcodes
    assert all(background.sample_code == 'BL')
    assert len(controls) + len(background) == len(compound_0)

    del compound_0

    def subtract_background(df,
                            bg=(background.groupby(u'barcode')[u'signal']
                                .agg(u'mean std'.split()))):
        ret = df.set_index(u'barcode')
        ret[u'signal'] -= bg[u'mean']

        ret.insert(list(ret.columns).index(u'signal') + 1,
                   u'bg_std',
                   ret.join(bg['std'])['std'])
        
        ret.reset_index(inplace=True)
        return ret, ret[ret.signal < 0]

    data1, negdata = subtract_background(data)
    controls1, negcrls = subtract_background(controls)

    comparison1 = (controls.groupby([u'barcode',
                           lambda c: (u'2' if controls.ix[c][u'column'] == '2'
                                      else '12,13')]).mean().unstack().reset_index())

    # comparison1 = (
    #     controls.groupby([u'barcode',
    #                       lambda c: (u'2' if controls.ix[c][u'column'] == '2'
    #                                  else '12,13')])
    #             .mean().unstack().reset_index())

    glob = globals()
    for name in (u'data data0 data1 platedata controls1 background '
                 'negdata negcrls comparison1'.split()):
        df = glob[name]
        df.to_csv(tsv_path(name), '\t', index=False,
                  float_format='%.1f')

    def filterobj(test, obj):
      return obj[test(obj)]

    # filterobj(lambda x: x["mean"] >= 0,
    #           gb.xs(u'R', level=1) - gb.xs(u'L', level=1))
              

if False:
    multikeycols = (u'cell_name compound_number compound_concentration time '
                    '__replicate__'.split())

    data.insert(len(multikeycols) - 1,
                multikeycols[-1],
                Replicates(data, keycolumns=multikeycols[:-1]))

    data.set_index(multikeycols, inplace=True)
                   
    # import pandas.core.common as com
    # import pandas.hashtable as htable
    # k, c = htable.value_count_object(cc, com.isnull(cc))
    # result = se(c, index=k)
    # from pdb import set_trace as ST
    # ST()
    # s = repr(result)

if False:
    import os, time, select, pty
         
    verbose = bool(os.environ.get("TEST_VERBOSE"))
         
    #def runWithTimeout(cmd, timeout):
    def runWithTimeout(callback, timeout):
        # args = cmd.split()
        pid, fd = pty.fork();
        startTime = time.time()
        endTime = startTime + timeout
         
        if pid == 0:
            #os.execvp(args[0], args)
            callback()
            exit(0)

        output = ""
        while True:
            timeleft = endTime - time.time()
            if timeleft <= 0:
                break
            i, o, e = select.select([fd], [], [], timeleft)
            if fd in i:
                try:
                    str = os.read(fd, 1)
                    output += str
                except OSError, e:
                    exitPid, status = os.waitpid(pid, os.WNOHANG)
         
                    if exitPid == pid:
                        if verbose:
                            print "Child exited with %i" % status
                        return status, output
         
        if verbose:
            print "Command timed out: killing pid %i" % pid
         
        os.kill(pid, signal.SIGINT)
        raise Exception("Command execution time exceeded %i seconds" % timeout,
                        outputSoFar=output)

    # import sys
    # s = set((float(u'nan'), float(u'nan')))
    # import numpy as np
    # s = set((np.nan, np.nan))
    # print 'ok'

    import re
    # nows = re.compile(u'\S')
    # data = data.drop([c for c in data
    #                   if all([x is None or not nows.match(str(x))
    #                           for x in data[c].dropna()])], 1)

    # y = data[u'None.5']

    # yk = y.keys()

    # ykt = tuple(yk)

    # print 'ok'

    # yks = set(ykt)

    # print len(yks)

    # yv = y.values
    # yvc = y.value_counts()

    # yvs = set(yv)
    # print "yvs ok"

    # yvcs = set(yvc)
    # print "yvcs ok"

    # print len(yvs)
    # print len(yvcs)

    x = data[None]

    # xk = x.keys()

    # xkt = tuple(xk)

    # print 'ok'

    # xks = set(xkt)

    # print len(xks)

    # xvc = x.value_counts()
    # print 'xvc ok'
    # print len(xvc)

    # xvcs = set(xvc)
    # print "xvcs ok"
    # print len(xvcs)

    xv = x.values
    print 'xv ok'
    print len(xv)

    xvt = tuple(xv)
    print 'xv ok'
    print len(xvt)

    import signal as sig
    def handler(s, f):
        print 'signal %s received' % s
        raise Exception(u'timeout')

    sig.signal(sig.SIGALRM, handler)

    left, right = 0, len(xv)
    maxlen = right
    minfailed = right
    timeout = 30
    timeout = 600
    n = None
    xvs = None
    def callback():
        global xvs
        xvs = set(xv[:n])

    while left < right:
        n = (left + right)/2
        # n = right

        print left, n, right
        l = n
        starttime = time.time()
        print 'about to call runWithTimeout'
        try:
            sig.alarm(timeout)
            runWithTimeout(callback, timeout)
        except Exception, e:
            if not 'timeout' in str(e):
                raise
            print "length = %d FAILED" % n
            if minfailed > n:
                minfailed = n
            right = n
        else:
            elapsed = time.time() - starttime
            d = elapsed/l
            print "length = %d ok (%.0f, %.1g, %.0f)" % (l, elapsed, d, d*maxlen)
            left = max(n, left+1)

    left, right = 0, minfailed
    while left + 1 < right:
        n = (left + right)/2
        print left, n, right
        l = n - left
        print 'about to call runWithTimeout'
        try:
            sig.alarm(1)
            runWithTimeout(callback, timeout)
        except Exception, e:
            if not 'timeout' in str(e):
                raise
            print "length = %d FAILED" % l
            if minfailed > right:
                minfailed = right
            left = n
        else:
            elapsed = time.time() - starttime
            d = elapsed/l
            print "length = %d ok (%.0f, %.1g, %.0f)" % (l, elapsed, d, d*maxlen)
            right = n

