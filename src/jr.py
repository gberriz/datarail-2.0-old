if True:
    import sys
    # print sys.path
    # sys.path.insert(0, '/home/berriz/@/Work/Sites/hmslincs/src')
    # import shell_utils as shu

    import warnings as wrn
    import exceptions as exc
    wrn.filterwarnings(u'ignore',
                       message=r'.*with\s+inplace=True\s+will\s+return\s+None',
                       category=exc.FutureWarning,
                       module=r'pandas')

    import os.path as op
    import pandas as pd
    # pd.options.display.line_width=104
    import readdf as rdf
    reload(rdf)

    # from pandas import DataFrame as df, Series as se
    from pandas import Series as se
    # import matplotlib.pyplot as plt
    import numpy as np
    import math as ma


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

    class Bag(object):
        # the use of ____ instead of 'self' as the first argument in __init__'s
        # signature is meant to avoid conflicts with a possible 'self' keyword in
        # kwargs (a very poor solution to the problem).
        def __init__(____, **kwargs):
            ____.__dict__.update(**kwargs)

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



    class Replicates(pd.Series):
        def __new__(cls, data, keycolumns=None, base=1):
            reps = dict()
            ret = []
            idx = data.index
            if keycolumns is not None:
                data = data.ix[:, keycolumns]
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

    # datadir = op.join(basedir, u'3caseStudy_130210')
    # datadir = op.join(basedir, u'Data_SameerJeremie_130221')
    datadir = op.join(basedir, u'data')
    outputdir = op.join(basedir, u'dataframes/jr')
    # shu.mkdirp(outputdir)

    def tsv_path(name, _outputdir=outputdir):
        return op.join(_outputdir, u'%s.tsv' % name)


    default_basename = 'Data_Jeremie'
    # default_basename = 'Data_Jeremie1000'; rdf.OPTS.iterator_chunksize = 5000
    # default_basename = 'Data_Jeremie20'; rdf.OPTS.iterator_chunksize = 1000

    filename = '%s.tsv' % (sys.argv[1] if len(sys.argv) > 1
                           else default_basename)

    print op.join(datadir, filename)
    print 'reading data...\t',; sys.stdout.flush()

    data0 = rdf.read_dataframe(op.join(datadir, filename))

    glob = globals()
    for name in (u'data0'.split()):
        df = glob[name]
        df.to_csv(tsv_path(name), '\t', index=False,
                  float_format='%.3f')



if False:
    def read_int(s):
        return unicode(int(round(float(s))))

    def read_float(s):
        return unicode(float(s))

    def metadata_reader(type_, _gl=globals()):
        return _gl.get('read_%s' % type_, unicode)

    # opts = Bag(iterator_chunksize=3)
    opts = Bag(iterator_chunksize=10000)

    def df_from_iter(iterator, chunksize=opts.iterator_chunksize, **kwargs):
        first = next(iterator)
        stride = max((chunksize//
                      len(''.join(unicode(s) for s in first))), 1)
        chunks = []
        batch = [first] + list(it.islice(iterator, stride - 1))
        while batch:
            chunks.append(pd.DataFrame(batch, **kwargs))
            batch = list(it.islice(iterator, stride))

        return pd.concat(chunks).reset_index(drop=True)

    def mkprocessor(converters, delim=u'\t'):
        def process_line(line):
            return tuple(c(s.strip()) for c, s in
                         zip(converters,
                             unicode(line[:-1]).split(delim)))
        return process_line

    def mkiter(fh, converters):
        process_line = mkprocessor(converters)
        return (process_line(l) for l in fh)


    nptype = dict(int='int32', float='float64', unicode='<U10')

    _hrecord = co.namedtuple(u'_hrecord', u'name type category process')
    with open(op.join(datadir, filename)) as fh:
        columns = []
        converters = []
        # dtype = []
        for line in fh:
            if line.startswith(u'#'): continue
            if line.startswith(u'*'): break

            name, type_, category = unicode(line).split()[:3]
            columns.append(name)

            if category == u'metadata':
                proc = metadata_reader(type_)
                # dtype.append((str(name), nptype['unicode']))
            else:
                assert category == u'data'
                # dtype.append((str(name), nptype[type_]))
                proc = eval(type_)

            converters.append(proc)
            # columns.append(_hrecord(name, type_, category, proc))

        data0 = df_from_iter(mkiter(fh, converters), columns=columns)


        # chunks = []
        # nlines = 0
        # # stride = 4000
        # # stride = 600000
        # stride = 500000
        # while True:
        #     lines = []
        #     n = 0
        #     while n < stride:
        #         try:
        #             lines.append(fh.next())
        #         except:
        #             break
        #         n += 1
        #     if not n: break
        #     chunks.append(pd.DataFrame(np.array([process_line(line)
        #                                          for line in lines],
        #                                         dtype=dtype)))
        #     nlines += len(lines)
        #     print nlines

        # lines = list(fh)
        # fh.close()
        # tuples = [process_line(l) for l in lines]
        # _record = co.namedtuple('_record', [c.name for c in columns])
        # records = [_record(*t) for t in tuples]
        # nprecs = np.array(tuples, dtype=dtype)

if True:
    print 'done'


if False:
    workbook = pd.ExcelFile(op.join(datadir, filename))

    welldatamapped = workbook.parse(u'WellDataMapped')
    platedata = workbook.parse(u'PlateData')
    seeddata = workbook.parse(u'RefSeedSignal', header=1, skiprows=[0],
                              skip_footer=7)

    # del workbook
    print 'done'

    for df in (welldatamapped, platedata):
        df.rename(columns=_normalize_label, inplace=True)


    seeddata.rename(columns={seeddata.columns[0]:
                               _normalize_label(seeddata.columns[0])},
                    inplace=True)
    fixre=re.compile(ur'^MCFDCIS\.COM$')
    seeddata.rename(columns=lambda l: fixre.sub(u'MCF10DCIS.COM', l),
                    inplace=True)
    del fixre

    seeddata.set_index(seeddata.columns[0], inplace=True)
    seeddata.columns.names[0] = 'cell_name'
    seeddata = seeddata.stack().swaplevel(0, 1).sortlevel()


    def dropna(df):
        return df.dropna(axis=1, thresh=len(df)//10).dropna(axis=0, how='all')

    welldatamapped.rename(columns={u'compound_no': u'compound_number',
                                   u'compound_conc': u'compound_concentration'},
                          inplace=True)

    platedata[u'time'] = platedata.protocol_name.apply(lambda s: s[-4])
    platedata = platedata.ix[:, [u'barcode', u'time']]

    # data0 = welldatamapped.dropna(axis=1, how='all')
    # data0 = data0.drop(u'none_5', axis=1)
    data0 = dropna(welldatamapped)
    data0 = data0[data0[u'sample_code'] != 'BDR']
    data0 = data0[data0.columns.drop(u'cell_id well_id modified '
                                     u'created'.split())]
    for c in 'compound_number sample_code column'.split():
        data0[c] = map(maybe_to_int, data0[c])

    data = pd.merge(data0, platedata, on='barcode')

    # data.compound_number = map(lambda x: unicode(int(x)), data.compound_number)
    # data.compound_concentration = map(FloatLabel, data.compound_concentration)

    def neglog10(f):
        return (u'inf' if f == 0.0 else
                unicode(-round(ma.log10(f), 1)))
    data.compound_concentration = map(neglog10,
                                      data.compound_concentration)
    del neglog10

    data.rename(columns={u'compound_concentration':
                            u'neg_log10_compound_concentration'},
                inplace=True)
                                      
    barcodes = set(data.barcode)

    # it'd be nice to have an "extract" method that combines both of the following:
    compound_0 = data[data.compound_number == '0']
    compound_0.reset_index(inplace=True, drop=True)
    assert len(compound_0) > 0
    assert set(compound_0.barcode) == barcodes
    assert (compound_0.neg_log10_compound_concentration == 'inf').all()

    data = data[data.compound_number != '0']
    data.reset_index(inplace=True, drop=True)
    assert len(data) > 0
    assert len(data[data.sample_code == 'CRL']) == 0
    assert len(data[data.sample_code == 'BL']) == 0
    assert (data.neg_log10_compound_concentration != 'inf').all()

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

