from __future__ import division
import re
import numpy as np
from operator import itemgetter
from collections import namedtuple
from itertools import product

from sdc_analysis import Analysis as superclass

class Analysis(superclass):
    """
Coordinate_X              (1022.0, 0.0),                    	
Coordinate_Y              (1022.0, 0.0),				
Cyto_w460 (Integrated)    (215367664.0, 0.0),			
Cyto_w460 (Mean)          (64929.8789, 0.0),			
Cyto_w530 (Integrated)    (940253184.0, -128099096.0),		
Cyto_w530 (Mean)          (58773.6172, -8309.9043),		
Cyto_w685 (Integrated)    (574348224.0, 0.0),			
Cyto_w685 (Mean)          (64807.832, -7949.20752),		
Nucleus_w460 (Integrated) (2063257340.0, -1837519620.0),		
Nucleus_w460 (Mean)       (65127.4805, 3954.68994),		
Nucleus_w530 (Integrated) (2083038720.0, -2052217340.0),		
Nucleus_w530 (Mean)       (56817.0234, -4116.99023),		
Nucleus_w685 (Integrated) (1532248830.0, -2080841220.0),		
Nucleus_w685 (Mean)       (62624.543, -7776.32715),		
Ratio_nuc/cyt_w460        (49.1386337, -8.7982378),		
Ratio_nuc/cyt_w530        (894977.0, -609279.0),			
Ratio_nuc/cyt_w685        (12795905.0, -102268928.0),		
Size_cyto                 (90445.0, 0.0),			
Size_nucleus              (139100.0, 11.0),			
Size_whole                (149583.0, 11.0),			
Whole_w460 (Integrated)   (2082018050.0, -1778621820.0),		
Whole_w460 (Mean)         (65127.4805, 1063.09961),		
Whole_w530 (Integrated)   (2083038720.0, -2041875070.0),		
Whole_w530 (Mean)         (52409.0703, -3679.90625),		
Whole_w685 (Integrated)   (1532248830.0, -2076459780.0),		
Whole_w685 (Mean)         (61718.582, -7755.56152)
"""

    _feature_names = ('Coordinate_X,Coordinate_Y,Cyto_w460 (Integrated),Cyto_w4'
                      '60 (Mean),Cyto_w530 (Integrated),Cyto_w530 (Mean),Cyto_w'
                      '685 (Integrated),Cyto_w685 (Mean),Nucleus_w460 (Integrat'
                      'ed),Nucleus_w460 (Mean),Nucleus_w530 (Integrated),Nucleu'
                      's_w530 (Mean),Nucleus_w685 (Integrated),Nucleus_w685 (Me'
                      'an),Ratio_nuc/cyt_w460,Ratio_nuc/cyt_w530,Ratio_nuc/cyt_'
                      'w685,Size_cyto,Size_nucleus,Size_whole,Whole_w460 (Integ'
                      'rated),Whole_w460 (Mean),Whole_w530 (Integrated),Whole_w'
                      '530 (Mean),Whole_w685 (Integrated),'
                      'Whole_w685 (Mean)'.split(','))
    _wavelengths = ('460', '530', '685')
    _compartments = ('CYTO', 'NUCLEUS', 'WHOLE')

    def __init__(self):
        super(Analysis, self).__init__()
        self._mins = []
        self._minabs = []
        self._maxs = []
        self._idxs = []
        # self._oddmost = []

        _fix_re = re.compile(r'\W')
        self._fnames = tuple(_fix_re.sub('_', s).strip('_').upper()
                             for s in Analysis._feature_names)



        #print row


    def table__function__0__minmax(self, rows, index):
        self._idxs.append(index)

        cols = namedtuple('_cols', self._fnames)(*rows.transpose())

        n = len(rows)
        j0 = sorted(enumerate(cols.SIZE_WHOLE),
                    key=lambda pr: pr[1])[n//2][0]
            
        size = cols.SIZE_WHOLE

        assert (cols.SIZE_CYTO >= 0).all()
        assert (cols.SIZE_NUCLEUS >= 0).all()
        assert (cols.SIZE_WHOLE == cols.SIZE_CYTO + cols.SIZE_NUCLEUS).all()

        _bg = np.vectorize(lambda i, m, s:
                           i/s - m if i > 0 and s > 0 else np.nan)

        bgs = []
        cd = cols._asdict()
        range_n = xrange(n)

        for w in ['W%s' % w for w in Analysis._wavelengths]:
#             cintg = cd['CYTO_%s__INTEGRATED' % w]
#             nintg = cd['NUCLEUS_%s__INTEGRATED' % w]
#             wintg = cd['WHOLE_%s__INTEGRATED' % w]

#             if (wintg < cintg).any():
#                 k = wintg < cintg
#                 for c in Analysis._compartments:
#                     print (cd['%s_%s__INTEGRATED' % (c, w)][k],
#                            cd['%s_%s__MEAN' % (c, w)][k],
#                            cd['SIZE_%s' % c][k],
#                            cd['%s_%s__MEAN' % (c, w)][k] *
#                            cd['SIZE_%s' % c][k])
#                 exit(0)

#             assert (wintg >= cintg).all()
#             assert (wintg >= nintg).all()

            intg = np.vstack([cd['%s_%s__INTEGRATED' % (c, w)]
                              for c in ('CYTO', 'NUCLEUS')])
            maxintg = intg.max() + 1
            k = np.vectorize(lambda x: x if x > 0 else maxintg)(intg).argmin(0)
            intg = intg[k, range_n]
            mean = np.vstack([cd['%s_%s__MEAN' % (c, w)]
                              for c in ('CYTO', 'NUCLEUS')])[k, range_n]
            size = np.vstack([cd['SIZE_%s' % c]
                              for c in ('CYTO', 'NUCLEUS')])[k, range_n]

            bgv = _bg(intg, mean, size)
            nans = np.isnan(bgv)
            if nans.any():
                bgv[nans] = bgv[~nans][0]
            assert not np.isnan(bgv).any()
            bgs.append(bgv)

        bgs = np.array(bgs).transpose()

        u = []
        for c in Analysis._compartments:
            for w in ['W%s' % w for w in Analysis._wavelengths]:
                u.append(cd['%s_%s__MEAN' % (c, w)] * cd['SIZE_%s' % c])

        u = np.array(u).transpose()

        rows = np.hstack((rows, u, bgs))

#         intg = np.array(filter(lambda x: int(x) % 2 > 0,
#                                np.fabs(np.hstack(
#                                    [cd['WHOLE_W%s__INTEGRATED' % w]
#                                     for w in Analysis._wavelengths]))))

#         self._oddmost.append(intg.max())

#         self._mins.append(np.hstack([rows.min(0), bgs]))
#         self._minabs.append(np.hstack([np.fabs(rows).min(0), np.fabs(bgs)]))
#         self._maxs.append(np.hstack([rows.max(0), bgs]))

#         self._mins.append(rows.min(0))
#         self._minabs.append(np.fabs(rows).min(0))
#         self._maxs.append(rows.max(0))

        vars_ = bgs.var(0)
        extra = np.hstack((vars_, np.array([float(n)])))
        self._mins.append(np.hstack([rows.min(0), extra]))
        self._minabs.append(np.hstack([np.fabs(rows).min(0), extra]))
        self._maxs.append(np.hstack([rows.max(0), extra]))


#     def test__1(self, row):
#         pass


    def finish(self, pass_=None):
        if pass_ == 0:
            l = lambda m: m.argmin(0)
            u = lambda m: m.argmax(0)

            #l = lambda m: m.argmin(0), m.min(0)
            #u = lambda m: m.argmax(0), m.max(0)

            i = self._idxs

#             k = np.array(self._oddmost).argmax()
#             print 'oddmost_integrated\t%s' % ((self._oddmost[k], i[k]),)
#             print

            pps = [Analysis._feature_names +
                   ['%s_W%s__INTEGRATED_CORRECTED' % (c, w)
                    for c, w in product(Analysis._compartments,
                                        Analysis._wavelengths)] +
                   list(sum(zip(*(['background_w%s' % w,
                                   'background_w%s_var' % w]
                             for w in Analysis._wavelengths)), ())) +
                   ['number_of_cells']]

            def _fmt(x):
                x = float(x)
                if str(x).endswith('.0'):
                    return str(int(round(x, 0)))
                elif abs(x) < 0.01:
                    return '%0.3g' % x
                else:
                    return str(round(x, 2))                    

            for f, d in zip((l, l, u),
                            (self._mins, self._minabs, self._maxs)):

                m = np.array(d)
                pps.append([','.join([_fmt(v[j]), i[j]])
                            for v, j in zip(m.transpose(),
                                            map(int, f(m)))])

            for r in zip(*pps):
                print '\t'.join(map(str, r))

            return
        elif pass_ == 1:
            return
        elif pass_ is None:
            return
        else:
            assert False


    def report(self):
        print 'reporting...'


