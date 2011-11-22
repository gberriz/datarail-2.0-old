from __future__ import division
from toycube import *
import os
import pickle
import math
import re
from numbers import Number
from operator import add
# DON'T U COMMIT THIS!
import graphics
try:
    reload(graphics)
except:
    pass
from graphics import *
from pprint import pprint as pp
from infinity import infinity

from pdb import set_trace as ST

VFUDGE = 1.02
INPUTPATH = os.environ['HOME'] + '/_/prj/datarail/g/PRIV/dumps/pickle/alexopoulos10_fig2_data.cube.pkl'

def rect2line(rect):
    (l, r), (b, t) = rect.bounds
    return Line(((l, b), (l, t), (r, t), (r, b), (l, b)))


def mk_labels(grid, labels, offset=None):

    if offset is None:
        offset = 0.02 * min(grid.rect.size)
    if isinstance(offset, Number):
        offset = (offset, offset)
    try:
        offset = Point(offset)
        assert len(offset) == 2
    except ValueError, AssertionError:
        raise TypeError('the offset parameter must be either a positive number '
                        'or a 2-d Point or convertible to a 2-d Point.')

    if not filter(None, labels):
        raise ValueError('no labels provided')

    (supercolnames, rownames,
     colnames, rowmaxes) = (tuple(labels) +
                            ((None,) * max(0, 4 - len(labels))))

    supercolcenters = grid.colcenters

    (t_labels, l_labels, b_labels, r_labels) = ret = ([], [], [], [])

    supercols = grid.members
    rect = Interval2D((supercols[0].rect[0], supercols[-1].rect[1]))

    top = rect.bounds[1][1]

    for i, sr in enumerate(supercols):
        basepoint = Point((supercolcenters[i], top + offset[0]))
        if not supercolnames is None:
            t_labels.append(Text(supercolnames[i], basepoint, (0, -1)))
            # t_labels.append(Text('(%s, %s)' % (basepoint), basepoint, (0, -1)))

        supercol = sr.region
        trans = sr.transformation

        rect_abs = supercol.rect.canonical().transform(trans)
        rect_abs = Interval2D((rect_abs[0] - offset,
                               rect_abs[1] + offset))

        bottom = rect_abs.bottom

        if not colnames is None:
            for j, c in enumerate(supercol.colcenters):
                x = Point((c, 0)).transform(trans)[0]
                basepoint = Point((x, bottom))
                b_labels.append(Text(colnames[j], basepoint, (1, 0), (0, 1)))
                # b_labels.append(Text('(%s, %s)' % (basepoint), basepoint, (1, 0), (0, 1)))

        if 0 < i < grid.ncols - 1:
            continue

        if i == 0:
            if rownames is None: continue
            x, a, lbls, lst = rect_abs.left, (1, 0), rownames, l_labels
        else:
            if rowmaxes is None: continue
            x, a, lbls, lst = rect_abs.right, (-1, 0), rowmaxes, r_labels

        for j, c in enumerate(supercol.rowcenters):
            y = Point((0, c)).transform(trans)[1]
            basepoint = Point((x, y))
            # lst.append(Text('(%s, %s)' % (basepoint), basepoint, a))
            lst.append(Text(lbls[j], basepoint, a))

    ret[-1].append(Mma("Red"))
    rect = Interval2D((rect[0] - offset, rect[1] + offset))
    ret[-1].append(rect2line(rect))

    return tuple(tuple(lst) for lst in ret)


def _mean_signal(data):
    return sum(data)/len(data)


def _log_mean_ratio(a, b):
    return math.log10((_mean_signal(a) + 1)/(_mean_signal(b) + 1))


def _log_max_ratio(a, b):
    return math.log10((max(a) + 1)/(max(b) + 1))


def _logistic(t):
    return 1/(1 + math.exp(-t))

def _sigmoid(t):
    return 2*_logistic(t) - 1


class RGB(tuple):
    def __new__(cls, seq):
        if hasattr(seq, '__iter__') and not hasattr(seq, 'len'):
            seq = list(seq)
        if not (hasattr(seq, '__iter__') and
                len(seq) == 3 and all([isinstance(x, Number) for x in seq])):
            raise ValueError('argument must be an iterable of 3 '
                             'real numbers in the interval [0, 1]')
        return super(RGB, cls).__new__(RGB, seq)

    def __mul__(self, scalar):
        return type(self)(scalar*x for x in self)

    def __rmul__(self, scalar):
        return type(self)(x*scalar for x in self)

    def __add__(self, other):
        return type(self)(u + v for u, v in zip(self, other))

    def __radd__(self, other):
        return other.__add__(self)

    def __sub__(self, other):
        return self + (-1) * other


def color_interpolator(rng, rgbs):
    nv = len(rgbs)
    assert len(rng) == 2 and rng[0] < rng[1] and nv > 1

    assert all(0 <= v <= 1 for v in sum(rgbs, ()))
    min_, max_ = rng
    delta = (max_ - min_)/(nv - 1)
    def fxn(t):
        if not min_ <= t <= max_:
            raise ValueError("argument (%s) outside of "
                             "interporlator's range" % t)
        if t == max_:
            return 1.0 * rgbs[-1]

        i, y = divmod(t - min_, delta)
        i = int(i)
        x = y/delta
        return (1 - x) * rgbs[i] + x * rgbs[i + 1]

    return fxn


def mma_rgb(r, g, b):
    return Mma('RGBColor[%s, %s, %s]' % (r, g, b))

def bgcolor_datarail_1_4(signal, reference):
    # Relatio  =  log(max(signalplot)/max(refplot))+1;
    try:
        index = math.log10(max(signal)/max(reference)) + 1;
    except ZeroDivisionError:
        index = infinity

    # if Relatio<1% Red if Ref>Signal
    #     if Relatio<0
    #         ColorBack=[1 0.4 0.4];
    #     else
    #         ColorBack=[1 Relatio Relatio];
    #     end
    #     backplot=[refplot(1:end-2) MaxYPlot MaxYPlot];
    # elseif Relatio>1 %blue if Ref<Signal
    #     Relatio=1/Relatio;
    #     if Relatio<0
    #         ColorBack=[ 0.4 0.4 1];
    #     elseif Relatio>1
    #         ColorBack=[1 0.4 0.4];
    #     else
    #         ColorBack=[Relatio Relatio 1];
    #     end
    # end

    if index < 1:
        if index < 0:
            rgb = (1, 0.4, 0.4)
        else:
            rgb = (1, index, index)
    elif index > 1:
        index = 1/index
        if index < 0:
            rgb = (0.4, 0.4, 1)
        elif index > 1:
            rgb = (1, 0.4, 0.4)
        else:
            rgb = (index, index, 1)
    else:
        #raise Exception('internal error')
        #rgb = (1, 1, 1)
        #rgb = (1, 0.5, 0)
        rgb = (0, 0, 0)

    return rgb


# QUESTION: why did this (fxn redefining itself) fail???
# def bgcolor_datarail_2_0(signal, reference):
#     sortkey = lambda x: x[0][0][1]
#     maxabs = -1
#     sig, ref = 'PriHu', 'HepG2'
#     lines = set([sig, ref])
#     for q in the_cube.iterate_over('cytokine', 'inhibitor', 'readout:name'):
#         data = {}
#         for curve in q.iterate_over('cell line'):
#             tf = curve._tupleform()
#             cell_line = curve.attribute_values(('cell line',))[0][0]
#             assert cell_line in lines
#             data[cell_line] = tuple([d[1] for d in sorted(tf, key=sortkey)])

#         maxabs = max(maxabs, abs(_log_max_ratio(data[sig], data[ref])))

#     rng = _logistic(-maxabs), _logistic(maxabs)

#     # global to_color
#     to_color = color_interpolator(rng, (RGB([1, 0, 0]),
#                                         RGB([1, 1, 1]),
#                                         RGB([0, 0, 1])))

#     import sys
#     print >> sys.stderr, "done computing to_color"
#     global bgcolor_datarail_2_0
#     bgcolor_datarail_2_0 = \
#         lambda s, r: mma_rgb(*to_color(_logistic(_log_max_ratio(s, r))))
#     print >> sys.stderr, "returning first color directive"
#     return bgcolor_datarail_2_0(signal, reference)


to_color = None
def bgcolor_datarail_2_0(signal, reference):
    global to_color
    if to_color is None:
        sortkey = lambda x: x[0][0][1]
        maxabs = -1
        sig, ref = 'PriHu', 'HepG2'
        lines = set([sig, ref])

        for _, q in the_cube.iterate_over(('cytokine', 'inhibitor',
                                           'readout:name')):
            data = {}
            for cell_line, curve in q.iterate_over('cell line'):
                tf = curve._tupleform()
                assert cell_line in lines
                data[cell_line] = tuple([d[1] for d in sorted(tf, key=sortkey)])

            maxabs = max(maxabs, abs(_log_max_ratio(data[sig], data[ref])))

        rng = _logistic(-maxabs), _logistic(maxabs)

        to_color = color_interpolator(rng, (RGB([1, 0, 0]),
                                            RGB([1, 1, 1]),
                                            RGB([0, 0, 1])))

    return to_color(_logistic(_log_max_ratio(signal, reference)))


def fgcolor(signal, top):
    # this is the DR1.4 scheme for assigning colors to "discretized"
    # signal values:
    # green sustained [0 1 1]
    # yellow transient [0 1 0]
    # magenta late [0 0 1]
    # grey no signal [0 0 0]

    min_, max_ = min(signal), max(signal)

    # return (0,) + tuple([y if y >= 0.01 else 0 for y in [x/max_ for x in signal]])[1:]

    # return tuple([y if y >= 0.01 else 0 for y in [x/max_ for x in signal]])

    # return tuple([y if y >= 0.01 else 0 for y in [x/top for x in signal]])

    noise = 0.1 * max_

    if max_ - min_ <= noise:
        # no response -> light gray
        return (0.8, 0.8, 0.8)

    start = signal[0]
    finish = signal[-1]
    middle = max(signal[1:-1])

    if not start < max(middle, finish):
        return (0, 0, 0)

    if max(start, middle) - min_ <= noise:
        assert finish - min_ > noise
        # late -> magenta
        return (1, 0, 1)

    # if finish <= start + noise:
    #     # transient -> yellow
    #     return (1, 1, 0)

    if finish <= 0.5 * (max_ + min_) + noise:
        # transient -> yellow
        return (1, 1, 0)

    # sustained
    return (0, 1, 0)


bgcolor = bgcolor_datarail_1_4
bgcolor = bgcolor_datarail_2_0
# bgcolor = lambda x, y: (1, 1, 1)

def color_polys(signal_line, reference_line, rect):
    signal, reference = [tuple([p[1] for p in line])
                         for line in signal_line, reference_line]
    (l, r), (b, t) = rect.bounds

    bg = mma_rgb(*bgcolor(signal, reference))
    fg = mma_rgb(*fgcolor(signal, t))

    bgpoly = Polygon(tuple(reference_line) + (Point((r, t)), Point((l, t))))
    fgpoly = Polygon(tuple(signal_line) + (Point((r, b)), Point((l, b))))

    return ((bg, bgpoly), (fg, fgpoly))


rowname = 'IL8'
new_cube = pickle.load(open(re.sub(r'fig2', 'xtra', INPUTPATH)))
new_cube = new_cube.slice('time=1440', 'readout:name=' + rowname,
                          'cell line=PriHu').forget_trivial()
il8 = list(new_cube)
minmax = [mm(il8) for mm in min, max]
readout_base = minmax[0]
readout_range = minmax[1] - readout_base

# ST()
new_cube = (new_cube - readout_base)/readout_range

# print new_cube.attribs
# print (min(il8), max(il8))
# print il8
# for (cyt, inh), k in new_cube.iterate_over(('cytokine', 'inhibitor')):
#     val = k._tupleform()[-1][-1]
#     print '\t'.join((cyt, inh, str(val)))
# exit(0)

colnames = tuple([t[0] for t in new_cube.attribute_values(('inhibitor',))])
supercolnames = tuple([t[0] for t in new_cube.attribute_values(('cytokine',))])
rect_corners = ((0, 0), (1, 1))

heatmap_rgb = color_interpolator((0, 1), (RGB([1, 1, 0]), RGB([1, 0.25, 0])))

#scaling_factor = (1, 2.5)
scaling_factor = 1
stride = Point(rect_corners[1]) * scaling_factor * (1, -1)

supercols = []

for _, supercol in new_cube.iterate_over('cytokine'):
    facts = list(supercol)

    regs = [Region(rect=Interval2D(*rect_corners),
                   members=(mma_rgb(*heatmap_rgb(v)),
                            Rectangle(*rect_corners)))
            for v in supercol]

    supercols.append(RegionGrid.make_grid((regs,), stride, scaling_factor,
                                          with_gridlines=True).flatten())



supercol_sizes = [f.size for f in supercols]
assert len(set(supercol_sizes)) == 1
scs = supercol_sizes[0]

quadrant = (1, -1)
hsep = 0.05 * scs[0]
stride = quadrant * Point(scs) + 2 * hsep * Point(quadrant)

origin = hsep * Point(quadrant)
origin = Point((-200, -50))

heatmap = RegionGrid.make_grid((supercols,), stride=stride,
                               origin=origin, quadrant=quadrant)

colnames = new_cube.attribute_values('inhibitor')
heatmap_labels = mk_labels(heatmap,
                           (None, ('IL8',), colnames,
                                  (int(round(minmax[1], -2)),)), hsep)

# ----------------------------------------------------------------------

the_cube = pickle.load(open(INPUTPATH))

signal, reference = 'PriHu', 'HepG2'
#signal, reference = 'HepG2', 'PriHu'


max_mp_x = len(the_cube.attribute_values(('time',))) - 1
assert max_mp_x > 0

max_mp_y = VFUDGE * max(the_cube.facts())
assert max_mp_y > 0

mp_aspect_ratio = 3
mp_scale = (1, mp_aspect_ratio * max_mp_x/max_mp_y)

xstride = max_mp_x * mp_scale[0]
ystride = -max_mp_y * mp_scale[1]

rownames = tuple([t[0] for t in the_cube.attribute_values(('readout:name',))])
colnames = tuple([t[0] for t in the_cube.attribute_values(('inhibitor',))])
supercolnames = tuple([t[0] for t in the_cube.attribute_values(('cytokine',))])
rowmaxes = []

for ro in rownames:
    slc = the_cube.slice('readout:name=%s' % ro).forget_trivial()
    m = max(d[1] for d in slc._tupleform())
    rowmaxes.append(int(round(m, -2)))

sortkey = lambda x: x[0][0][1]
supercols = []

for cy in supercolnames:
    supercol = the_cube.slice('cytokine=%s' % cy)
    supercoldata = []
    for ro in rownames:
        rowregs = []
        row = supercol.slice('readout:name=%s' % ro)
        for ih in colnames:
            data = {}
            col = row.slice('inhibitor=%s' % ih)
            for cell_line in signal, reference:
                curve = col.slice('cell line=%s' %
                                  cell_line).forget_trivial()
                tf = curve._tupleform()
                data[cell_line] = tuple([d[1] for d in sorted(tf, key=sortkey)])

            signal_line, reference_line = lines = \
                tuple([Line(list(enumerate(data[x]))) for x in
                       signal, reference])

            rect = Interval2D((0, 0), (max_mp_x, max_mp_y))
            bg = color_polys(signal_line, reference_line, rect)
            members = add(*zip(bg, lines))
            colreg = Region(rect=rect, members=(add(*zip(bg, lines))))

            rowregs.append(colreg)
        supercoldata.append(rowregs)

    supercolreg = RegionGrid.make_grid(supercoldata, Point((xstride, ystride)),
                                       mp_scale, with_gridlines=True)

    #supercolreg = supercolreg.flatten()
    supercols.append(supercolreg)

supercol_sizes = [f.size for f in supercols]

assert len(set(supercol_sizes)) == 1
scs = supercol_sizes[0]

quadrant = (1, -1)
hsep = 0.05 * scs[0]

origin = hsep * Point(quadrant)

origin = Point((-200, -50))

stride = quadrant * Point(scs) + 2 * hsep * Point(quadrant)

grid = RegionGrid.make_grid((supercols,), stride=stride,
                            origin=origin, quadrant=quadrant)

import sys
if len(sys.argv) > 1 and sys.argv[1] == '--dump':
    outh = open('PRIV/dumps/pickle/heatmap.pkl', 'w')
    pickle.dump(heatmap, outh)
    outh.close()
    outh = open('PRIV/dumps/pickle/grid.pkl', 'w')
    pickle.dump(grid, outh)
    outh.close()
    exit(0)


scaling = numeric_vector((grid.size[0], grid.members[0].region.members[0].size[1]))/heatmap.size

ST()

rowmaxes = map(str, rowmaxes)
ls = mk_labels(grid, (supercolnames, rownames, colnames, rowmaxes), hsep)
# ls = mk_labels(grid, (supercolnames, rownames, colnames, rowmaxes))

heatmap.extend(heatmap_labels)
grid.extend(ls)

# print grid.as_string()
# print grid.as_mma()
print grid.as_string()

# pp(grid.__dict__)

# from copy import deepcopy

# print '\n'
# pp(deepcopy(grid).__dict__)
