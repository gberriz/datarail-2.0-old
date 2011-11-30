import sys
from collections import defaultdict, namedtuple
from multikeydict import MultiKeyDict as mkd
from pdb import set_trace as ST

path = sys.argv[1]
ALT = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False
if ALT:
    print 'running under ALT: %s' % str(ALT)

KeyCoords = namedtuple('KeyCoords',
                       'cell_line ligand_name ligand_concentration time signal')

ValCoords = namedtuple('ValCoords',
                       'assay plate well field channel antibody')

Coords = namedtuple('Coords', KeyCoords._fields + ('repno',) + ValCoords._fields)

if ALT:
    BYSTUDY = mkd(1, list)
else:
    pre = defaultdict(list)

def convert(s):
    try:
        return float(s) if '.' in s else int(s)
    except ValueError:
        return s.decode('utf-8')

with open(path) as fh:
    assert 'cell_line' in fh.next()
    # print '\t'.join((','.join(KeyCoords._fields),
    #                  ','.join(ValCoords._fields)))
    # print ','.join(KeyCoords._fields + ('\t',) + ValCoords._fields)

    for line in fh:
        h, t = line.strip().split('\t')
        k = KeyCoords(*map(convert, h.split(','))[:5])
        v = ValCoords(*map(convert, t.split(',')))
        r = Coords(*(k + ((),) + v))
        # print u'\t'.join((u','.join(map(unicode, k)),
        #                   u','.join(map(unicode, v)))).encode('utf-8')
        # print u','.join(map(unicode, k) + [u'\t'] + map(unicode, v)).encode('utf-8')

        if ALT:
            BYSTUDY.get((k,)).append(r)
        else:
            pre[k].append(r)


# exit(0)

if ALT:
    pass
else:
    BYSTUDY = mkd()
    BYSTUDY[mkd.NIL] = pre


def regroup0(div, sz, group):
    lkp = defaultdict(set)
    newgroup = mkd(sz + 1, list)
    j = Coords._fields.index(div)
    first = True
    for ck, rs in group.iteritemsmk():
        c, k = tuple(ck[:-1]), (ck[-1],)
        if div == 'assay' or div == 'plate':
            if first:
                first = False
                print c # prints ((),) for assay
        u = set()
        for r in rs:
            f = r[j]
            u.add(f)
            newgroup.get(c + (f,) + k).append(r)
        if len(u) > 1:
            lkp[c].add(tuple(sorted(u)))
    return lkp, newgroup


def regroup(div, sz, group):
    lkps = defaultdict(set)
    j = Coords._fields.index(div)
    newgroup = mkd(sz, lambda: defaultdict(list))
    # first = True
    for c, krs in group.iteritemsmk():
        # c = cc[1:] if cc[0] == mkd.NIL else cc
        # c = cc[1:] if (len(cc) and cc[0] == mkd.NIL) else cc
        sigs = set()
        # if first:
        #     first = False
        #     print c # prints ((),) for assay
        for k, rs in krs.items():
            u = set([r[j] for r in rs])
            if len(u) > 1:
                sigs.add(tuple(sorted(u)))

        if sigs:
            assert len(set(map(len, sigs))) == 1
            lkp = dict(sum([[(v, i + 1) for v in vs] for i, vs in
                            enumerate(zip(*sorted(sigs)))], []))
            torepno = lambda s: lkp.get(s, 0)
            # if div == 'well' or div == 'field':
            #     ST()
            #     pass
        else:
            torepno = lambda s: 0

        lkps[c] = torepno

        for k, rs in krs.items():
            for r in rs:
                f = r[j]
                newgroup.get(c + (f,))[k].append(r._replace(repno=r.repno + (torepno(f),)))

    return lkps, newgroup


grp = BYSTUDY
lookup = dict()
rg = regroup0 if ALT else regroup
for i, div in enumerate('assay plate well field'.split()):
    lookup[div], grp = rg(div, i + 1, grp)

# for cc, krs in grp.iteritemsmk():
#     ST()
#     pass

print lookup['assay']
print lookup['plate']
print lookup['well']
