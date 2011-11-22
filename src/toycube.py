from __future__ import division
import sys
import os
import re
import operator
from numbers import Number
import collections
from collections import defaultdict
from frozendict import frozendict
from itertools import product
from cartesianproduct import cartesianproduct
from pdb import set_trace as ST

from dimension import Dimension

# -----------------------------------------------------------------------------

INPUTPATH = os.environ['HOME'] + '/_/prj/datarail/g/PRIV/data/alexopoulos10_fig2_data_star.tsv'
INPUTPATH = os.environ['HOME'] + '/_/prj/datarail/g/PRIV/data/alexopoulos10_fig2_data_star.md2.1'
FLAG = False

# -----------------------------------------------------------------------------

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

class Cube(object):
    def __init__(self, records):
        lr = len(records)
        main = [None] * lr
        star = [None] * lr
        attribs = set()
        _lookup = defaultdict(lambda: defaultdict(set))
        for id_, record in enumerate(sorted(records)):
            info, fact = record
            star[id_] = (id_, frozendict(info))
            main[id_] = (id_, fact)

            attribs.update(set(info.keys()))

            for attrib, value in info.items():
                if not value is None:
                    _lookup[attrib][value].add(id_)

        self.main = frozendict(main)
        self.star = frozendict(star)
        self.attribs = frozenset(attribs)

        self._attrib_dir = dict((a.name, a) for a in attribs)
        self._lookup = frozendict([(k, frozendict(v.items()))
                                   for k, v in _lookup.items()])

        self._check()


    def __len__(self):
        return len(self.main.keys())


    # PREMATURE OPTIMIZATION: USE ASSERTS
    def _check(self):
        """Check the validity of the arguments to __init__.

        Checks that all the keys in the _lookup argument are valid
        attribute names (i.e. they don't contain the separator "=").
        """

        bad_names = filter(lambda x: '=' in str(x), list(self.attribs))
        if bad_names:
            raise ValueError('Invalid attribute names: %s' %
                             ', '.join(bad_names))


    # PREMATURE OPTIMIZATION: USE ASSERTS
    def _check_specs(self, *specs):
        """Check validity of specifications list in specs.

        Returns the dict if (attribute name, attribute value) pairs
        encoded in specs.
        """

        specmap = dict()
        _lookup = self._lookup
        for spec in specs:
            try:
                attrib, attrib_value = spec.split('=')
            except ValueError:
                raise ValueError('Invalid spec: %s' % spec)
            if not attrib in _lookup:
                raise ValueError('Invalid attribute: %s' % attrib)

            attrib = self._attrib_dir[str(attrib)]
            # attrib now is a Dimension object, not a just string

            if attrib in specmap:
                raise ValueError('Repeated attribute: %s' % attrib)
            specmap[attrib] = attrib.convert(attrib_value)
        return specmap


    def _attribute_values(self, attribs):
        dir_ = self._attrib_dir
        attribs = [dir_[a] for a in attribs]
        la = len(attribs)
        if la == 1:
            attrib = attribs[0]
            d = self._lookup[attrib]
            values = attrib.values
            if values:
                values = filter(lambda x: x in d, values)
            else:
                values = d.keys()
            return tuple(map(lambda x: (x,), values))
        else:
            ret = []
            for v in self._lookup[attribs[0]].keys():
                c = self.slice('%s=%s' % (attribs[0], str(v)))
                for t in c._attribute_values(attribs[1:]):
                    ret.append((v,) + t)
            return tuple(ret)
                

    def attribute_values(self, attribs):
        """Return list of tuples corresponding to all values for attribs.

        The attribs parameter is a sequence of strings, representing
        (data) attributes of self (not to be confused with Python
        attributes).  The ordering of the values in each of the tuples
        in the returned list is determined by the iteration order of
        the attributes in attribs.
        """

        return_tuples = hasattr(attribs, '__iter__')
        if not return_tuples:
            attribs = (attribs,)

        assert len(attribs) > 0
        assert len(attribs) == 1 or return_tuples

        s = set(attribs)
        unknown = s.difference(self.attribs)
        if unknown:
            raise ValueError('Unknown attribute(s): %s' %
                             ', '.join(filter(lambda a: a in unknown, attribs)))

        vals = self._attribute_values(attribs)
        return vals if return_tuples else tuple([t[0] for t in vals])


    def __iter__(self):
        for _, c in self.iterate_over([a.name for a in self.attribs]):
            yield c._tupleform()[-1][-1]


    def iterate_over(self, attribs, drop=True):
        """Return an iterator for subcubes over sets of attribute values.
        """

        if not (drop is True or drop is False):
            raise ValueError('the drop parameter must be a boolean')

        if drop:
            def slicer(*specs):
                return self.slice(*specs).forget_trivial()
        else:
            slicer = self.slice

        if hasattr(attribs, '__iter__'):
            ps = [[s[0] for s in self.attribute_values((a,))] for a in attribs]
            for index in product(*ps):
                args = ['%s=%s' % (a, v) for a, v in zip(attribs, index)]
                yield (index, slicer(*args))
        else:
            for index in [s[0] for s in self.attribute_values((attribs,))]:
                yield (index, slicer('%s=%s' % (attribs, index)))


    def extend(self, newinfo):
        if hasattr(newinfo, 'keys'):
            newattribs = set(newinfo.keys())
        else:
            # newinfo is a sequence of pairs
            newattribs = set(x[0] for x in newinfo)

        overlap = self.attribs.intersection(newattribs)
        del newattribs

        if overlap:
            raise ValueError('Attribute(s) already exist: %s'
                             % ', '.join(overlap))
        del overlap
        if hasattr(newinfo, 'items'):
            newinfo = newinfo.items()

        orig_data = self._data()
        cls = type(orig_data[0][0])
        data = [(cls(info.items() + newinfo), val) for info, val in orig_data]

        return Cube(data)


    def _data(self, idset=None, copy=False):
        main = self.main
        star = self.star
        if idset is None:
            idset = set(main.keys())
        if copy:
            return [(star[id_].copy(), main[id_]) for id_ in idset]
        else:
            return [(star[id_], main[id_]) for id_ in idset]


    def _subcube(self, idset):
        return Cube(self._data(idset))


    def _tupleform(self):
        ret = []
        for info, fact in self._data():
            keys = [(str(k), v) for k, v in info.items()]
            ret.append((tuple(sorted(keys)), fact))
        return tuple(sorted(ret))


    def facts(self):
        return [p[1] for p in self._data()]


    def __repr__(self):
        tf = self._tupleform()
        if len(tf) > 0:
            headers = [p[0] for p in tf[0][0]]
        else:
            headers = sorted(list(self.attribs))

        ret = ['\t'.join(map(str, headers) + ['fact']) + '\n']
        for info, fact in tf:
            row = [p[1] for p in info] + [fact]
            ret.append('\t'.join(map(str, row)) + '\n')

        return ''.join(ret)


    def merge(self, *cubes):
        return Cube(sum([c._data() for c in [self] + list(cubes)], []))
                

    def _id_slice(self, *specs):
        specmap = self._check_specs(*specs)
        found = set(self.main.keys())
        for attrib_name, attrib_value in specmap.items():
            found.intersection_update(self._lookup[attrib_name][attrib_value])
            if len(found) == 0:
                break
        return found


    def slice(self, *specs):
        return self._subcube(self._id_slice(*specs))


    def sliceoff(self, *specs):
        wanted = self._id_slice(*specs)
        rest = set(self.main.keys()).difference(wanted)
        return self._subcube(wanted), self._subcube(rest)


    def forget(self, *attribs):
        """Make cube obtained by discarding all attributes in attribs.
        """

        todrop = set(attribs)
        data = []
        for info, fact in self._data():
            items = filter(lambda i: not i[0] in todrop, info.items())
            data.append((type(info)(items), fact))
        return Cube(data)


    def forget_trivial(self):
        """Make cube obtained by discarding all "trivial" attributes.
        """

        return self.forget(*list(self.trivial_attributes()))


    def trivial_attributes(self):
        """Return the set of all trivial attributes.

        A "trivial" attribute is one that has at most one value that
        is not a "trivial" value.  A "trivial" value is one that is
        held by no fact.
        """

        trivial = set()
        for attrib, d in self._lookup.items():
            if len(filter(lambda s: len(s) > 0, d.values())) < 2:
                trivial.add(attrib)
        return trivial


    def copy(self):
        return self._subcube(set(self.main.keys()))

    # -------------------------------------------------------------------------
    # Cube arithmetic

    def __sub__(self, other):
        if isinstance(other, Number):
            ret = self.copy()
            ret.main = frozendict([(k, operator.__sub__(v, other))
                                   for k, v in ret.main.items()])
        elif isinstance(other, type(self)):
            raise NotImplementedError

        return ret


    def __truediv__(self, other):
        if isinstance(other, Number):
            ret = self.copy()
            ret.main = frozendict([(k, v/other) for k, v in ret.main.items()])
        elif isinstance(other, type(self)):
            raise NotImplementedError

        return ret

    __div__ = __truediv__


## sketch:
# def vapply(kvs, op, other=None):
#     if other is None:
#         return [(k, op(v)) for k, v in kvs]
#     else:
#         return [(k, op(v, other)) for k, v in kvs]

## there's still much room for refactoring-away common code
# def _make_op(methodname, arity=2):
#     op = getattr(operator, methodname)
#     if arity == 1:
#         def method(self):
#             if isinstance(other, Number):
#                 ret = self.copy()
#                 ret.main = frozendict(vapply(ret.main.items(), op))
#             elif isinstance(other, type(self)):
#                 raise NotImplementedError

#             return ret
#     else:
#         def method(self, other):
#             if isinstance(other, Number):
#                 ret = self.copy()
#                 ret.main = frozendict(vapply(ret.main.items(), op, other))
#             elif isinstance(other, type(self)):
#                 raise NotImplementedError

#             return ret

#     method.__name__ = methodname
#     return method

# for methodname in ('__add__ __div__ __floordiv__ __mul__ __pow__ __sub__ '
#                    '__truediv__ __iadd__ __idiv__ __ifloordiv__ __imul__ '
#                    '__ipow__ __isub__ __itruediv__'.split()):
#     setattr(Cube, methodname, _make_binary_op(methodname)

# del _make_binary_op

## numeric input(s) and output
## unary
# __abs__ __neg__ __pos__
## binary
# __add__ __div__ __floordiv__ __mul__ __pow__ __sub__ __truediv__ __iadd__ __idiv__ __ifloordiv__ __imul__ __ipow__ __isub__ __itruediv__


## integer input(s) and output
## unary
# __index__ __inv__ __invert__
## binary
# __lshift__ __rshift__ __mod__ __ilshift__ __irshift__ __imod__

## arbitrary inputs; boolean output (all binary)
# __eq__ __ge__ __gt__ __le__ __lt__ __ne__

## boolean input(s) and output
## unary
# __not__
## binary
# __and__ __or__ __xor__ __iand__ __ior__ __ixor__


# -----------------------------------------------------------------------------

def propagate_zero(cube, zerospec, *specs):
    z, c = cube.sliceoff(zerospec, *specs)

    if len(specs) > 0:
        dims = tuple(cube._check_specs(*specs).keys())
    else:
        zdim = set(cube._check_specs(zerospec).keys())
        nontrivial = cube.attribs.difference(cube.trivial_attributes())
        trivial = z.trivial_attributes()
        dims = tuple(nontrivial.intersection(trivial).difference(zdim))
        del nontrivial, trivial
        if not dims:
            raise ValueError('Unable to determine dimension(s) '
                             'to propagate over')

    z = z.forget(*dims)
    cubes = list()
    for t in cube.attribute_values(dims):
        cubes.append(z.extend(zip(dims, t)))
    c = c.merge(*cubes)
    return c
    

def _read_data_DISABLE():
    '''Return data in input file as a list of records.

    Each record is a tuple of two elements: "info" and "fact".  The
    "info" element is a (small) dictionary (describing the coordinates
    of the record's "fact" element).  Its keys are attribute names,
    and its values are the settings for these attributes corresponding
    to the record's "fact" element.
    '''

    infh = open(INPUTPATH, 'U')

    headers = infh.readline().strip().split('\t')
    attrib_names = [Dimension(name=h.split('=')[0])
                    for h in headers]

    records = []
    for line in infh:
        fields = line.strip().split('\t')
        info = dict(zip(attrib_names[:-1], fields[:-1]))
        fact = fields[-1]
        records.append((info, fact))

    return records


def _read_data(inputpath):
    '''Return data in inputpath as a list of records.

    Each record is a tuple of two elements: "info" and "fact".  The
    "info" element is a (small) dictionary (describing the coordinates
    of the record's "fact" element).  Its keys are attribute names,
    and its values are the settings for these attributes corresponding
    to the record's "fact" element.
    '''

    def _to_dict(line):
        return dict(s.split('=') for s in line.strip().split('\t'))

    infh = open(inputpath, 'U')
    firstline = _to_dict(infh.readline())
    version = firstline['version']
    nrows = int(firstline['rows'])
    ncols = int(firstline['columns'])

    colnum = 0
    headers = {}
    dims = []
    for line in infh:
        if re.search(r'^\s*#', line):
            continue
        fields = _to_dict(line)

        name = fields['name']
        assert not name in headers

        fields['type_'] = fields.pop('type', None)

        dim = Dimension(**fields)
        dims.append(dim)
        headers[name] = dim
        colnum += 1
        if colnum == ncols:
            break
    del colnum

    records = []
    rownum = 0
    for line in infh:
        fields = line.strip().split('\t')
        assert len(fields) == ncols
        fields = [x[0].read(x[1]) for x in zip(dims, fields)]
        info = OrderedDict(zip(dims[:-1], fields[:-1]))
        fact = fields[-1]
        records.append((info, fact))
        rownum += 1

    assert rownum == nrows
    return records

if __name__ == '__main__':
    # for s in 'fig2', 'xtra':
    #     INPUTPATH = os.environ['HOME'] + ('/_/prj/datarail/g/PRIV/data/alexopoulos10_%s_data_star.md2.1' % s)
    #     c = propagate_zero(Cube(_read_data(INPUTPATH)), 'time=0')
    #     import pickle
    #     OUTPUTPATH = re.sub(r'(?<=PRIV/)data', 'dumps/pickle',
    #                         re.sub(r'_star.*$', '.cube.pkl', INPUTPATH))
    #     outfh = open(OUTPUTPATH, 'w')
    #     pickle.dump(c, outfh)
    #     outfh.close()
    # exit(0)

    c = propagate_zero(Cube(_read_data()), 'time=0')

    print len(c.main.keys())
    # print c.slice('cell line=PriHu', 'ligand=IL1a', 'inhibitor=NO-INHIB', 'feature=HSP27')
    # print c.slice('feature=HSP27', 'inhibitor=NO-INHIB', 'ligand=IL1a', 'cell line=PriHu')
    # print c.slice('feature=HSP27', 'inhibitor=NO-INHIB').slice('ligand=IL1a', 'cell line=PriHu')
    # k = c.slice('cell line=PriHu', 'ligand=IL1a', 'inhibitor=NO-INHIB', 'feature=HSP27')

    print c.slice('cell line=PriHu', 'cytokine=IL1a', 'inhibitor=NO-INHIB', 'readout:name=HSP27')
    print c.slice('readout:name=HSP27', 'inhibitor=NO-INHIB', 'cytokine=IL1a', 'cell line=PriHu')
    print c.slice('readout:name=HSP27', 'inhibitor=NO-INHIB').slice('cytokine=IL1a', 'cell line=PriHu')
    k = c.slice('cell line=PriHu', 'cytokine=IL1a', 'inhibitor=NO-INHIB', 'readout:name=HSP27')

    k = k.forget_trivial()
    print k
