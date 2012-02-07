import unittest

# import unittest, doctest, operator
# import inspect
from test import test_support
from test import list_tests

# from collections import namedtuple, Counter, OrderedDict

from nodup import NoDup
from test import list_tests

# from test import mapping_tests
# import pickle, cPickle, copy
# from random import randrange, shuffle
# import keyword
# import re
# import sys
# from collections import Hashable, Iterable, Iterator
# from collections import Sized, Container, Callable
# from collections import Set, MutableSet
# from collections import Mapping, MutableMapping
# from collections import Sequence, MutableSequence


class NoDupTest(list_tests.CommonTest):
    type2test = NoDup

    def test_getslice(self):
        super(NoDupTest, self).test_getslice()
        l = [0, 1, 2, 3, 4]
        u = self.type2test(l)
        for i in range(-3, 6):
            self.assertEqual(u[:i], l[:i])
            self.assertEqual(u[i:], l[i:])
            for j in xrange(-3, 6):
                self.assertEqual(u[i:j], l[i:j])

    def test_add_specials(self):
        u = NoDup("spam")
        u2 = u + "eggs"
        self.assertEqual(u2, list("spameggs"))

    def test_radd_specials(self):
        u = NoDup("eggs")
        u2 = "spam" + u
        self.assertEqual(u2, list("spameggs"))
        u2 = u.__radd__(NoDup("spam"))
        self.assertEqual(u2, list("spameggs"))

    def test_iadd(self):
        super(NoDupTest, self).test_iadd()
        u = [0, 1]
        u += NoDup([0, 1])
        self.assertEqual(u, [0, 1, 0, 1])

    def test_mixedcmp(self):
        u = self.type2test([0, 1])
        self.assertEqual(u, [0, 1])
        self.assertNotEqual(u, [0])
        self.assertNotEqual(u, [0, 2])

    def test_mixedadd(self):
        u = self.type2test([0, 1])
        self.assertEqual(u + [], u)
        self.assertEqual(u + [2], [0, 1, 2])

    def test_getitemoverwriteiter(self):
        # Verify that __getitem__ overrides *are* recognized by __iter__
        class T(self.type2test):
            def __getitem__(self, key):
                return str(key) + '!!!'
        self.assertEqual(iter(T((1,2))).next(), "0!!!")



    def test_addmul(self):
        u1 = self.type2test([0])
        u2 = self.type2test([0, 1])
        self.assertEqual(u1, u1 + self.type2test())
        self.assertEqual(u1, self.type2test() + u1)
        self.assertEqual(u1 + self.type2test([1]), u2)
        self.assertEqual(self.type2test([-1]) + u1, self.type2test([-1, 0]))
        self.assertEqual(self.type2test(), u2*0)
        self.assertEqual(self.type2test(), 0*u2)
        self.assertEqual(self.type2test(), u2*0L)
        self.assertEqual(self.type2test(), 0L*u2)
        self.assertEqual(u2, u2*1)
        self.assertEqual(u2, 1*u2)
        self.assertEqual(u2, u2*1L)
        self.assertEqual(u2, 1L*u2)

        class subclass(self.type2test):
            pass
        u3 = subclass([0, 1])
        self.assertEqual(u3, u3*1)
        self.assertIsNot(u3, u3*1)


    def test_contains_fake(self):
        pass

    def test_contains_order(self):
        pass

    def test_count(self):
        a = self.type2test([0, 1, 2]*3)
        self.assertEqual(a.count(0), 1)
        self.assertEqual(a.count(1), 1)
        self.assertEqual(a.count(3), 0)

        self.assertRaises(TypeError, a.count)

        class BadExc(Exception):
            pass

        class BadCmp:
            def __eq__(self, other):
                if other == 2:
                    raise BadExc()
                return False

        self.assertRaises(BadExc, a.count, BadCmp())


    def test_imul(self):
        u = self.type2test([0, 1])
        u *= 1
        self.assertEqual(u, self.type2test([0, 1]))
        u *= 0
        self.assertEqual(u, self.type2test([]))
        s = self.type2test([])
        oldid = id(s)
        s *= 1
        self.assertEqual(id(s), oldid)
        s *= 0
        self.assertEqual(id(s), oldid)


    def test_insert(self):
        a = self.type2test([0, 1, 2])
        a.insert(0, -3)
        a.insert(1, -2)
        a.insert(2, -1)
        self.assertEqual(a, [-2, -1, 0, 0, 1, 2])

        b = a[:]
        b.insert(-2, "foo")
        b.insert(-200, "left")
        b.insert(200, "right")
        self.assertEqual(b, self.type2test(["left", -3, -2, -1, 0,
                                            "foo", 1, 2, "right"]))

        self.assertRaises(TypeError, a.insert)


    def test_print(self):
        pass

    def test_repeat(self):
        for m in xrange(4):
            s = tuple(range(m))
            for n in xrange(-3, 2):
                self.assertEqual(self.type2test(s*n), self.type2test(s)*n)
            self.assertEqual(self.type2test(s)*(-4), self.type2test([]))
            self.assertEqual(id(s), id(s*1))


    def test_set_subscript(self):
        a = self.type2test(range(20))
        self.assertRaises(ValueError, a.__setitem__, slice(0, 10, 0), [1,2,3])
        self.assertRaises(TypeError, a.__setitem__, slice(0, 10), 1)
        self.assertRaises(ValueError, a.__setitem__, slice(0, 10, 2), [1,2])
        self.assertRaises(TypeError, a.__getitem__, 'x', 1)
        a[slice(2,10,3)] = [1,2,3]
        self.assertEqual(a, self.type2test([0, 1, 1, 3, 4, 2, 6, 7, 3,
                                            9, 10, 11, 12, 13, 14, 15,
                                            16, 17, 18, 19]))


    def test_setitem(self):
        a = self.type2test([0, 1])
        a[0] = 0
        a[1] = 100
        self.assertEqual(a, self.type2test([0, 100]))
        a[-1] = 200
        self.assertEqual(a, self.type2test([0, 200]))
        a[-2] = 100
        self.assertEqual(a, self.type2test([100, 200]))
        self.assertRaises(IndexError, a.__setitem__, -3, 200)
        self.assertRaises(IndexError, a.__setitem__, 2, 200)

        a = self.type2test([])
        self.assertRaises(IndexError, a.__setitem__, 0, 200)
        self.assertRaises(IndexError, a.__setitem__, -1, 200)
        self.assertRaises(TypeError, a.__setitem__)

        a = self.type2test([0,1,2,3,4])
        a[0L] = 1
        a[1L] = 2
        a[2L] = 3
        self.assertEqual(a, self.type2test([1,2,3,3,4]))
        a[0] = 5
        a[1] = 6
        a[2] = 7
        self.assertEqual(a, self.type2test([5,6,7,3,4]))
        a[-2L] = 88
        a[-1L] = 99
        self.assertEqual(a, self.type2test([5,6,7,88,99]))
        a[-2] = 8
        a[-1] = 9
        self.assertEqual(a, self.type2test([5,6,7,8,9]))


# def test_main():
#     with test_support.check_py3k_warnings(
#             (".+__(get|set|del)slice__ has been removed", DeprecationWarning)):
#         test_support.run_unittest(NoDupTest)

# if __name__ == "__main__":
#     test_main()


# class TestOrderedDict(unittest.TestCase):

#     def test_init(self):
#         with self.assertRaises(TypeError):
#             OrderedDict([('a', 1), ('b', 2)], None)                                 # too many args
#         pairs = [('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5)]
#         self.assertEqual(sorted(OrderedDict(dict(pairs)).items()), pairs)           # dict input
#         self.assertEqual(sorted(OrderedDict(**dict(pairs)).items()), pairs)         # kwds input
#         self.assertEqual(list(OrderedDict(pairs).items()), pairs)                   # pairs input
#         self.assertEqual(list(OrderedDict([('a', 1), ('b', 2), ('c', 9), ('d', 4)],
#                                           c=3, e=5).items()), pairs)                # mixed input

#         # make sure no positional args conflict with possible kwdargs
#         self.assertEqual(inspect.getargspec(OrderedDict.__dict__['__init__']).args,
#                          ['self'])

#         # Make sure that direct calls to __init__ do not clear previous contents
#         d = OrderedDict([('a', 1), ('b', 2), ('c', 3), ('d', 44), ('e', 55)])
#         d.__init__([('e', 5), ('f', 6)], g=7, d=4)
#         self.assertEqual(list(d.items()),
#             [('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5), ('f', 6), ('g', 7)])

#     def test_update(self):
#         with self.assertRaises(TypeError):
#             OrderedDict().update([('a', 1), ('b', 2)], None)                        # too many args
#         pairs = [('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5)]
#         od = OrderedDict()
#         od.update(dict(pairs))
#         self.assertEqual(sorted(od.items()), pairs)                                 # dict input
#         od = OrderedDict()
#         od.update(**dict(pairs))
#         self.assertEqual(sorted(od.items()), pairs)                                 # kwds input
#         od = OrderedDict()
#         od.update(pairs)
#         self.assertEqual(list(od.items()), pairs)                                   # pairs input
#         od = OrderedDict()
#         od.update([('a', 1), ('b', 2), ('c', 9), ('d', 4)], c=3, e=5)
#         self.assertEqual(list(od.items()), pairs)                                   # mixed input

#         # Issue 9137: Named argument called 'other' or 'self'
#         # shouldn't be treated specially.
#         od = OrderedDict()
#         od.update(self=23)
#         self.assertEqual(list(od.items()), [('self', 23)])
#         od = OrderedDict()
#         od.update(other={})
#         self.assertEqual(list(od.items()), [('other', {})])
#         od = OrderedDict()
#         od.update(red=5, blue=6, other=7, self=8)
#         self.assertEqual(sorted(list(od.items())),
#                          [('blue', 6), ('other', 7), ('red', 5), ('self', 8)])

#         # Make sure that direct calls to update do not clear previous contents
#         # add that updates items are not moved to the end
#         d = OrderedDict([('a', 1), ('b', 2), ('c', 3), ('d', 44), ('e', 55)])
#         d.update([('e', 5), ('f', 6)], g=7, d=4)
#         self.assertEqual(list(d.items()),
#             [('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5), ('f', 6), ('g', 7)])

#     def test_clear(self):
#         pairs = [('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)]
#         shuffle(pairs)
#         od = OrderedDict(pairs)
#         self.assertEqual(len(od), len(pairs))
#         od.clear()
#         self.assertEqual(len(od), 0)

#     def test_delitem(self):
#         pairs = [('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)]
#         od = OrderedDict(pairs)
#         del od['a']
#         self.assertNotIn('a', od)
#         with self.assertRaises(KeyError):
#             del od['a']
#         self.assertEqual(list(od.items()), pairs[:2] + pairs[3:])

#     def test_setitem(self):
#         od = OrderedDict([('d', 1), ('b', 2), ('c', 3), ('a', 4), ('e', 5)])
#         od['c'] = 10           # existing element
#         od['f'] = 20           # new element
#         self.assertEqual(list(od.items()),
#                          [('d', 1), ('b', 2), ('c', 10), ('a', 4), ('e', 5), ('f', 20)])

#     def test_iterators(self):
#         pairs = [('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)]
#         shuffle(pairs)
#         od = OrderedDict(pairs)
#         self.assertEqual(list(od), [t[0] for t in pairs])
#         self.assertEqual(od.keys()[:], [t[0] for t in pairs])
#         self.assertEqual(od.values()[:], [t[1] for t in pairs])
#         self.assertEqual(od.items()[:], pairs)
#         self.assertEqual(list(od.iterkeys()), [t[0] for t in pairs])
#         self.assertEqual(list(od.itervalues()), [t[1] for t in pairs])
#         self.assertEqual(list(od.iteritems()), pairs)
#         self.assertEqual(list(reversed(od)),
#                          [t[0] for t in reversed(pairs)])

#     def test_popitem(self):
#         pairs = [('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)]
#         shuffle(pairs)
#         od = OrderedDict(pairs)
#         while pairs:
#             self.assertEqual(od.popitem(), pairs.pop())
#         with self.assertRaises(KeyError):
#             od.popitem()
#         self.assertEqual(len(od), 0)

#     def test_pop(self):
#         pairs = [('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)]
#         shuffle(pairs)
#         od = OrderedDict(pairs)
#         shuffle(pairs)
#         while pairs:
#             k, v = pairs.pop()
#             self.assertEqual(od.pop(k), v)
#         with self.assertRaises(KeyError):
#             od.pop('xyz')
#         self.assertEqual(len(od), 0)
#         self.assertEqual(od.pop(k, 12345), 12345)

#     def test_equality(self):
#         pairs = [('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)]
#         shuffle(pairs)
#         od1 = OrderedDict(pairs)
#         od2 = OrderedDict(pairs)
#         self.assertEqual(od1, od2)          # same order implies equality
#         pairs = pairs[2:] + pairs[:2]
#         od2 = OrderedDict(pairs)
#         self.assertNotEqual(od1, od2)       # different order implies inequality
#         # comparison to regular dict is not order sensitive
#         self.assertEqual(od1, dict(od2))
#         self.assertEqual(dict(od2), od1)
#         # different length implied inequality
#         self.assertNotEqual(od1, OrderedDict(pairs[:-1]))

#     def test_copying(self):
#         # Check that ordered dicts are copyable, deepcopyable, picklable,
#         # and have a repr/eval round-trip
#         pairs = [('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)]
#         od = OrderedDict(pairs)
#         update_test = OrderedDict()
#         update_test.update(od)
#         for i, dup in enumerate([
#                     od.copy(),
#                     copy.copy(od),
#                     copy.deepcopy(od),
#                     pickle.loads(pickle.dumps(od, 0)),
#                     pickle.loads(pickle.dumps(od, 1)),
#                     pickle.loads(pickle.dumps(od, 2)),
#                     pickle.loads(pickle.dumps(od, -1)),
#                     eval(repr(od)),
#                     update_test,
#                     OrderedDict(od),
#                     ]):
#             self.assertTrue(dup is not od)
#             self.assertEqual(dup, od)
#             self.assertEqual(list(dup.items()), list(od.items()))
#             self.assertEqual(len(dup), len(od))
#             self.assertEqual(type(dup), type(od))

#     def test_yaml_linkage(self):
#         # Verify that __reduce__ is setup in a way that supports PyYAML's dump() feature.
#         # In yaml, lists are native but tuples are not.
#         pairs = [('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)]
#         od = OrderedDict(pairs)
#         # yaml.dump(od) -->
#         # '!!python/object/apply:__main__.OrderedDict\n- - [a, 1]\n  - [b, 2]\n'
#         self.assertTrue(all(type(pair)==list for pair in od.__reduce__()[1]))

#     def test_reduce_not_too_fat(self):
#         # do not save instance dictionary if not needed
#         pairs = [('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)]
#         od = OrderedDict(pairs)
#         self.assertEqual(len(od.__reduce__()), 2)
#         od.x = 10
#         self.assertEqual(len(od.__reduce__()), 3)

#     def test_repr(self):
#         od = OrderedDict([('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)])
#         self.assertEqual(repr(od),
#             "OrderedDict([('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)])")
#         self.assertEqual(eval(repr(od)), od)
#         self.assertEqual(repr(OrderedDict()), "OrderedDict()")

#     def test_repr_recursive(self):
#         # See issue #9826
#         od = OrderedDict.fromkeys('abc')
#         od['x'] = od
#         self.assertEqual(repr(od),
#             "OrderedDict([('a', None), ('b', None), ('c', None), ('x', ...)])")

#     def test_setdefault(self):
#         pairs = [('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)]
#         shuffle(pairs)
#         od = OrderedDict(pairs)
#         pair_order = list(od.items())
#         self.assertEqual(od.setdefault('a', 10), 3)
#         # make sure order didn't change
#         self.assertEqual(list(od.items()), pair_order)
#         self.assertEqual(od.setdefault('x', 10), 10)
#         # make sure 'x' is added to the end
#         self.assertEqual(list(od.items())[-1], ('x', 10))

#     def test_reinsert(self):
#         # Given insert a, insert b, delete a, re-insert a,
#         # verify that a is now later than b.
#         od = OrderedDict()
#         od['a'] = 1
#         od['b'] = 2
#         del od['a']
#         od['a'] = 1
#         self.assertEqual(list(od.items()), [('b', 2), ('a', 1)])

#     def test_views(self):
#         s = 'the quick brown fox jumped over a lazy dog yesterday before dawn'.split()
#         od = OrderedDict.fromkeys(s)
#         self.assertEqual(list(od.viewkeys()),  s)
#         self.assertEqual(list(od.viewvalues()),  [None for k in s])
#         self.assertEqual(list(od.viewitems()),  [(k, None) for k in s])


# class GeneralMappingTests(mapping_tests.BasicTestMappingProtocol):
#     type2test = OrderedDict

#     def test_popitem(self):
#         d = self._empty_mapping()
#         self.assertRaises(KeyError, d.popitem)

# class MyOrderedDict(OrderedDict):
#     pass

# class SubclassMappingTests(mapping_tests.BasicTestMappingProtocol):
#     type2test = MyOrderedDict

#     def test_popitem(self):
#         d = self._empty_mapping()
#         self.assertRaises(KeyError, d.popitem)

# import collections
import nodup

def test_main(verbose=None):
    # NamedTupleDocs = doctest.DocTestSuite(module=collections)
    # test_classes = [TestNamedTuple, NamedTupleDocs, TestOneTrickPonyABCs,
    #                 TestCollectionABCs, TestCounter,
    #                 TestOrderedDict, GeneralMappingTests, SubclassMappingTests]
    with test_support.check_py3k_warnings(
            (".+__(get|set|del)slice__ has been removed", DeprecationWarning)):
        # test_support.run_unittest(NoDupTest)
        test_classes = [NoDupTest]
        test_support.verbose = True
        test_support.run_unittest(*test_classes)
        # test_support.run_doctest(nodup, verbose)

if __name__ == "__main__":
#     class AllEq:
#         # Sequences must use rich comparison against each item
#         # (unless "is" is true, or an earlier item answered)
#         # So instances of AllEq must be found in all non-empty sequences.
#         def __eq__(self, other):
#             return True
#         __hash__ = None # Can't meet hash invariant requirements

#     from UserList import UserList
#     assert not AllEq() in UserList([])
#     assert AllEq() in UserList([1])

#     assert not AllEq() in NoDup([])
#     assert AllEq() in NoDup([1])

    test_main(verbose=True)
