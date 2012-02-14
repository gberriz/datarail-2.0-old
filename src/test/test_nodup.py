import sys
import copy as cp
import unittest as ut
import test.test_support as ts
import test.seq_tests as tst
import test.list_tests as tlt

import nodup as nd

class NoDupTest(tlt.CommonTest):
    type2test = nd.NoDup

    def test_constructors(self):
        l0 = []
        l1 = [0]
        l2 = [0, 1]

        u = self.type2test()
        u0 = self.type2test(l0)
        u1 = self.type2test(l1)
        u2 = self.type2test(l2)

        uu = self.type2test(u)
        uu0 = self.type2test(u0)
        uu1 = self.type2test(u1)
        uu2 = self.type2test(u2)

        v = self.type2test(tuple(u))
        class OtherSeq:
            def __init__(self, initseq):
                self.__data = initseq
            def __len__(self):
                return len(self.__data)
            def __getitem__(self, i):
                return self.__data[i]
        s = OtherSeq(u0)
        v0 = self.type2test(s)
        self.assertEqual(len(v0), len(s))

        s = 'abcdefghijklmnopqrstuvwxyz'
        vv = self.type2test(s)
        self.assertEqual(len(vv), len(s))

        # Create from various iteratables
        for s in ('123', '', range(1000), ('do', 1.2), xrange(2000, 2200, 5)):
            for g in (tst.Sequence, tst.IterFunc, tst.IterGen, tst.itermulti,
                      tst.iterfunc):
                self.assertEqual(self.type2test(g(s)), self.type2test(s))
            self.assertEqual(self.type2test(tst.IterFuncStop(s)),
                             self.type2test())
            self.assertEqual(self.type2test(c for c in '123'),
                             self.type2test('123'))
            self.assertRaises(TypeError, self.type2test, tst.IterNextOnly(s))
            self.assertRaises(TypeError, self.type2test, tst.IterNoNext(s))
            self.assertRaises(ZeroDivisionError, self.type2test,
                              tst.IterGenExc(s))


    def test_getslice(self):
        super(NoDupTest, self).test_getslice()
        l = [0, 1, 2, 3, 4]
        u = self.type2test(l)
        for i in range(-3, 6):
            self.assertEqual(u[:i], l[:i])
            self.assertEqual(u[i:], l[i:])
            for j in xrange(-3, 6):
                self.assertEqual(u[i:j], l[i:j])


    def test_copy(self):
        u = self.type2test([0, 1])
        v = cp.copy(u)
        w = cp.deepcopy(u)
        self.assertEqual(u, v)
        self.assertEqual(type(u), type(v))
        self.assertEqual(u, w)
        self.assertEqual(type(u), type(w))
        self.assertNotEqual(id(u), id(v))
        self.assertNotEqual(id(u), id(w))
        self.assertNotEqual(id(v), id(w))


    # getting test.seq_tests.test_contains_order to pass is a hopeless
    # proposition, because it is based on raising an exception upon
    # equality testing on the contents; this means that, for example,
    # even creating objects like the test's NoDup([1, StopCompares()])
    # trigger the exception; this is no "corner-case": NoDup depends
    # critically on equality testing for its core functionality, so
    # there is little hope of getting this test to work with NoDup;
    # more importantly, the property the test is testing (strict
    # linear search to test for containment) is one that, NoDup, by
    # design, does *not* satisfy.
    @ut.skip('not applicable (original in test.seq_tests)')
    def test_contains_order(self):
        # see overridden method in tst (test.seq_tests)
        pass

    # copied directly from tst (test.seq_tests)
    # def test_contains_order(self):
    #     # Sequences must test in-order.  If a rich comparison has side
    #     # effects, these will be visible to tests against later members.
    #     # In this test, the "side effect" is a short-circuiting raise.
    #     class DoNotTestEq(Exception):
    #         pass
    #     class StopCompares:
    #         def __eq__(self, other):
    #             raise DoNotTestEq

    #     checkfirst = self.type2test([1, StopCompares()])
    #     self.assertIn(1, checkfirst)
    #     checklast = self.type2test([StopCompares(), 1])
    #     self.assertRaises(DoNotTestEq, checklast.__contains__, 1)


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


    def test_iadd(self):
        super(NoDupTest, self).test_iadd()
        u = [0, 1]
        u += self.type2test([0, 1])
        self.assertEqual(u, [0, 1, 0, 1])


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


    # The test test_getitemoverwriteiter is originally defined in
    # test.seq_tests, and it is overridden in test.test_userlist;
    # these two versions of the test enforce *opposite* invariants: if
    # test.test_userlist.test_getitemoverwriteiter succeeds,
    # test.seq_tests.test_getitemoverwriteiter should fail; if
    # test.seq_tests.test_getitemoverwriteiter succeeds,
    # test.test_userlist.test_getitemoverwriteiter should fail; the
    # version below is a close replica of
    # test.test_userlist.test_getitemoverwriteiter;

    # getting test_getitemoverwriteiter to pass (i.e. to succeed iff
    # "__getitem__ overrides *are* recognized by __iter__") requires
    # implementing a list iterator class for NoDup, as well as
    # overriding NoDup.__iter__; the benefits of doing this do not
    # seem worth the loss in performance resulting from an all-Python
    # list iterator and/or __iter__ method; hence, for now, I'll just
    # replace this test with a no-op;
    #

    # copied (with superficial modifications) from test.test_userlist
    @ut.skip('enforces a performance-costly feature '\
             '(original in test.test_userlist)')
    def test_getitemoverwriteiter(self):
        # Verify that __getitem__ overrides *are* recognized by __iter__
        class T(self.type2test):
            def __getitem__(self, idx):
                return str(idx) + '!!!'
        self.assertEqual(iter(T((1, 2))).next(), '0!!!')


    def test_repeat(self):
        for m in xrange(4):
            s = tuple(range(m))
            for n in xrange(-3, 2):
                self.assertEqual(self.type2test(s*n), self.type2test(s)*n)
            self.assertEqual(self.type2test(s)*(-4), self.type2test([]))
            self.assertEqual(id(s), id(s*1))


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


    def test_index(self):
        u = self.type2test([0, 1])
        self.assertEqual(u.index(0), 0)
        self.assertEqual(u.index(1), 1)
        self.assertRaises(ValueError, u.index, 2)

        u = self.type2test([-2, -1, 0, 0, 1, 2])
        self.assertEqual(u.count(0), 1)
        self.assertEqual(u.index(0), 2)
        self.assertEqual(u.index(0, 2), 2)
        self.assertEqual(u.index(-2, -10), 0)
        self.assertEqual(u.index(0, 2, 3), 2)
        self.assertRaises(ValueError, u.index, 2, 0, -10)

        self.assertRaises(TypeError, u.index)

        class BadExc(Exception):
            pass

        class BadCmp:
            def __eq__(self, other):
                if other == 2:
                    raise BadExc()
                return False

        a = self.type2test([0, 1, 2, 3])
        self.assertRaises(BadExc, a.index, BadCmp())

        a = self.type2test([-2, -1, 0, 0, 1, 2])
        self.assertEqual(a.index(0, -3), 2)
        self.assertEqual(a.index(0, -3, -2), 2)
        self.assertEqual(a.index(0, -4*sys.maxint, 4*sys.maxint), 2)
        self.assertRaises(ValueError, a.index, 0, 4*sys.maxint,-4*sys.maxint)
        self.assertRaises(ValueError, a.index, 2, 0, -10)
        a.remove(0)
        self.assertRaises(ValueError, a.index, 2, 0, 3)
        self.assertEqual(a, self.type2test([-2, -1, 1, 2]))

        # Test modifying the list during index's iteration
        class EvilCmp:
            def __init__(self, victim):
                self.victim = victim
            def __eq__(self, other):
                del self.victim[:]
                return False
            def __hash__(self):
                return id(self)

        a = self.type2test()
        a[:] = [EvilCmp(a) for _ in xrange(100)]
        # This used to seg fault before patch #1005778
        self.assertRaises(ValueError, a.index, None)


    def test_init(self):
        # Iterable arg is optional
        self.assertEqual(self.type2test([]), self.type2test())

        # Init clears previous values
        a = self.type2test([1, 2, 3])
        a.__init__()
        self.assertEqual(a, self.type2test([]))

        # Init overwrites previous values
        a = self.type2test([1, 2, 3])
        a.__init__([4, 5, 6])
        self.assertEqual(a, self.type2test([4, 5, 6]))

        # Mutables always return a new object
        b = self.type2test(a)
        self.assertNotEqual(id(a), id(b))
        self.assertEqual(a, b)


    def test_repr(self):
        l0 = []
        a0 = self.type2test(l0)
        self.assertEqual(str(a0), str(l0))

        l2 = [0, 1, 2]
        a2 = self.type2test(l2)
        self.assertEqual(str(a2), '[0, 1, 2]')


    # the test (in test.list_tests) that the stub below "overrides"
    # focuses on how recursive objects are printed, which is not
    # applicable to the current version of nodup.NoDup
    @ut.skip('not applicable (original in test.list_tests)')
    def test_print(self):
        pass


    def test_set_subscript(self):
        a = self.type2test(range(20))
        self.assertRaises(ValueError, a.__setitem__, slice(0, 10, 0), [1,2,3])
        self.assertRaises(TypeError, a.__setitem__, slice(0, 10), 1)
        self.assertRaises(ValueError, a.__setitem__, slice(0, 10, 2), [20, 21])
        self.assertRaises(TypeError, a.__getitem__, 'x', 1)
        a[slice(2, 10, 3)] = [20, 21, 22]
        self.assertEqual(a, self.type2test([0, 1, 20, 3, 4, 21, 6, 7, 22, 9,
                                            10, 11, 12, 13, 14, 15, 16, 17, 18,
                                            19]))


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
        a[0L] = 11
        a[1L] = 21
        a[2L] = 31
        self.assertEqual(a, self.type2test([11,21,31,3,4]))
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


    def test_setslice(self):
        l = [0, 1]
        a = self.type2test(l)

        for i in range(-5, 5):
            a[:i] = l[:i]
            self.assertEqual(a, l)
            a2 = a[:]
            a2[:i] = a[:i]
            self.assertEqual(a2, a)
            a[i:] = l[i:]
            self.assertEqual(a, l)
            a2 = a[:]
            a2[i:] = a[i:]
            self.assertEqual(a2, a)
            for j in range(-5, 5):
                a[i:j] = l[i:j]
                self.assertEqual(a, l)
                a2 = a[:]
                a2[i:j] = a[i:j]
                self.assertEqual(a2, a)

        aa2 = a2[:]
        aa2[:0] = [-2, -1]
        self.assertEqual(aa2, [-2, -1, 0, 1])
        aa2[0:] = []
        self.assertEqual(aa2, [])

        a = self.type2test([1, 2, 3, 4, 5])
        a[:-1] = [x - 1 for x in a]
        self.assertEqual(a, self.type2test([0, 1, 2, 3, 4, 5]))
        a = self.type2test([1, 2, 3, 4, 5])
        a[1:] = [x + 1 for x in a]
        self.assertEqual(a, self.type2test([1, 2, 3, 4, 5, 6]))
        a = self.type2test([1, 3, 5, 7, 9])
        a[1:-1] = range(2, 9)
        self.assertEqual(a, self.type2test([1, 2, 3, 4, 5, 6, 7, 8, 9]))

        a = self.type2test([])
        a[:] = tuple(range(10))
        self.assertEqual(a, self.type2test(range(10)))

        self.assertRaises(TypeError, a.__setslice__, 0, 1, 5)
        self.assertRaises(TypeError, a.__setitem__, slice(0, 1, 5))

        self.assertRaises(TypeError, a.__setslice__)
        self.assertRaises(TypeError, a.__setitem__)


    def test_extend(self):
        a1 = self.type2test([0])
        a2 = self.type2test((1, 2))
        a = a1[:]
        a.extend(a2)
        self.assertEqual(a, a1 + a2)

        a.extend(self.type2test([]))
        self.assertEqual(a, a1 + a2)

        a.extend([x + 3 for x in a])
        self.assertEqual(a, self.type2test([0, 1, 2, 3, 4, 5]))

        a = self.type2test('spam')
        a.extend('eggs')
        self.assertEqual(a, list('spameg'))

        self.assertRaises(TypeError, a.extend, None)

        self.assertRaises(TypeError, a.extend)


    def test_insert(self):
        a = self.type2test([0, 1, 2])
        a.insert(0, -3)
        a.insert(1, -2)
        a.insert(2, -1)
        self.assertEqual(a, [-3, -2, -1, 0, 1, 2])

        b = a[:]
        b.insert(-2, 'foo')
        b.insert(-200, 'left')
        b.insert(200, 'right')
        self.assertEqual(b, self.type2test(['left', -3, -2, -1, 0,
                                            'foo', 1, 2, 'right']))

        self.assertRaises(TypeError, a.insert)


    def test_remove(self):
        a = self.type2test([0, 1])
        a.remove(1)
        self.assertEqual(a, [0])
        a.remove(0)
        self.assertEqual(a, [])

        self.assertRaises(ValueError, a.remove, 0)

        self.assertRaises(TypeError, a.remove)

        class BadExc(Exception):
            pass

        class BadCmp(int):
            def __eq__(self, other):
                if other == 2:
                    raise BadExc()
                return False

        a = self.type2test([0, 1, 2, 3])
        self.assertRaises(BadExc, a.remove, BadCmp())

        class BadCmp2(str):
            def __eq__(self, other):
                raise BadExc()

        d = self.type2test('abcdefghij')
        d.remove('c')
        self.assertEqual(d, self.type2test('abdefghij'))
        self.assertRaises(ValueError, d.remove, 'c')
        self.assertEqual(d, self.type2test('abdefghij'))

        # Handle comparison errors
        d = self.type2test(['a', 'b', BadCmp2(), 'c'])
        e = self.type2test(d)
        self.assertRaises(BadExc, d.remove, 'c')
        for x, y in zip(d, e):
            # verify that original order and values are retained.
            self.assertIs(x, y)


    def test_sort(self):
        with ts.check_py3k_warnings(
                ("the cmp argument is not supported", DeprecationWarning)):
            self._test_sort()


    def _test_sort(self):
        u = self.type2test([1, 0])
        u.sort()
        self.assertEqual(u, [0, 1])

        u = self.type2test([2,1,0,-1,-2])
        u.sort()
        self.assertEqual(u, self.type2test([-2,-1,0,1,2]))

        self.assertRaises(TypeError, u.sort, 42, 42)

        def revcmp(a, b):
            return cmp(b, a)
        u.sort(revcmp)
        self.assertEqual(u, self.type2test([2,1,0,-1,-2]))

        # The following dumps core in unpatched Python 1.5:
        def myComparison(x,y):
            return cmp(x%3, y%7)
        z = self.type2test(range(12))
        z.sort(myComparison)

        self.assertRaises(TypeError, z.sort, 2)

        def selfmodifyingComparison(x,y):
            z.append(1)
            return cmp(x, y)
        self.assertRaises(ValueError, z.sort, selfmodifyingComparison)

        self.assertRaises(TypeError, z.sort, lambda x, y: 's')

        self.assertRaises(TypeError, z.sort, 42, 42, 42, 42)


    def test_extendedslicing(self):
        #  subscript
        a = self.type2test([0,1,2,3,4])

        #  deletion
        del a[::2]
        self.assertEqual(a, self.type2test([1,3]))
        a = self.type2test(range(5))
        del a[1::2]
        self.assertEqual(a, self.type2test([0,2,4]))
        a = self.type2test(range(5))
        del a[1::-2]
        self.assertEqual(a, self.type2test([0,2,3,4]))
        a = self.type2test(range(10))
        del a[::1000]
        self.assertEqual(a, self.type2test([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        #  assignment
        a = self.type2test(range(1, 11))
        a[::2] = [-1, -3, -5, -7, -9]
        self.assertEqual(a, self.type2test([-1, 2, -3, 4, -5, 6, -7, 8, -9, 10]))
        a = self.type2test(range(10))
        a[::-4] = [-9, -5, -1]
        self.assertEqual(a, self.type2test([0, -1, 2, 3, 4, -5, 6, 7, 8, -9]))
        a = self.type2test(range(4))
        a[::-1] = a
        self.assertEqual(a, self.type2test([3, 2, 1, 0]))
        a = self.type2test(range(10))
        b = a[:]
        c = a[:]
        a[2:3] = self.type2test(['two', 'elements'])
        b[slice(2,3)] = self.type2test(['two', 'elements'])
        c[2:3:] = self.type2test(['two', 'elements'])
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        a = self.type2test(range(10))
        a[::2] = tuple(range(10, 15))
        self.assertEqual(a, self.type2test([10, 1, 11, 3, 12, 5, 13, 7, 14, 9]))
        # test issue7788
        a = self.type2test(range(10))
        del a[9::1<<333]


    def test_mixedcmp(self):
        u = self.type2test([0, 1])
        self.assertEqual(u, [0, 1])
        self.assertNotEqual(u, [0])
        self.assertNotEqual(u, [0, 2])


    def test_mixedadd(self):
        u = self.type2test([0, 1])
        self.assertEqual(u + [], u)
        self.assertEqual(u + [2], [0, 1, 2])




def test_main(verbose=None):
    with ts.check_py3k_warnings(('.+__(get|set|del)slice__ has been removed',
                                 DeprecationWarning)):
        ts.verbose = verbose
        ts.run_unittest(NoDupTest)
        ts.run_doctest(nd, verbose)


if __name__ == '__main__':
    test_main(verbose=True)
