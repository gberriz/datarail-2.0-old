import unittest
from unittest import TestCase
from itertools import product, izip
from numpy import ndindex

from icbp45_utils import coords_to_index_vector as c2iv
from icbp45_utils import index_vector_to_field_number as iv2fn

class SDCIterTestCase(TestCase):

    def test_coords_to_index_vector(self):
        comps = [list('ABCD'),
                 ['%02d' % i for i in range(5)],
                 list('uvwxyz')]
        shape = tuple(map(len, comps))
        for abc, ijk in izip(product(*comps),
                             ndindex(shape)):
            print '%s == c2iv(%s, %s)' % (list(ijk), list(abc), comps)
            self.assertEqual(list(ijk), c2iv(list(abc), comps))


    def test_index_vector_to_field_number(self):
        shape = (4, 5, 6)
        n = 0
        for ijk in ndindex(shape):
            print '%d == iv2fn(%s, %s)' % (n, ijk, shape)
            self.assertEqual(n, iv2fn(list(ijk), shape))
            n += 1


if __name__ == '__main__':
    unittest.main()
