# Distributed under the MIT License.
# See LICENSE.txt for details.

import math
import unittest

import numpy as np

from spectre.DataStructures import Matrix


class TestMatrix(unittest.TestCase):
    def test_shape(self):
        M = Matrix(3, 5)
        self.assertEqual(M.shape, (3, 5))

    def test_getitem(self):
        M = Matrix(3, 4)
        M[1, 2] = 3
        M[0, 0] = 0
        M[2, 1] = 1
        self.assertEqual(M[1, 2], 3)
        self.assertEqual(M[0, 0], 0)
        self.assertEqual(M[2, 1], 1)

    def test_string(self):
        M = Matrix(2, 2)
        M[0, 0] = 1
        M[0, 1] = 2
        M[1, 0] = 3
        M[1, 1] = 4
        self.assertEqual(
            str(M),
            "(            1            2 )\n(            3            4 )\n",
        )

    def test_bounds_check(self):
        a = Matrix(2, 2)
        self.assertRaises(RuntimeError, lambda: a[5, 2])

        def assignment_test():
            a[5, 2] = 8

        self.assertRaises(RuntimeError, assignment_test)

    def test_numpy_interoperability(self):
        M = Matrix(2, 2)
        M[0, 0] = 1.0
        M[1, 0] = 2.7
        M[0, 1] = 5.42
        M[1, 1] = -2.3
        # Convert Matrix to Numpy array
        A = np.array(M)
        self.assertTrue(isinstance(A, np.ndarray))
        self.assertEquals(A[0, 0], 1.0)
        self.assertEquals(A[1, 0], 2.7)
        self.assertEquals(A[0, 1], 5.42)
        self.assertEquals(A[1, 1], -2.3)
        # Convert Numpy array to Matrix
        M_from_array = Matrix(A)
        self.assertEquals(M_from_array[0, 0], 1.0)
        self.assertEquals(M_from_array[1, 0], 2.7)
        self.assertEquals(M_from_array[0, 1], 5.42)
        self.assertEquals(M_from_array[1, 1], -2.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
