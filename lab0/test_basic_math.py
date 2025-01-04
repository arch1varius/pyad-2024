import random
import unittest
import scipy.stats

import basic_math


class MATHTestCase(unittest.TestCase):
    def test_matrix_multiplication(self):
        a1 = [[1, 2, 3],
             [4, 5, 6]]
        b1 = [[7, 8],
             [9, 10],
             [11, 12]]
        self.assertEqual(basic_math.matrix_multiplication(a1, b1), [[58, 64],[139, 154]])

        a2 = [[1, 2],
             [3, 4]]
        b2 = [[1, 2]]
        with self.assertRaises(ValueError):
            basic_math.matrix_multiplication(a2, b2)
        
        a3 = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        b3 = [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        self.assertEqual(basic_math.matrix_multiplication(a3, b3), a3)

        a4 = [[1, 2, 3],
             [4, 5, 6]]
        b4 =[[0, 0],
             [0, 0],
             [0, 0]]
        self.assertEqual(basic_math.matrix_multiplication(a4, b4), [[0, 0], [0, 0]])
        
        a5 = [[1, 2, 3]]
        b5 = [[4],
             [5],
             [6]]
        self.assertEqual(basic_math.matrix_multiplication(a5, b5), [[32]])

        a6 = [[3]]
        b6 = [[4]]
        self.assertEqual(basic_math.matrix_multiplication(a6, b6), [[12]])


    def test_functions(self):
        coeffs1 = "1 0 -4"
        coeffs2 = "1 -2 0"
        self.assertEqual(basic_math.functions(coeffs1, coeffs2), [(2, 0)])

        coeffs3 = "1 0 4"
        coeffs4 = "1 0 1"
        self.assertEqual(basic_math.functions(coeffs3, coeffs4), [])

        coeffs5 = "1 2 1"
        coeffs6 = "1 2 1"
        self.assertIsNone(basic_math.functions(coeffs5, coeffs6))

        coeffs7 = "1 2 3"
        coeffs8 = "1 2 1"
        self.assertEqual(basic_math.functions(coeffs7, coeffs8), [])

        coeffs9 = "0 2 -1"
        coeffs10 = "1 -4 4"
        self.assertEqual(basic_math.functions(coeffs9, coeffs10), [(1, 1), (5, 9)])


    def test_skew(self):
        x1 = [2,3,5,7,8]
        x2 = [2,3,2,5,7,2,2,8]
        self.assertEqual(basic_math.skew(x1), round(scipy.stats.skew(x1), 2))
        self.assertEqual(basic_math.skew(x2), round(scipy.stats.skew(x2), 2))

        random.seed(100)
        random_floats = [random.random() for _ in range(10000)]
        random_integers = [random.randint(1, 99) for _ in range(10000)]
        self.assertEqual(basic_math.skew(random_floats), round(scipy.stats.skew(random_floats), 2))
        self.assertEqual(basic_math.skew(random_integers), round(scipy.stats.skew(random_integers), 2))


    def test_kurtosis(self):
        x1 = [2,3,5,7,8]
        x2 = [2,3,2,5,7,2,2,8]
        self.assertEqual(basic_math.kurtosis(x1), round(scipy.stats.kurtosis(x1), 2))
        self.assertEqual(basic_math.kurtosis(x2), round(scipy.stats.kurtosis(x2), 2))

        random.seed(100)
        random_floats = [random.random() for _ in range(10000)]
        random_integers = [random.randint(1, 99) for _ in range(10000)]
        self.assertEqual(basic_math.kurtosis(random_floats), round(scipy.stats.kurtosis(random_floats), 2))
        self.assertEqual(basic_math.kurtosis(random_integers), round(scipy.stats.kurtosis(random_integers), 2))
