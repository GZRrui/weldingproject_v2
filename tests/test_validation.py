import unittest
from validate import stats


class MetricTests(unittest.TestCase):
    def test_signed_bias_and_absolute_error_are_distinct(self):
        result = stats([1, 2, 3], [2, 1, 3])
        self.assertEqual(result['n'], 3)
        self.assertEqual(result['bias'], 0)
        self.assertAlmostEqual(result['mae'], 2/3)
        self.assertEqual(result['worst'], 1)


if __name__ == '__main__':
    unittest.main()
