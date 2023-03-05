import unittest
from wsknn.preprocessing.utils.calc import get_smaller_value


class TestGetSmallerValue(unittest.TestCase):

    def test_int_inputs(self):
        INPUT1 = 10
        INPUT2 = 0
        INPUT3 = -10

        EXPECTED_OUTPUT_1 = 0
        EXPECTED_OUTPUT_2 = 0
        EXPECTED_OUTPUT_3 = -10

        t1 = get_smaller_value(INPUT1, INPUT2)
        t2 = get_smaller_value(INPUT2, INPUT2)
        t3 = get_smaller_value(INPUT2, INPUT3)

        # Simple test
        self.assertEqual(t1, EXPECTED_OUTPUT_1)
        self.assertEqual(t2, EXPECTED_OUTPUT_2)
        self.assertEqual(t3, EXPECTED_OUTPUT_3)

    def test_float_inputs(self):
        INPUT1 = 9.999999
        INPUT2 = 0.0
        INPUT3 = 10.

        EXPECTED_OUTPUT_1 = 9.999999
        EXPECTED_OUTPUT_2 = 0.0

        t1 = get_smaller_value(INPUT1, INPUT3)
        t2 = get_smaller_value(INPUT2, INPUT2)

        # Simple test
        self.assertEqual(t1, EXPECTED_OUTPUT_1)
        self.assertEqual(t2, EXPECTED_OUTPUT_2)


if __name__ == '__main__':
    unittest.main()
