import unittest
from wsknn.preprocessing.utils.transform import parse_seconds_to_dt, parse_dt_to_seconds


INPUT_DATE_LONG = '2021-12-05T00:00:00.000Z'
INPUT_DATE_SHORT = '2021-12-05T00:00:00'
INPUT_DATE_TIMESTAMP = 1638658800
WRONG_INPUT = '2021-01-01'


class TestParseTime(unittest.TestCase):

    def test_dt_to_seconds(self):
        output_long = parse_dt_to_seconds(INPUT_DATE_LONG)
        output_short = parse_dt_to_seconds(INPUT_DATE_SHORT)
        self.assertEqual(output_short, output_long)
        self.assertRaises(TypeError, parse_dt_to_seconds, WRONG_INPUT)

    def test_seconds_to_dt(self):

        output = parse_seconds_to_dt(INPUT_DATE_TIMESTAMP)
        self.assertIsInstance(output, str)


if __name__ == '__main__':
    unittest.main()
