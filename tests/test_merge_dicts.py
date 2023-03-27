import unittest
from wsknn.preprocessing.utils.transform import merge_dicts


class TestMergeDicts(unittest.TestCase):

    def test_merging(self):
        d1 = {
            'a': [
                [0, 1, 2],
                [10, 20, 30],
                ['a', 'ac', 'as'],
                [0.1, 0.2, 0.3]
            ],
            'c': [
                ['a'],
                [1],
                ['l'],
                [1]
            ]
        }
        d2 = {
            'd': [
                [4, 4, 4],
                [10, 20, 545],
                ['as', 'rt', 'k'],
                [0.2, 0.5, 0.8]
            ],
            'a': [
                [7],
                [2],
                ['c'],
                [0.3]
            ],
            'c': [
                ['a', 'b'],
                [3, 4],
                ['ac', 'a'],
                [0.1, 0.1]
            ]
        }

        EXPECTED_MERGED_FULLY = {
            'a': [
                [7, 0, 1, 2],
                [2, 10, 20, 30],
                ['c', 'a', 'ac', 'as'],
                [0.3, 0.1, 0.2, 0.3]
            ],
            'c': [
                ['a', 'a', 'b'],
                [1, 3, 4],
                ['l', 'ac', 'a'],
                [1, 0.1, 0.1]
            ],
            'd': [
                [4, 4, 4],
                [10, 20, 545],
                ['as', 'rt', 'k'],
                [0.2, 0.5, 0.8]
            ]
        }

        EXPECTED_MERGED_PARTIALLY = {
            'a': [
                [7, 0, 1, 2],
                [2, 10, 20, 30],
                ['c', 'a', 'ac', 'as'],
                [0.3, 0.1, 0.2, 0.3]
            ],
            'c': [
                ['a', 'b'],
                [3, 4],
                ['ac', 'a'],
                [0.1, 0.1]
            ],
            'd': [
                [4, 4, 4],
                [10, 20, 545],
                ['as', 'rt', 'k'],
                [0.2, 0.5, 0.8]
            ]
        }

        merged = merge_dicts(d1, d2)

        # Merged fully
        for k in EXPECTED_MERGED_FULLY.keys():
            self.assertIn(k, merged)
            self.assertEqual(EXPECTED_MERGED_FULLY[k], merged[k])


if __name__ == '__main__':
    unittest.main()
