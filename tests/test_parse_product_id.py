import unittest
from preprocessing.utils.transform import parse_product_id, parse_product_id_from_product_context


class TestParseEventType(unittest.TestCase):

    def test_parse_id(self):
        product_id_str = '123'
        product_id_custom_dict = {'product_id': '456'}
        product_id_dict = {'$oid': '789'}

        EXPECTED_STR = '123'
        EXPECTED_DICT = '789'
        EXPECTED_ERROR = TypeError

        pstr = parse_product_id(product_id_str)
        pdict = parse_product_id(product_id_dict)

        self.assertEqual(EXPECTED_STR, pstr)
        self.assertEqual(EXPECTED_DICT, pdict)

        self.assertRaises(EXPECTED_ERROR, parse_product_id, product_id_custom_dict)

    def test_parse_id_from_product_context(self):
        pid = {'product': {
            '_id': {
                '$oid': '123'
            }
        }}
        parsed = parse_product_id_from_product_context([pid])
        self.assertEqual(parsed[0], '123')


if __name__ == '__main__':
    unittest.main()
