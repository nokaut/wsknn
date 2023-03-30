import unittest
from wsknn.preprocessing.structure.item import Items


EVENT_PRODUCT_KEY = 'product'
EVENT_SESSION_KEY = 'session'
EVENT_TIME_KEY = 'time'

class TestItemsClass(unittest.TestCase):

    def test_init(self):
        items = Items(event_product_key=EVENT_PRODUCT_KEY,
                      event_session_key=EVENT_SESSION_KEY,
                      event_time_key=EVENT_TIME_KEY)
        self.assertIsInstance(items, Items)

        EXPECTED_MAP = dict()
        EXPECTED_TIME_START = 1_000_000_000_000_000
        EXPECTED_TIME_END = 0
        EXPECTED_LONGEST_SESSIONS_VECTOR_SIZE = 0
        EXPECTED_NUMBER_OF_ITEMS = 0

        self.assertEqual(EXPECTED_MAP, items.item_sessions_map)
        self.assertEqual(EXPECTED_TIME_START, items.time_start)
        self.assertEqual(EXPECTED_TIME_END, items.time_end)
        self.assertEqual(EXPECTED_LONGEST_SESSIONS_VECTOR_SIZE, items.longest_sessions_vector_size)
        self.assertEqual(EXPECTED_NUMBER_OF_ITEMS, items.number_of_items)

    def test_append(self):
        EVENT = {
            EVENT_PRODUCT_KEY: 'test_product',
            EVENT_SESSION_KEY: 'test_session',
            EVENT_TIME_KEY: 0
        }
        EXPECTED_NUMBER_OF_ITEMS = 1
        EXPECTED_LONGEST_SESSION_SIZE = 1
        EXPECTED_TIME_START = 0
        EXPECTED_TIME_END = 0
        PRODUCT = 'test_product'

        items = Items(event_product_key=EVENT_PRODUCT_KEY,
                      event_session_key=EVENT_SESSION_KEY,
                      event_time_key=EVENT_TIME_KEY)
        items.append(EVENT)

        self.assertEqual(EXPECTED_TIME_START, items.time_start)
        self.assertEqual(EXPECTED_TIME_END, items.time_end)
        self.assertEqual(EXPECTED_LONGEST_SESSION_SIZE, items.longest_sessions_vector_size)
        self.assertEqual(EXPECTED_NUMBER_OF_ITEMS, items.number_of_items)
        self.assertIn(PRODUCT, items.item_sessions_map)

    def test_save_and_load(self):
        EVENT = {
            EVENT_PRODUCT_KEY: 'test_product',
            EVENT_SESSION_KEY: 'test_session',
            EVENT_TIME_KEY: 0
        }

        EMPTY_ITEM_PATH = 'tdata/test_save_empty_item.pkl'
        ITEM_PATH = 'tdata/test_save_item.pkl'

        items = Items(event_product_key=EVENT_PRODUCT_KEY,
                      event_session_key=EVENT_SESSION_KEY,
                      event_time_key=EVENT_TIME_KEY)

        # Save empty
        items.save(EMPTY_ITEM_PATH)
        # Append event
        items.append(EVENT)
        # Save
        items.save(ITEM_PATH)

        del items

        # Load empty
        empty_items = Items(event_product_key=EVENT_PRODUCT_KEY,
                            event_session_key=EVENT_SESSION_KEY,
                            event_time_key=EVENT_TIME_KEY)
        empty_items.load(EMPTY_ITEM_PATH)

        # Load processed
        items = Items(event_product_key=EVENT_PRODUCT_KEY,
                      event_session_key=EVENT_SESSION_KEY,
                      event_time_key=EVENT_TIME_KEY)
        items.load(ITEM_PATH)

        # TESTS - EMPTY
        EMPTY_EXPECTED_MAP = dict()
        EMPTY_EXPECTED_TIME_START = 1_000_000_000_000_000
        EMPTY_EXPECTED_TIME_END = 0
        EMPTY_EXPECTED_LONGEST_SESSIONS_VECTOR_SIZE = 0
        EMPTY_EXPECTED_NUMBER_OF_ITEMS = 0
        self.assertEqual(EMPTY_EXPECTED_MAP, empty_items.item_sessions_map)
        self.assertEqual(EMPTY_EXPECTED_TIME_START, empty_items.time_start)
        self.assertEqual(EMPTY_EXPECTED_TIME_END, empty_items.time_end)
        self.assertEqual(EMPTY_EXPECTED_LONGEST_SESSIONS_VECTOR_SIZE, empty_items.longest_sessions_vector_size)
        self.assertEqual(EMPTY_EXPECTED_NUMBER_OF_ITEMS, empty_items.number_of_items)

        # TESTS - processed
        EXPECTED_NUMBER_OF_ITEMS = 1
        EXPECTED_LONGEST_SESSION_SIZE = 1
        EXPECTED_TIME_START = 0
        EXPECTED_TIME_END = 0
        PRODUCT = 'test_product'
        self.assertEqual(EXPECTED_TIME_START, items.time_start)
        self.assertEqual(EXPECTED_TIME_END, items.time_end)
        self.assertEqual(EXPECTED_LONGEST_SESSION_SIZE, items.longest_sessions_vector_size)
        self.assertEqual(EXPECTED_NUMBER_OF_ITEMS, items.number_of_items)
        self.assertIn(PRODUCT, items.item_sessions_map)

    def test_add(self):
        EVENT_0 = {
            EVENT_PRODUCT_KEY: 'test_product',
            EVENT_SESSION_KEY: 'test_session',
            EVENT_TIME_KEY: 0
        }
        EVENT_1 = {
            EVENT_PRODUCT_KEY: 'other_test_product',
            EVENT_SESSION_KEY: 'test_session',
            EVENT_TIME_KEY: 1000
        }
        EVENT_2 = {
            EVENT_PRODUCT_KEY: 'test_product',
            EVENT_SESSION_KEY: 'other_test_session',
            EVENT_TIME_KEY: 500
        }

        items_0 = Items(event_product_key=EVENT_PRODUCT_KEY,
                        event_session_key=EVENT_SESSION_KEY,
                        event_time_key=EVENT_TIME_KEY)
        items_0.append(EVENT_0)
        items_1 = Items(event_product_key=EVENT_PRODUCT_KEY,
                        event_session_key=EVENT_SESSION_KEY,
                        event_time_key=EVENT_TIME_KEY)
        items_1.append(EVENT_1)
        items_2 = Items(event_product_key=EVENT_PRODUCT_KEY,
                        event_session_key=EVENT_SESSION_KEY,
                        event_time_key=EVENT_TIME_KEY)
        items_2.append(EVENT_2)

        # keys: 'test_product', 'other_test_product'
        items_01 = items_0 + items_1

        # keys: 'other_test_product', 'test_product'
        # time_start = 500
        # time_end = 1000
        items_12 = items_1 + items_2

        # keys: 'test_product', 'other_test_product'
        # time_start = 0
        # time_end = 1000
        items_012 = items_0 + items_1 + items_2

        # keys: 'test_product'
        # time_start = 0
        # time_end = 0
        items_00 = items_0 + items_0

        EXPECTED_KEYS_01 = ['test_product', 'other_test_product']
        EXPECTED_KEYS_12 = EXPECTED_KEYS_01
        EXPECTED_KEYS_012 = EXPECTED_KEYS_01
        EXPECTED_KEYS_00 = ['test_product']
        EXPECTED_TIME_START_01 = 0
        EXPECTED_TIME_END_01 = 1000
        EXPECTED_TIME_START_12 = 500
        EXPECTED_TIME_END_12 = 1000
        EXPECTED_TIME_START_012 = 0
        EXPECTED_TIME_END_012 = 1000
        EXPECTED_TIME_START_00 = 0
        EXPECTED_TIME_END_00 = 0

        for k in items_01.item_sessions_map.keys():
            self.assertIn(k, EXPECTED_KEYS_01)
        for k in items_12.item_sessions_map.keys():
            self.assertIn(k, EXPECTED_KEYS_12)
        for k in items_012.item_sessions_map.keys():
            self.assertIn(k, EXPECTED_KEYS_012)
        for k in items_00.item_sessions_map.keys():
            self.assertIn(k, EXPECTED_KEYS_00)

        self.assertEqual(items_01.time_start, EXPECTED_TIME_START_01)
        self.assertEqual(items_01.time_end, EXPECTED_TIME_END_01)
        self.assertEqual(items_12.time_start, EXPECTED_TIME_START_12)
        self.assertEqual(items_12.time_end, EXPECTED_TIME_END_12)
        self.assertEqual(items_012.time_start, EXPECTED_TIME_START_012)
        self.assertEqual(items_012.time_end, EXPECTED_TIME_END_012)
        self.assertEqual(items_00.time_start, EXPECTED_TIME_START_00)
        self.assertEqual(items_00.time_end, EXPECTED_TIME_END_00)

    def test_str(self):
        EVENT = {
            EVENT_PRODUCT_KEY: 'test_product',
            EVENT_SESSION_KEY: 'test_session',
            EVENT_TIME_KEY: 0
        }

        empty_items = Items(event_product_key=EVENT_PRODUCT_KEY,
                            event_session_key=EVENT_SESSION_KEY,
                            event_time_key=EVENT_TIME_KEY)
        items = Items(event_product_key=EVENT_PRODUCT_KEY,
                      event_session_key=EVENT_SESSION_KEY,
                      event_time_key=EVENT_TIME_KEY)
        items.append(EVENT)

        EMPTY_MSG = 'Empty object. Append events or load pickled item-sessions map!'
        self.assertEqual(empty_items.__str__(), EMPTY_MSG)


if __name__ == '__main__':
    unittest.main()
