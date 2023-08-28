import unittest
from wsknn.preprocessing.structure.session import Sessions


SESSION_KEY = 'sid'
ITEM_KEY = 'pid'
T_KEY = 'dt'
ACTION_KEY = 'action'
ACTION_WEIGHTS = {'products_view': 0.1,
                  'purchase': 1,
                  'add_to_wishlist': 0.3,
                  'add_to_cart': 0.5}

EVENT1 = {
    'sid': '0',
    'pid': 'a',
    'dt': 1,
    'action': 'products_view'
}

EVENT2 = {
    'sid': '0',
    'pid': 'b',
    'dt': 2,
    'action': 'products_view'
}

EVENT3 = {
    'sid': '0',
    'pid': 'c',
    'dt': 3,
    'action': 'products_view'
}

EVENT4 = {
    'sid': '0',
    'dt': 4,
    'action': 'purchase'
}

EVENT_4_PRODUCTS = ['a', 'c']

# TODO: update Sessions class test
class TestSessionsClass(unittest.TestCase):

    def test_init(self):
        sessions = Sessions(event_session_key=SESSION_KEY,
                            event_product_key=ITEM_KEY,
                            event_time_key=T_KEY,
                            event_action_key=ACTION_KEY,
                            event_action_weights=ACTION_WEIGHTS)

        self.assertIsInstance(sessions, Sessions)

        EXPECTED_MAP = dict()
        EXPECTED_TIME_START = 1_000_000_000_000_000
        EXPECTED_TIME_END = 0
        EXPECTED_LONGEST_ITEMS_SEQUENCE = 0
        EXPECTED_NUMBER_OF_SESSIONS = 0

        self.assertEqual(EXPECTED_MAP, sessions.session_items_actions_map)
        self.assertEqual(EXPECTED_TIME_START, sessions.time_start)
        self.assertEqual(EXPECTED_TIME_END, sessions.time_end)
        self.assertEqual(EXPECTED_LONGEST_ITEMS_SEQUENCE, sessions.longest_items_vector_size)
        self.assertEqual(EXPECTED_NUMBER_OF_SESSIONS, sessions.number_of_sessions)

    def test_append(self):

        expected_number_of_sessions = 1
        expected_longest_items_seq = 1
        expected_ts = 1
        expected_tend = 1
        session_id = '0'

        sessions = Sessions(event_session_key=SESSION_KEY,
                            event_product_key=ITEM_KEY,
                            event_time_key=T_KEY,
                            event_action_key=ACTION_KEY,
                            event_action_weights=ACTION_WEIGHTS)
        sessions.append(EVENT1)

        self.assertEqual(expected_ts, sessions.time_start)
        self.assertEqual(expected_tend, sessions.time_end)
        self.assertEqual(expected_longest_items_seq, sessions.longest_items_vector_size)
        self.assertEqual(expected_number_of_sessions, sessions.number_of_sessions)
        self.assertIn(session_id, sessions.session_items_actions_map)

    def test_save_and_load(self):

        EMPTY_SESSION_PATH = 'tdata/test_save_empty_session.pkl'
        SESSION_PATH = 'tdata/test_save_session.pkl'

        sessions = Sessions(event_session_key=SESSION_KEY,
                            event_product_key=ITEM_KEY,
                            event_time_key=T_KEY,
                            event_action_key=ACTION_KEY,
                            event_action_weights=ACTION_WEIGHTS)
        # Save empty
        sessions.save(EMPTY_SESSION_PATH)
        # Append event
        sessions.append(EVENT1)
        # Save
        sessions.save(SESSION_PATH)

        del sessions

        # Load empty
        empty_sessions = Sessions(event_session_key=SESSION_KEY,
                                  event_product_key=ITEM_KEY,
                                  event_time_key=T_KEY,
                                  event_action_key=ACTION_KEY,
                                  event_action_weights=ACTION_WEIGHTS)
        empty_sessions.load(EMPTY_SESSION_PATH)

        # Load processed
        sessions = Sessions(event_session_key=SESSION_KEY,
                            event_product_key=ITEM_KEY,
                            event_time_key=T_KEY,
                            event_action_key=ACTION_KEY,
                            event_action_weights=ACTION_WEIGHTS)
        sessions.load(SESSION_PATH)

        # TESTS - EMPTY
        empty_expected_map = dict()
        empty_expected_ts = 1_000_000_000_000_000
        empty_expected_tend = 0
        empty_expected_seq = 0
        empty_expected_no_sessions = 0
        self.assertEqual(empty_expected_map, empty_sessions.session_items_actions_map)
        self.assertEqual(empty_expected_ts,empty_sessions.time_start)
        self.assertEqual(empty_expected_tend, empty_sessions.time_end)
        self.assertEqual(empty_expected_seq, empty_sessions.longest_items_vector_size)
        self.assertEqual(empty_expected_no_sessions, empty_sessions.number_of_sessions)

        # TESTS - processed
        no_sessions = 1
        longest_seq = 1
        ts = 1
        tend = 1
        session_id = '0'
        self.assertEqual(ts, sessions.time_start)
        self.assertEqual(tend, sessions.time_end)
        self.assertEqual(longest_seq, sessions.longest_items_vector_size)
        self.assertEqual(no_sessions, sessions.number_of_sessions)
        self.assertIn(session_id, sessions.session_items_actions_map)

    def test_add(self):
        sessions1 = Sessions(event_session_key=SESSION_KEY,
                            event_product_key=ITEM_KEY,
                            event_time_key=T_KEY,
                            event_action_key=ACTION_KEY,
                            event_action_weights=ACTION_WEIGHTS)
        sessions2 = Sessions(event_session_key=SESSION_KEY,
                            event_product_key=ITEM_KEY,
                            event_time_key=T_KEY,
                            event_action_key=ACTION_KEY,
                            event_action_weights=ACTION_WEIGHTS)
        sessions3 = Sessions(event_session_key=SESSION_KEY,
                            event_product_key=ITEM_KEY,
                            event_time_key=T_KEY,
                            event_action_key=ACTION_KEY,
                            event_action_weights=ACTION_WEIGHTS)
        sessions4 = Sessions(event_session_key=SESSION_KEY,
                             event_product_key=ITEM_KEY,
                             event_time_key=T_KEY,
                             event_action_key=ACTION_KEY,
                             event_action_weights=ACTION_WEIGHTS)
        sessions1.append(EVENT1)
        sessions2.append(EVENT2)
        sessions3.append(EVENT3)

        sessions12 = sessions1 + sessions2

        sessions123 = sessions12 + sessions3
        sessions123.update_weights_of_purchase_session(EVENT4['sid'],
                                                       additive_factor=1,
                                                       bought_products=EVENT_4_PRODUCTS)

        weights12 = sessions12.session_items_actions_map['0'][-1]
        weights123 = sessions123.session_items_actions_map['0'][-1]

        w12 = sum(weights12)
        w123 = sum(weights123)
        diff = w123 - w12
        expected_diff = 2

        self.assertEqual(expected_diff, diff)

    def test_no_actions(self):
        sessions = Sessions(event_session_key=SESSION_KEY,
                            event_product_key=ITEM_KEY,
                            event_time_key=T_KEY,
                            event_action_key=None,
                            event_action_weights=None)

        self.assertIsInstance(sessions, Sessions)

        EXPECTED_MAP = dict()
        EXPECTED_TIME_START = 1_000_000_000_000_000
        EXPECTED_TIME_END = 0
        EXPECTED_LONGEST_ITEMS_SEQUENCE = 0
        EXPECTED_NUMBER_OF_SESSIONS = 0

        self.assertEqual(EXPECTED_MAP, sessions.session_items_actions_map)
        self.assertEqual(EXPECTED_TIME_START, sessions.time_start)
        self.assertEqual(EXPECTED_TIME_END, sessions.time_end)
        self.assertEqual(EXPECTED_LONGEST_ITEMS_SEQUENCE, sessions.longest_items_vector_size)
        self.assertEqual(EXPECTED_NUMBER_OF_SESSIONS, sessions.number_of_sessions)


if __name__ == '__main__':
    unittest.main()
