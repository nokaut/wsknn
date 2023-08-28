import unittest

from wsknn.preprocessing.parse_static import parse_files
from wsknn.preprocessing.structure.item import Items
from wsknn.preprocessing.structure.session import Sessions

POSSIBLE_ACTIONS = {'products_view': 0.1,
                    'purchase': 1,
                    'add_to_wishlist': 0.3,
                    'add_to_cart': 0.5}

PURCHASE_ACTION_NAME = 'purchase'
SESSIONS = ['tdata/events1.json',
            'tdata/events2.json',
            'tdata/events3.json']

SESSION_KEY = 'sid'
PRODUCT_KEY = 'pid'
ACTION_KEY = 'act'
TIME_KEY = 'ts'


class TestParseFn(unittest.TestCase):

    def test_instances(self):

        for fs in SESSIONS:
            out_item, out_session = parse_files(fs,
                                                session_id_key=SESSION_KEY,
                                                product_key=PRODUCT_KEY,
                                                action_key=ACTION_KEY,
                                                time_key=TIME_KEY,
                                                allowed_actions=POSSIBLE_ACTIONS,
                                                purchase_action_name=PURCHASE_ACTION_NAME,
                                                time_to_numeric=True)

            self.assertIsInstance(out_item, Items)
            self.assertIsInstance(out_session, Sessions)

    def test_output_values(self):

        base_items, base_sessions = parse_files(SESSIONS[0],
                                                session_id_key=SESSION_KEY,
                                                product_key=PRODUCT_KEY,
                                                action_key=ACTION_KEY,
                                                time_key=TIME_KEY,
                                                allowed_actions=POSSIBLE_ACTIONS,
                                                purchase_action_name=PURCHASE_ACTION_NAME,
                                                time_to_numeric=True)

        base_items_keys = list(base_items.item_sessions_map.keys()).copy()
        base_sessions_keys = list(base_sessions.session_items_actions_map.keys()).copy()

        other_items_keys = [base_items_keys]
        other_sessions_keys = [base_sessions_keys]

        for fs in SESSIONS[1:]:
            out_item, out_session = parse_files(fs,
                                                allowed_actions=POSSIBLE_ACTIONS,
                                                purchase_action_name=PURCHASE_ACTION_NAME,
                                                session_id_key=SESSION_KEY,
                                                product_key=PRODUCT_KEY,
                                                action_key=ACTION_KEY,
                                                time_key=TIME_KEY,
                                                time_to_numeric=True,
                                                time_to_datetime=False,
                                                datetime_format='')
            base_items = base_items + out_item
            base_sessions = base_sessions + out_session

            # Append keys
            other_items_keys.append(
                list(out_item.item_sessions_map.keys())
            )
            other_sessions_keys.append(
                list(out_session.session_items_actions_map.keys())
            )

        # Check items keys
        ikeys = set(base_items.item_sessions_map.keys())
        for ik in other_items_keys:
            ikset = set(ik)
            isik = ikset.issubset(ikeys)
            msg = 'Missing keys in items-sessions map dictionary!'
            self.assertTrue(isik, msg)

        # Check sessions keys
        skeys = set(base_sessions.session_items_actions_map.keys())
        for sk in other_sessions_keys:
            skset = set(sk)
            issk = skset.issubset(skeys)
            msg = 'Missing keys in sessions-items map dictionary!'
            self.assertTrue(issk, msg)

    def test_session_weighting(self):
        sample_session = Sessions(event_session_key='sid',
                                  event_product_key='pid',
                                  event_time_key='dt',
                                  event_action_key='action',
                                  event_action_weights={
                                      'products_view': 0.1,
                                      'purchase': 1
                                  })
        event1 = {
            'sid': '0',
            'pid': 'a',
            'dt': 1,
            'action': 'products_view'
        }
        event2 = {
            'sid': '0',
            'pid': 'b',
            'dt': 2,
            'action': 'products_view'
        }
        event3 = {
            'sid': '0',
            'pid': 'c',
            'dt': 3,
            'action': 'products_view'
        }
        event4 = {
            'sid': '0',
            'dt': 4,
            'action': 'purchase'
        }

        event_4_products = ['a', 'c']

        sample_session.append(event1)
        sample_session.append(event2)
        sample_session.append(event3)
        sample_session.update_weights_of_purchase_session(event4['sid'], 1, bought_products=event_4_products)

        expected_output = {
            '0': (
                ['a', 'b', 'c'],
                [1, 2, 3],
                ['products_view', 'products_view', 'products_view'],
                [1.1, 0.1, 1.1]
            )
        }
        self.assertEqual(sample_session.session_items_actions_map, expected_output)

    def test_none_action(self):

        for fs in SESSIONS:
            out_item, out_session = parse_files(fs,
                                                session_id_key=SESSION_KEY,
                                                product_key=PRODUCT_KEY,
                                                time_key=TIME_KEY,
                                                action_key=None,
                                                allowed_actions=None,
                                                purchase_action_name=None,
                                                time_to_numeric=True)

            self.assertIsInstance(out_item, Items)
            self.assertIsInstance(out_session, Sessions)


if __name__ == '__main__':
    unittest.main()
