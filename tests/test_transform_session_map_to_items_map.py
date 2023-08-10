from wsknn import parse_files
from wsknn.preprocessing.structure.session_to_item_map import map_sessions_to_items

JSONFILE = 'tdata/events1.json'
SESSION_KEY = 'sid'
PRODUCT_KEY = 'pid'
ACTION_KEY = 'act'
TIME_KEY = 'ts'
POSSIBLE_ACTIONS = {'products_view': 0.1,
                    'purchase': 1,
                    'add_to_wishlist': 0.3,
                    'add_to_cart': 0.5}
PURCHASE_ACTION_NAME = 'purchase'


def test_get_items_map_from_sessions_map():
    parsed_items, parsed_sessions = parse_files(
        dataset=JSONFILE,
        session_id_key=SESSION_KEY,
        product_key=PRODUCT_KEY,
        action_key=ACTION_KEY,
        time_key=TIME_KEY,
        time_to_numeric=True,
        allowed_actions=POSSIBLE_ACTIONS,
        purchase_action_name=PURCHASE_ACTION_NAME
    )
    sessions_map = parsed_sessions.session_items_actions_map
    items_map = parsed_items.item_sessions_map

    items_from_sessions_map = map_sessions_to_items(sessions_map, sort_items_map=False)

    for original_key, original_mapping in items_map.items():
        assert original_key in items_from_sessions_map
        sessions, timestamps = original_mapping[0], original_mapping[1]
        for idx, sess in enumerate(sessions):
            assert sess in items_from_sessions_map[original_key][0]
            assert timestamps[idx] in items_from_sessions_map[original_key][1]
