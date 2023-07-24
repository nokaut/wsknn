from wsknn import parse_files, Items, Sessions


JSONFILE = 'tdata/events1.json'
GZJSONFILE = 'tdata/events1.json.gz'
CSVFILE = 'tdata/events1.csv'
SESSION_KEY = 'sid'
PRODUCT_KEY = 'pid'
ACTION_KEY = 'act'
TIME_KEY = 'ts'
POSSIBLE_ACTIONS = {'products_view': 0.1,
                    'purchase': 1,
                    'add_to_wishlist': 0.3,
                    'add_to_cart': 0.5}
PURCHASE_ACTION_NAME = 'purchase'


def test_parse_json():
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

    assert isinstance(parsed_items, Items)
    assert isinstance(parsed_sessions, Sessions)


def test_parse_csv():
    parsed_items, parsed_sessions = parse_files(
        dataset=CSVFILE,
        session_id_key=SESSION_KEY,
        product_key=PRODUCT_KEY,
        action_key=ACTION_KEY,
        time_key=TIME_KEY,
        time_to_numeric=True,
        allowed_actions=POSSIBLE_ACTIONS,
        purchase_action_name=PURCHASE_ACTION_NAME
    )

    assert isinstance(parsed_items, Items)
    assert isinstance(parsed_sessions, Sessions)


def test_parse_gzip_json():
    parsed_items, parsed_sessions = parse_files(
        dataset=GZJSONFILE,
        session_id_key=SESSION_KEY,
        product_key=PRODUCT_KEY,
        action_key=ACTION_KEY,
        time_key=TIME_KEY,
        time_to_numeric=True,
        allowed_actions=POSSIBLE_ACTIONS,
        purchase_action_name=PURCHASE_ACTION_NAME
    )

    assert isinstance(parsed_items, Items)
    assert isinstance(parsed_sessions, Sessions)
