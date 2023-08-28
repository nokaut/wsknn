from wsknn import Items, Sessions
from wsknn.preprocessing.parse_static import parse_flat_file

CSVFILE = 'tdata/events1.csv'
SESSION_IDX = 3
PRODUCT_IDX = 4
ACTION_IDX = 0
TIME_IDX = 1
POSSIBLE_ACTIONS = {'products_view': 0.1,
                    'purchase': 1,
                    'add_to_wishlist': 0.3,
                    'add_to_cart': 0.5}
PURCHASE_ACTION_NAME = 'purchase'


def test_parse_flat_file():
    parsed_items, parsed_sessions = parse_flat_file(
        dataset=CSVFILE,
        sep=',',
        session_index=SESSION_IDX,
        product_index=PRODUCT_IDX,
        action_index=ACTION_IDX,
        time_index=TIME_IDX,
        use_header_row=True,
        time_to_numeric=True
    )

    assert isinstance(parsed_items, Items)
    assert isinstance(parsed_sessions, Sessions)
