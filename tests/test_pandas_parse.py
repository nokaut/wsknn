from typing import Dict

import numpy as np
import pandas as pd

from wsknn.preprocessing.static_parsers.pandas_parser import parse_pandas


df = pd.read_csv('tdata/ml-25m-test/ratings.csv')

_ = parse_pandas(
    df=df,
    session_id_key='userId',
    product_key='movieId',
    time_key='timestamp'
)

possible_actions = ['ORDER', 'VIEW']
actions_probas = [0.05, 0.95]

actions = np.random.choice(possible_actions, size=len(df), p=actions_probas)
df['action'] = actions

def test_parsing():
    parsed = parse_pandas(
        df=df,
        session_id_key='userId',
        product_key='movieId',
        time_key='timestamp',
        action_key='action',
        get_items_map=False
    )

    assert isinstance(parsed, Dict)
    assert parsed.get('session-map')
    assert not parsed.get('item-map')

    parsed = parse_pandas(
        df=df,
        session_id_key='userId',
        product_key='movieId',
        time_key='timestamp',
        action_key='action',
        allowed_actions=['ORDER']
    )

    assert isinstance(parsed, Dict)
    assert parsed.get('session-map')
    assert parsed.get('item-map')

    parsed = parse_pandas(
        df=df,
        session_id_key='userId',
        product_key='movieId',
        time_key='timestamp',
        action_key='action',
        purchase_action_name='ORDER',
        min_session_length=3
    )

    assert isinstance(parsed, Dict)
    assert parsed.get('session-map')
    assert parsed.get('item-map')
