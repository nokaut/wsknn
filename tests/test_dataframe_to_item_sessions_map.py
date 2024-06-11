import pandas as pd
from wsknn.utils.transform import dataframe_to_item_sessions_map


INDEX_COL = 'item'
MAIN_COL = 'id'
TIME_COL = 'ts'
DF = pd.read_json('tdata/item_sessions_map.json', lines=True, dtype={
    INDEX_COL: str
})


def test_fn():
    mapped = dataframe_to_item_sessions_map(df=DF, main_col=MAIN_COL, time_col=TIME_COL, index_col=INDEX_COL)
    assert isinstance(mapped, dict)
    assert set(mapped.keys()) == set(DF[INDEX_COL].unique())
