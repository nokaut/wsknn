import pandas as pd
from wsknn.utils.transform import dataframe_to_session_items_map


INDEX_COL = 'id'
MAIN_COL = 'item'
TIME_COL = 'ts'
WEIGHTS_COL = 'w'
DF = pd.read_json('tdata/session_items_map.json', lines=True, dtype={
    INDEX_COL: str
})


def test_fn():
    mapped = dataframe_to_session_items_map(df=DF,
                                            main_col=MAIN_COL,
                                            time_col=TIME_COL,
                                            index_col=INDEX_COL,
                                            weights_col=WEIGHTS_COL)
    assert isinstance(mapped, dict)
    assert set(mapped.keys()) == set(DF[INDEX_COL].unique())
