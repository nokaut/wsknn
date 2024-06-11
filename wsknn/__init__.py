from wsknn.evaluate import (score_model, get_recall, get_precision,
                            get_mean_reciprocal_rank)
from wsknn.fit_transform import fit, Items, Sessions
from wsknn.predict import predict, batch_predict
from wsknn.model.wsknn import WSKNN
from wsknn.preprocessing.parse_static import parse_files, parse_flat_file
from wsknn.preprocessing.static_parsers.pandas_parser import parse_pandas
from wsknn.utils.transform import dataframe_to_session_items_map, dataframe_to_item_sessions_map


__version__ = '1.2.3'
