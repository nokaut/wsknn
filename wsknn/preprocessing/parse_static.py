import pathlib
from typing import Dict, Tuple

from wsknn.preprocessing.static_parsers.csv_parser.parse import parse_csv_fn
from wsknn.preprocessing.static_parsers.json_parser.parse import parse_gzipped_fn, parse_jsonl_fn
from wsknn.preprocessing.structure.item import Items
from wsknn.preprocessing.structure.session import Sessions


def parse_files(dataset: str,
                session_id_key: str,
                product_key: str,
                action_key: str,
                time_key: str,
                time_to_numeric=False,
                time_to_datetime=False,
                datetime_format='',
                allowed_actions: Dict = None,
                purchase_action_name=None) -> Tuple[Items, Sessions]:
    """
    Function parses data from csv, json and gzip json into item-sessions and session-items maps.

    Parameters
    ----------
    dataset : str
        The gzipped JSONL, JSON, CSV file with events.

    session_id_key : str
        The name of the session key.

    product_key : str
        The name of the product key.

    action_key : str
        The name of the event action type key.

    time_key : str
        The name of the event timestamp key.

    time_to_numeric : bool, default = False
        Transforms input timestamps to float values.

    time_to_datetime : bool, default = False
        Transforms input timestamps to datatime objects. Setting ``datetime_format`` parameter is required.

    datetime_format : str
        The format of datetime object.

    allowed_actions : Dict, optional
        Allowed actions and their weights.

    purchase_action_name: Any, optional
        The name of the final action (it is required to apply weight into the session vector).

    Returns
    -------
    items, sessions : Items, Sessions
        The mappings of item-session and session-items.
    """

    if dataset.endswith('.gz'):
        items, sessions = parse_gzipped_fn(dataset,
                                           allowed_actions,
                                           purchase_action_name,
                                           session_id_key,
                                           product_key,
                                           action_key,
                                           time_key,
                                           time_to_numeric,
                                           time_to_datetime,
                                           datetime_format)
    elif dataset.endswith('.json') or dataset.endswith('.jsonl'):
        items, sessions = parse_jsonl_fn(dataset,
                                         allowed_actions,
                                         purchase_action_name,
                                         session_id_key,
                                         product_key,
                                         action_key,
                                         time_key,
                                         time_to_numeric,
                                         time_to_datetime,
                                         datetime_format)
    elif dataset.endswith('.csv'):
        items, sessions = parse_csv_fn(dataset,
                                       allowed_actions,
                                       purchase_action_name,
                                       session_id_key,
                                       product_key,
                                       action_key,
                                       time_key,
                                       time_to_numeric,
                                       time_to_datetime,
                                       datetime_format)
    else:
        ftype = pathlib.Path(dataset).suffix
        raise TypeError(f'Unrecognized input file type. Parser works with "gz" (gzipped json), "json", and "csv"'
                        f'files, you have provided {ftype} type.')

    return items, sessions
