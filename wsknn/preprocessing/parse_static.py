import pathlib
from typing import Dict, Tuple

from wsknn.preprocessing.static_parsers.csv_parser.parse import parse_csv_fn
from wsknn.preprocessing.static_parsers.flat_file_parser.parse import parse_flat_file_fn
from wsknn.preprocessing.static_parsers.json_parser.parse import parse_gzipped_fn, parse_jsonl_fn
from wsknn.preprocessing.structure.item import Items
from wsknn.preprocessing.structure.session import Sessions


def parse_files(dataset: str,
                session_id_key: str,
                product_key: str,
                time_key: str,
                action_key: str = None,
                time_to_numeric=False,
                time_to_datetime=False,
                datetime_format='',
                allowed_actions: Dict = None,
                purchase_action_name=None,
                progress_bar: bool = False) -> Tuple[Items, Sessions]:
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

    progress_bar : bool, default = False
        Show parsing progress.

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
                                           datetime_format,
                                           progress_bar)
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
                                         datetime_format,
                                         progress_bar)
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
                                       datetime_format,
                                       progress_bar)
    else:
        ftype = pathlib.Path(dataset).suffix
        raise TypeError(f'Unrecognized input file type. Parser works with "gz" (gzipped json), "json", and "csv"'
                        f'files, you have provided {ftype} type.')

    return items, sessions


def parse_flat_file(dataset: str,
                    sep: str,
                    session_index: int,
                    product_index: int,
                    time_index: int,
                    action_index: int = None,
                    use_header_row: bool = False,
                    time_to_numeric=False,
                    time_to_datetime=False,
                    datetime_format='',
                    allowed_actions: Dict = None,
                    purchase_action_name=None,
                    ignore_errors: bool = True):
    """
    Function parses data from flat file into item-sessions and session-items maps.

    Parameters
    ----------
    dataset : str
        Input file.

    sep : str
        Separator used to separate values.

    session_index : int
        The index of the session.

    product_index : int
        The index of the product.

    time_index : int
        The index of the event timestamp.

    action_index : int, optional
        The index of the event action.

    use_header_row : bool, default = False
        Use first row values as a header.

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

    ignore_errors : bool, default=True
        Ignore rows that raise exceptions.

    Returns
    -------
    items, sessions : Items, Sessions
        The mappings of item-session and session-items.
    """

    parsed_items, parsed_sessions = parse_flat_file_fn(
        dataset=dataset,
        sep=sep,
        session_index=session_index,
        product_index=product_index,
        time_index=time_index,
        action_index=action_index,
        use_header_row=use_header_row,
        time_to_numeric=time_to_numeric,
        time_to_datetime=time_to_datetime,
        datetime_format=datetime_format,
        allowed_actions=allowed_actions,
        purchase_action_name=purchase_action_name,
        ignore_errors=ignore_errors
    )

    return parsed_items, parsed_sessions
