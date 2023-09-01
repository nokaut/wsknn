from typing import Dict
import csv

from wsknn.preprocessing.static_parsers.parse import parse_fn


def parse_csv_fn(dataset: str,
                 allowed_actions: Dict,
                 purchase_action_name: str,
                 session_id_key: str,
                 product_key: str,
                 action_key: str,
                 time_key: str,
                 time_to_numeric: bool,
                 time_to_datetime: bool,
                 datetime_format: str,
                 progress_bar: bool):
    """
    Function parses given CSV file into Sessions and Items and Users objects.

    Parameters
    ----------
    dataset : str
        The JSON file with events.

    allowed_actions : Dict, optional
        Allowed actions and their weights.

    purchase_action_name: Any, optional
        The name of the final action (it is required to apply weight into the session vector).

    session_id_key : str
        The name of the session key (column header).

    product_key : str
        The name of the product key (column header).

    action_key : str
        The name of the event action type key (column header).

    time_key : str
        The name of the event timestamp key (column header).

    time_to_numeric : bool, default = True
        Transforms input timestamps to float values.

    time_to_datetime : bool, default = False
        Transforms input timestamps to datatime objects. Setting ``datetime_format`` parameter is required.

    datetime_format : str
        The format of datetime object.

    progress_bar : bool
        Show parsing progress.

    Returns
    -------
    ItemsMap, SessionsMap : Items, Sessions
    """
    with open(dataset, 'r', encoding='utf-8') as csvdata:
        try:
            csvstream = csv.DictReader(csvdata)
        except Exception as _:
            raise IOError(f'Cannot read {dataset} file!')

        parsed_items, parsed_sessions = parse_fn(
            csvstream,
            allowed_actions,
            purchase_action_name,
            session_id_key,
            product_key,
            action_key,
            time_key,
            time_to_numeric,
            time_to_datetime,
            datetime_format,
            progress_bar
        )

    return parsed_items, parsed_sessions
