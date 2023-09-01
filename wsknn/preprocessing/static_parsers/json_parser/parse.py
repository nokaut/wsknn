import gzip
import json
from typing import Dict

from wsknn.preprocessing.static_parsers.parse import parse_fn


def parse_gzipped_fn(dataset: str,
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
    Function parses given gzipped JSONL file into Sessions and Items objects.

    Parameters
    ----------
    dataset : str
        The gzipped JSONL file with events.

    allowed_actions : Dict, optional
        Allowed actions and their weights.

    purchase_action_name: Any, optional
        The name of the final action (it is required to apply weight into the session vector).

    session_id_key : str
        The name of the session key.

    product_key : str
        The name of the product key.

    action_key : str
        The name of the event action type key.

    time_key : str
        The name of the event timestamp key.

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
    parsed_items, parsed_sessions : Items, Sessions
        The mappings of item-session and session-items.
    """
    with gzip.open(dataset, 'rt', encoding='UTF-8') as unzipped_f:

        try:
            jstream = json.load(unzipped_f)
        except json.decoder.JSONDecodeError:
            jstream = [json.loads(x) for x in unzipped_f]

        parsed_items, parsed_sessions = parse_fn(
            jstream,
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


def parse_jsonl_fn(dataset: str,
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
    Function parses given JSONL file into Sessions and Items and Users objects.

    Parameters
    ----------
    dataset : str
        The JSON file with events.

    allowed_actions : Dict, optional
        Allowed actions and their weights.

    purchase_action_name: Any, optional
        The name of the final action (it is required to apply weight into the session vector).

    session_id_key : str
        The name of the session key.

    product_key : str
        The name of the product key.

    action_key : str
        The name of the event action type key.

    time_key : str
        The name of the event timestamp key.

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
    with open(dataset, 'r', encoding='utf-8') as jsondata:
        try:
            jstream = json.load(jsondata)
        except json.decoder.JSONDecodeError:
            jstream = [json.loads(x) for x in jsondata]

        parsed_items, parsed_sessions = parse_fn(
            jstream,
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
