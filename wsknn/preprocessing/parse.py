import gzip
import json
import pathlib
from typing import Dict, Iterable

from wsknn.preprocessing.structure.item import Items
from wsknn.preprocessing.structure.session import Sessions


def is_user_item_interaction(e_action: str, allowed_actions: Iterable) -> bool:
    """
    Checks if event has a valid action type.

    Parameters
    ----------
    e_action : str
        The name of the event action.

    allowed_actions : Iterable
        The set, list or array with allowed actions.

    Returns
    -------
    : bool
        ``True`` if ``e_action`` is in the list of allowed interactions.
    """

    if e_action in allowed_actions:
        return True
    return False


def parse_files(dataset: str,
                session_id_key: str,
                product_key: str,
                action_key: str,
                time_key: str,
                allowed_actions: Dict = None,
                purchase_action_name=None) -> (Items, Sessions):
    """
    Function parses data from json and gzip into item-sessions and session-items maps.

    Parameters
    ----------
    dataset : str
        The gzipped JSONL file or JSON file with events.

    session_id_key : str
        The name of the session key.

    product_key : str
        The name of the product key.

    action_key : str
        The name of the event action type key.

    time_key : str
        The name of the event timestamp key.

    allowed_actions : Dict, optional
        Allowed actions and their weights.

    purchase_action_name: Any, optional
        The name of the final action (it is required to apply weight into the session vector).

    Returns
    -------
    ItemsMap, SessionsMap : Items, Sessions
    """

    if dataset.endswith('.gz'):
        items, sessions = parse_gz_fn(dataset,
                                      allowed_actions,
                                      purchase_action_name,
                                      session_id_key,
                                      product_key,
                                      action_key,
                                      time_key)
    elif dataset.endswith('.json') or dataset.endswith('.jsonl'):
        items, sessions = parse_jsonl_fn(dataset,
                                         allowed_actions,
                                         purchase_action_name,
                                         session_id_key,
                                         product_key,
                                         action_key,
                                         time_key)
    else:
        ftype = pathlib.Path(dataset).suffix
        raise TypeError(f'Unrecognized input file type. Parser works with "gz" and "json" files, you have provided '
                        f'{ftype} type.')

    return items, sessions


def parse_gz_fn(dataset: str,
                allowed_actions: Dict,
                purchase_action_name: str,
                session_id_key: str,
                product_key: str,
                action_key: str,
                time_key: str):
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

    Returns
    -------
    ItemsMap, SessionsMap : Items, Sessions
    """
    with gzip.open(dataset, 'rt', encoding='UTF-8') as unzipped_f:
        parsed_items, parsed_sessions = parse_fn(
            unzipped_f,
            allowed_actions,
            purchase_action_name,
            session_id_key,
            product_key,
            action_key,
            time_key)

    return parsed_items, parsed_sessions


def parse_jsonl_fn(dataset: str, allowed_actions: Dict, purchase_action_name: str, session_id_key: str,
                   product_key: str,
                   action_key: str,
                   time_key: str):
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
            time_key)

    return parsed_items, parsed_sessions


def check_event_keys_and_values(event, session_id_key: str,
                                product_key: str,
                                action_key: str,
                                time_key: str):
    """
    Function checks if event has its all keys.

    Parameters
    ----------
    event : Dict

    session_id_key : str

    product_key : str

    action_key : str

    time_key : str

    Returns
    -------
    : Dict
        Empty dict if there is a missing key.
    """

    keys = {session_id_key, product_key, action_key, time_key}

    if not keys.issubset(set(event.keys())):
        return {}

    return event


def parse_fn(dataset,
             allowed_actions: Dict,
             purchase_action_name: str,
             session_id_key: str,
             product_key: str,
             action_key: str,
             time_key: str) -> (Items, Sessions):
    """
    Function parses given JSONL file into Sessions and Items objects.

    Parameters
    ----------
    dataset : Iterable
        Object with events.

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

    Returns
    -------
    ItemsMap, SessionsMap : Items, Sessions
    """

    # Initialize Items and Sessions

    items_obj = Items(event_session_key=session_id_key,
                      event_product_key=product_key,
                      event_time_key=time_key)
    sessions_obj = Sessions(event_session_key=session_id_key,
                            event_product_key=product_key,
                            event_time_key=time_key,
                            event_action_key=action_key)

    for event in dataset:
        event = check_event_keys_and_values(event,
                                            session_id_key,
                                            product_key,
                                            action_key,
                                            time_key)
        # Check if params are returned
        if event:
            products = event[product_key]  # List[List, List, List...]
            action = event[action_key]

            if len(products) == 1 and action != purchase_action_name:
                # Check how many products are returned
                event[product_key] = products[0]

                # Is session user interaction?
                possible_actions_list = list(allowed_actions.keys())
                if is_user_item_interaction(action, possible_actions_list):
                    # Append Event to Items and Sessions
                    items_obj.append(event)
                    sessions_obj.append(event)
            else:
                # It is a purchase, update weights accordingly
                purchase_additive_factor = allowed_actions[purchase_action_name]
                sessions_obj.update_weights(event[session_id_key], products, purchase_additive_factor)

    return items_obj, sessions_obj
