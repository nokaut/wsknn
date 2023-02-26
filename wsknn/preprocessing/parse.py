import gzip
import json
import pathlib
from typing import Dict, Iterable

from wsknn.preprocessing.structure.item import Items
from wsknn.preprocessing.structure.session import Sessions
from wsknn.preprocessing.structure.user import Users
from wsknn.preprocessing.utils.transform import parse_dt_to_seconds


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


def parse(dataset: str,
          allowed_actions: Dict = None,
          purchase_action_name=None) -> (Items, Sessions):
    """
    Function parses data from json and gzip into item-sessions and session-items maps.

    Parameters
    ----------
    dataset : str
        The gzipped JSONL file or JSON file with events.

    allowed_actions : Dict, optional
        Allowed actions and their weights.

    purchase_action_name: Any, optional
        The name of the final action (it is required to apply weight into the session vector).

    Returns
    -------
    ItemsMap, SessionsMap : Items, Sessions
    """

    if dataset.endswith('.gz'):
        items, sessions = parse_gz_fn(dataset, allowed_actions, purchase_action_name)
    elif dataset.endswith('.json') or dataset.endswith('.jsonl'):
        items, sessions = parse_jsonl_fn(dataset, allowed_actions, purchase_action_name)
    else:
        ftype = pathlib.Path(dataset).suffix
        raise TypeError(f'Unrecognized input file type. Parser works with "gz" and "json" files, you have provided '
                        f'{ftype} type.')

    return items, sessions


def parse_gz_fn(dataset: str, possible_actions: Dict, purchase_action_name: str):
    """
    Function parses given gzipped JSONL file into Sessions and Items objects.

    :param dataset: (str) gzipped JSONL file path with events.
    :param possible_actions: (Dict) dict with possible actions and their weights.
    :param purchase_action_name: (str) the name of the final action (it is required to apply weight into the session
                                 vector).

    :return: (Items, Sessions, Users)
    """
    with gzip.open(dataset, 'rt', encoding='UTF-8') as unzipped_f:
        parsed_items, parsed_sessions, parsed_users = _parse_fn(unzipped_f, possible_actions, purchase_action_name)
    return parsed_items, parsed_sessions, parsed_users


def parse_jsonl_fn(dataset: str, possible_actions: Dict, purchase_action_name: str):
    """
    Function parses given JSONL file into Sessions and Items and Users objects.

    :param dataset: (str) JSONL file path with events.
    :param possible_actions: (Dict) dict with possible actions and their weights.
    :param purchase_action_name: (str) the name of the final action (it is required to apply weight into the session
                                 vector).

    :return: (Items, Sessions, Users)
    """
    with open(dataset, 'r', encoding='utf-8') as jsondata:
        try:
            jstream = json.load(jsondata)
        except json.decoder.JSONDecodeError:
            jstream = [json.loads(x) for x in jsondata]

        parsed_items, parsed_sessions, parsed_users = _parse_fn(jstream, possible_actions, purchase_action_name)
    return parsed_items, parsed_sessions, parsed_users


def _parse_fn(datastream,
              possible_actions: Dict,
              purchase_action_name: str) -> (Items, Sessions, Users):
    """
    Function parses given JSONL file into Sessions and Items and Users objects.

    :param datastream: (stream) JSONL file path with events.
    :param possible_actions: (Dict) dict with possible actions and their weights.
    :param purchase_action_name: (str) the name of the final action (it is required to apply weight into the session
                                 vector).

    :return: (Items, Sessions, Users)
    """

    # Initialize Items and Sessions

    items_obj = Items(event_session_key='sid',
                      event_product_key='pid',
                      event_time_key='time')
    sessions_obj = Sessions(event_session_key='sid',
                            event_product_key='pid',
                            event_time_key='time',
                            event_action_key='action')
    users_obj = Users(event_session_key='sid', event_time_key='time', event_user_key='cid')

    for event in datastream:
        event_and_products = parse_event_parameters(event)
        # Check if params are returned
        if event_and_products:
            parsed_event = event_and_products[0]
            parsed_products = event_and_products[1]

            if len(parsed_products) == 1 and parsed_event['action'] != purchase_action_name:
                # Check how many products are returned
                parsed_event['pid'] = parsed_products[0]

                # Is session user interaction?
                possible_actions_list = list(possible_actions.keys())
                if is_user_item_interaction(parsed_event['action'], possible_actions_list):
                    # Append Event to Items and Sessions
                    items_obj.append(parsed_event)
                    sessions_obj.append(parsed_event)
                    users_obj.append(parsed_event)
            else:
                # It is a purchase, update weights accordingly
                purchase_additive_factor = possible_actions[purchase_action_name]
                sessions_obj.update_weights(parsed_event['sid'], parsed_products, purchase_additive_factor)

    return items_obj, sessions_obj, users_obj
