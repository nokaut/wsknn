import gzip
import json
import pathlib
from typing import Dict

from preprocessing.core.structure.item import Items
from preprocessing.core.structure.session import Sessions
from preprocessing.core.structure.user import Users
from preprocessing.utils.transform import parse_dt_to_seconds, parse_product_id_from_product_context


def is_user_item_interaction(e_action: str, possible_actions: list) -> bool:
    """
    Checks if event has valid type (CLICK or VIEW).

    :param e_action: (str) event action type.
    :param possible_actions: (list) list of possible actions.


    """

    if e_action in possible_actions:
        return True
    return False


def parse_event_parameters(event: Dict):
    """
    Function parses given event_type, event_category, session_id, product_id and event_time from AS events DB.
        Function returns those parameters in a dict. Empty dict means that event_type, product_id, event_time or
        session_id were not detected.

    :param event: (Dict)
    :return: (Tuple[Dict, List] or False)
    """
    try:
        event_action = event['action']

        ###
        # TODO: control this part of a parser(!) - select from where pid must be parsed
        product_id = parse_product_id_from_product_context(event['product_contexts'])
        # product_id = parse_product_id(event['product']['_id'])
        ###

        event_time = parse_dt_to_seconds(event['time']['$date'])
        session_id = event['session']['_id']
        customer_id = event['session']['customer_id']
    except KeyError:
        return False
    except ValueError:
        return False
    except TypeError as te:
        print(te)
        return False

    if not (event_action and product_id and event_time and session_id):
        return False
    else:
        event_dict = {
            'action': event_action,
            'time': event_time,
            'sid': session_id,
            'cid': customer_id
        }
        products = product_id
        return event_dict, products


def parse(dataset: str,
          possible_actions: Dict,
          purchase_action_name: str) -> (Items, Sessions, Users):
    """
    Function parses data from json and gzip into items, sessions and users objects.

    :param dataset: (str) gzipped JSONL file or JSON file path with events.
    :param possible_actions: (Dict) dict with possible actions and their weights.
    :param purchase_action_name: (str) the name of the final action (it is required to apply weight into the session
                                 vector).

    :return: (Items, Sessions, Users)
    """

    if dataset.endswith('.gz'):
        items, sessions, users = parse_gz_fn(dataset, possible_actions, purchase_action_name)
    elif dataset.endswith('.json') or dataset.endswith('.jsonl'):
        items, sessions, users = parse_jsonl_fn(dataset, possible_actions, purchase_action_name)
    else:
        ftype = pathlib.Path(dataset).suffix
        raise TypeError(f'Unrecognized input file type. Parser works with "gz" and "json" files, you have provided '
                        f'{ftype} type.')

    return items, sessions, users


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
