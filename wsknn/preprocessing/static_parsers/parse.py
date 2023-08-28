from typing import Dict, Iterable, IO, Tuple, List
import pandas as pd
from more_itertools import locate
from tqdm import tqdm
from wsknn.preprocessing.static_parsers.checkers.validation import check_event_keys_and_values, is_user_item_interaction
from wsknn.preprocessing.static_parsers.cleaners.time_transform import clean_time
from wsknn.preprocessing.structure.item import Items
from wsknn.preprocessing.structure.session import Sessions


def parse_fn(dataset: Iterable,
             allowed_actions: Dict,
             purchase_action_name: str,
             session_id_key: str,
             product_key: str,
             action_key: str,
             time_key: str,
             time_to_numeric: bool,
             time_to_datetime: bool,
             datetime_format: str,
             progress_bar: bool) -> (Items, Sessions):
    """
    Function parses given dataset into Sessions and Items objects.

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

    time_to_numeric : bool, default = True
        Transforms input timestamps to float values.

    time_to_datetime : bool, default = False
        Transforms input timestamps to datatime objects. Setting `datetime_format` parameter is required.

    datetime_format : str
        The format of datetime object.

    progress_bar : bool
        Show parsing progress.

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
                            event_action_key=action_key,
                            event_action_weights=allowed_actions)

    possible_actions_list = []
    if allowed_actions is not None:
        possible_actions_list = list(allowed_actions.keys())

    for event in (tqdm(dataset, disable=(not progress_bar))):
        event = check_event_keys_and_values(event,
                                            session_id_key,
                                            product_key,
                                            time_key,
                                            action_key)
        # Check if params are returned
        if event:
            # parse times
            if time_to_numeric or time_to_datetime:
                event[time_key] = clean_time(times=event[time_key],
                                             time_to_numeric=time_to_numeric,
                                             time_to_datetime=time_to_datetime,
                                             datetime_format=datetime_format)

            action = event.get(action_key, False)
            if not action:
                items_obj.append(event)
                sessions_obj.append(event)
            else:
                if action != purchase_action_name:
                    # Is session user interaction?
                    if is_user_item_interaction(action, possible_actions_list):
                        # Append Event to Items and Sessions
                        items_obj.append(event)
                        sessions_obj.append(event)
                else:
                    # It is a purchase, update weights accordingly
                    purchase_additive_factor = allowed_actions[purchase_action_name]
                    sessions_obj.update_weights_of_purchase_session(event[session_id_key], purchase_additive_factor)

    return items_obj, sessions_obj


def parse_stream(events: IO,
                 sep: str,
                 allowed_actions: Dict,
                 purchase_action_name: str,
                 session_index: int,
                 product_index: int,
                 time_index: int,
                 action_index: int,
                 time_to_numeric: bool,
                 time_to_datetime: bool,
                 datetime_format: str,
                 ignore_errors: bool = True,
                 header_names: Dict = None,
                 progress_bar: bool = False):
    """
    Function parses given stream of values.

    Parameters
    ----------
    events : IO
        Stream to file.

    sep : str
        Separator between file stream records.

    allowed_actions : Dict, optional
        Allowed actions and their weights.

    purchase_action_name: Any, optional
        The name of the final action (it is required to apply weight into the session vector).

    session_index : int
        The index of the session.

    product_index : int
        The index of the product.

    action_index : int
        The index of the event action.

    time_index : int
        The index of the event timestamp.

    time_to_numeric : bool, default = False
        Transforms input timestamps to float values.

    time_to_datetime : bool, default = False
        Transforms input timestamps to datatime objects. Setting ``datetime_format`` parameter is required.

    datetime_format : str
        The format of datetime object.

    ignore_errors : bool, default=True
        Ignore rows that raise exceptions.

    header_names : List, default = None
        Key names applied to the data.

    progress_bar : bool, default = False
        Show parsing progress.

    Returns
    -------
    items_obj, sessions_obj : Items, Sessions
    """

    # Initialize Items and Sessions

    if header_names is None:
        if action_index is None:
            header_names = {
                session_index: 'session',
                product_index: 'item',
                time_index: 'ts'
            }
        else:
            header_names = {
                session_index: 'session',
                product_index: 'item',
                time_index: 'ts',
                action_index: 'action'
            }

    items_obj = Items(event_session_key=header_names[session_index],
                      event_product_key=header_names[product_index],
                      event_time_key=header_names[time_index])

    if action_index is not None:
        sessions_obj = Sessions(event_session_key=header_names[session_index],
                                event_product_key=header_names[product_index],
                                event_time_key=header_names[time_index],
                                event_action_key=header_names[action_index],
                                event_action_weights=allowed_actions)
    else:
        sessions_obj = Sessions(event_session_key=header_names[session_index],
                                event_product_key=header_names[product_index],
                                event_time_key=header_names[time_index],
                                event_action_key=None,
                                event_action_weights=allowed_actions)

    if allowed_actions is None:
        possible_actions_list = None
    else:
        possible_actions_list = list(allowed_actions.keys())

    for raw_event in (tqdm(events, disable=(not progress_bar))):

        try:
            splitted = raw_event.split(sep)
            splitted = [s.strip() for s in splitted]
        except Exception as ex:
            if ignore_errors:
                continue
            else:
                raise ex

        if action_index is not None:
            event = {
                header_names[session_index]: splitted[session_index],
                header_names[product_index]: splitted[product_index],
                header_names[action_index]: splitted[action_index],
                header_names[time_index]: splitted[time_index]
            }
        else:
            event = {
                header_names[session_index]: splitted[session_index],
                header_names[product_index]: splitted[product_index],
                header_names[time_index]: splitted[time_index]
            }

        event = check_event_keys_and_values(event,
                                            header_names[session_index],
                                            header_names[product_index],
                                            header_names[time_index],
                                            header_names.get(action_index, None))
        # Check if params are returned
        if event:
            # parse times
            if time_to_numeric or time_to_datetime:
                event[header_names[time_index]] = clean_time(times=event[header_names[time_index]],
                                                             time_to_numeric=time_to_numeric,
                                                             time_to_datetime=time_to_datetime,
                                                             datetime_format=datetime_format)

            action = event.get(action_index, False)
            if not action:
                items_obj.append(event)
                sessions_obj.append(event)
            else:
                if action != purchase_action_name:
                    # Is session user interaction?
                    if is_user_item_interaction(action, possible_actions_list):
                        # Append Event to Items and Sessions
                        items_obj.append(event)
                        sessions_obj.append(event)
                else:
                    # It is a purchase, update weights accordingly
                    purchase_additive_factor = allowed_actions[purchase_action_name]
                    sessions_obj.update_weights_of_purchase_session(event[header_names[session_index]],
                                                                    purchase_additive_factor)

    return items_obj, sessions_obj
