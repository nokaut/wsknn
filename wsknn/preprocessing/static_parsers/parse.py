from typing import Dict, Iterable

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
             datetime_format: str) -> (Items, Sessions):
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

    possible_actions_list = list(allowed_actions.keys())

    for event in dataset:
        event = check_event_keys_and_values(event,
                                            session_id_key,
                                            product_key,
                                            action_key,
                                            time_key)
        # Check if params are returned
        if event:
            action = event[action_key]

            # parse times
            if time_to_numeric or time_to_datetime:
                event[time_key] = clean_time(times=event[time_key],
                                             time_to_numeric=time_to_numeric,
                                             time_to_datetime=time_to_datetime,
                                             datetime_format=datetime_format)

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
