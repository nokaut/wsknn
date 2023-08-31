from typing import Dict, Iterable


def check_event_keys_and_values(event: Dict,
                                session_id_key: str,
                                product_key: str,
                                time_key: str,
                                action_key: str = None):
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

    keys = {session_id_key, product_key, time_key}

    if action_key is not None:
        keys.add(action_key)

    if not keys.issubset(set(event.keys())):
        return {}

    return event


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
